#!/usr/bin/env python3
"""
Pure-stdlib reading and validation of the structure file RPXdock will be handed.

Shared by ``run_rpxdock.py`` (submit time, in the workstation) and
``rpxdock_worker.py`` (task time, in the slim image). It is deliberately
**stdlib-only** -- no numpy, no gemmi, no biopython -- for two reasons: the rpxdock
image is a slim Debian carrying RPXdock and nothing else, and these functions must
be importable and testable on their own, without the image and without prosapia.

WHY IT EXISTS
=============
RPXdock's input path goes through willutil's ``NotPose`` PDB reader, and that
reader has two properties that turn a bad input into an expensive silence rather
than an error:

* **It is slow, per atom, and it does its work BEFORE docking.** A two-chain input
  was measured to sit in ``willutil/pdb/pdbfile.py:295`` (``df.ri.iloc[i]`` in a
  per-atom pandas loop) for **over ten minutes** with no output, no ``.exit`` file
  and an empty design dir, before a single rigid-body placement was sampled. A
  whole array spent its wall clock there. Single-chain inputs of 100 and 119
  residues dock end to end in 58-75 s.
* **It never complains about chains.** ``Body.set_pose_info`` sets ``chain`` to a
  constant and ``resno`` to ``arange(n)``, so every chain of the input is collapsed
  into ONE chain renumbered from 1. A multi-chain input is therefore docked as one
  long concatenated chain and scored happily -- the score is for a scaffold nobody
  asked about. Docking a 2-chain dimer into ``C2`` is the clearest case: it becomes
  a single pseudo-chain of double the length, which is not a C2 dock of anything.

So the checks here run at **submit time**, in the workstation, before any container
starts: seconds, with a named error, instead of ten minutes per design.

SCOPE -- what these functions do NOT do
=======================================
* They read **PDB only** (optionally gzipped). mmCIF is converted upstream by
  ``prosapia.utils.ensure_pdb`` and the *converted* file is what gets scanned.
* They read the **first model only** (everything up to the first ``ENDMDL``) for
  coordinates, exactly as RPXdock's own reader does; later models are counted so a
  multi-model file can be refused, not silently truncated.
* They look at ``ATOM`` records only. ``HETATM`` is ignored, so a ligand, a
  modified residue written as HETATM, or waters are invisible here -- this is not a
  completeness check on the chemistry, only on the protein backbone RPXdock scores.
* They say nothing about whether the symmetry axis is where the caller thinks it
  is, nor whether the scaffold is the asymmetric unit of the right oligomer. That
  is the tool's premise and it stays on the caller.
"""

import gzip
from dataclasses import dataclass, field
from pathlib import Path

# Suffixes RPXdock reads natively. `.pdb.gz` is in here deliberately: RPXdock's own
# shipped fixture is `C3_1na0-1_1.pdb.gz`, and routing it through a CIF converter
# would be a pointless round trip through gemmi.
PDB_SUFFIXES = (".pdb", ".pdb.gz")
# Suffixes that must be converted to PDB before RPXdock sees them: willutil's
# NotPose has no mmCIF reader at all.
CIF_SUFFIXES = (".cif", ".cif.gz", ".mmcif")

# Backbone atoms willutil needs. N/CA/C are load-bearing; O is not, because
# willutil fills a missing one with `wu.chem.add_bb_o_guess` -- so its absence is a
# warning here, never an error.
REQUIRED_BACKBONE = ("N", "CA", "C")
OPTIONAL_BACKBONE = ("O",)

# Largest gap between two consecutive residue numbers of one chain that is still
# read as a plain loop gap rather than a broken file. Deliberately generous: real
# structures have gaps of tens of residues, and author numbering sometimes jumps
# into the hundreds. Anything past this is a concatenation or a parsing accident.
MAX_RESNUM_JUMP = 1000

# Chains allowed in the input, BY ARCHITECTURE FAMILY (see `check_chain_count`).
# Cyclic takes a single-chain monomer and nothing else: docking a dimer into C2
# would concatenate it into one pseudo-chain of double the length. The
# cage/dihedral families allow 2 because a `Dx_2` scaffold IS a dimer sitting on
# the 2-fold.
CYCLIC_PROTOCOLS = ("cyclic", "cyclic_stack")
MAX_CHAINS_CYCLIC = 1
MAX_CHAINS_ONECOMP = 2

# How many offending items a message lists before it says "and N more".
_MAX_LISTED = 5


class InputStructureError(ValueError):
    """The input file is not something RPXdock can meaningfully dock: an
    unsupported format, no parseable ATOM records, several models, a missing
    backbone atom, or nonsense residue numbering. Raised at manifest-build time so
    the run never reaches a container."""


class ChainCountError(ValueError):
    """The input has more chains than this tool will dock into the requested
    architecture. Raised at manifest-build time: RPXdock would concatenate them
    into one renumbered chain and score a scaffold nobody asked about -- after
    spending ten minutes in willutil's parser getting there."""


@dataclass
class StructureScan:
    """One pass over a PDB file, holding everything the checks and the trust
    metrics need. Built by :func:`scan_structure`; a plain value object, so it can
    be constructed by hand in a test."""

    path: str
    #: number of ``MODEL`` records (1 when the file has none but does have atoms)
    n_models: int = 0
    #: ``ATOM`` records in the first model
    n_atom_records: int = 0
    #: distinct residues (chain + resseq + icode) in the first model
    n_residues: int = 0
    #: distinct chain IDs, in the order they first appear
    chain_order: tuple[str, ...] = ()
    #: CA coordinates of the first model, in file order
    ca_coords: tuple[tuple[float, float, float], ...] = ()
    #: chain ID of each CA in ``ca_coords``
    ca_chains: tuple[str, ...] = ()
    #: every distinct atom name seen in the first model
    atom_names: frozenset[str] = frozenset()
    #: backbone atom -> residue labels lacking it, e.g. ``{"O": ("A:12", ...)}``
    missing_backbone: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: human-readable residue-numbering complaints, one per problem
    numbering_problems: tuple[str, ...] = ()
    #: 1-based line numbers of ``ATOM`` records that could not be parsed
    malformed_lines: tuple[int, ...] = ()

    @property
    def n_chains(self) -> int:
        return len(self.chain_order)


def _label(design: str) -> str:
    """``"d3: "`` or ``""`` -- so every message can name the design when it has
    one, and the functions stay usable standalone when it does not."""
    return f"{design}: " if design else ""


def _listed(items: tuple[str, ...] | list[str]) -> str:
    head = ", ".join(str(i) for i in items[:_MAX_LISTED])
    extra = len(items) - _MAX_LISTED
    return f"{head} and {extra} more" if extra > 0 else head


def _lower_name(path: Path) -> str:
    return path.name.lower()


def input_kind(path: Path, design: str = "") -> str:
    """``"pdb"`` (hand the file to RPXdock as-is) or ``"cif"`` (convert first).

    This is the whole routing decision, as a pure function over a filename, so the
    caller's conversion step can be tested without gemmi:

    =================  ========================================================
    ``.pdb``           ``"pdb"`` -- used directly
    ``.pdb.gz``        ``"pdb"`` -- used directly; RPXdock reads gzip natively
    ``.cif``           ``"cif"`` -- converted
    ``.cif.gz``        ``"cif"`` -- converted
    ``.mmcif``         ``"cif"`` -- converted
    anything else      ``InputStructureError``
    =================  ========================================================

    Note what the second row fixes: ``prosapia.utils.ensure_pdb`` short-circuits
    only on ``suffix == ".pdb"``, and a ``.pdb.gz``'s suffix is ``".gz"`` -- so it
    would round-trip a perfectly good gzipped PDB through gemmi for nothing.
    """
    name = _lower_name(path)
    for suffix in PDB_SUFFIXES:
        if name.endswith(suffix):
            return "pdb"
    for suffix in CIF_SUFFIXES:
        if name.endswith(suffix):
            return "cif"
    raise InputStructureError(
        f"{_label(design)}{path} is not a structure format rpxdock accepts. "
        f"Accepted: {', '.join(PDB_SUFFIXES)} (handed to RPXdock as they are -- it "
        f"reads gzipped PDB natively), and {', '.join(CIF_SUFFIXES)} (converted to "
        f"PDB at submit time). Point -i/--input-column at a structure column, not "
        f"at a sequence, a FASTA or a results file."
    )


def open_structure_text(path: Path):
    """Open a PDB for text reading, transparently handling ``.gz``."""
    if _lower_name(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def scan_structure(path: Path) -> StructureScan:
    """One pass over a PDB (optionally gzipped), collecting everything the checks
    and the trust metrics need.

    Coordinates come from the **first model only** -- the scan stops recording at
    the first ``ENDMDL``, which is what RPXdock's own reader effectively does --
    but the pass continues to the end of the file so later ``MODEL`` records can be
    counted and refused rather than silently dropped.

    Raises nothing about content: it reports. :func:`validate_structure` decides
    what is fatal.
    """
    n_models = 0
    n_atom_records = 0
    in_first_model = True
    chain_order: list[str] = []
    seen_chains: set[str] = set()
    ca_coords: list[tuple[float, float, float]] = []
    ca_chains: list[str] = []
    atom_names: set[str] = set()
    malformed: list[int] = []

    # (chain, resseq, icode) -> atom names, in first-seen order.
    residues: dict[tuple[str, int, str], set[str]] = {}
    # Residues whose atoms are not contiguous in the file: a residue that is
    # returned to after another residue intervened. The dict above would hide it.
    revisited: list[str] = []
    last_key: tuple[str, int, str] | None = None

    with open_structure_text(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.startswith("MODEL "):
                n_models += 1
                continue
            if line.startswith("ENDMDL"):
                in_first_model = False
                continue
            if not line.startswith("ATOM"):
                continue
            if not in_first_model:
                continue
            if len(line) < 54:
                malformed.append(lineno)
                continue
            atom = line[12:16].strip()
            chain = line[21]
            icode = line[26].strip()
            try:
                resseq = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                malformed.append(lineno)
                continue

            n_atom_records += 1
            atom_names.add(atom)
            if chain not in seen_chains:
                seen_chains.add(chain)
                chain_order.append(chain)
            key = (chain, resseq, icode)
            if key in residues and key != last_key:
                label = f"{chain}:{resseq}{icode}"
                if label not in revisited:
                    revisited.append(label)
            last_key = key
            residues.setdefault(key, set()).add(atom)
            if atom == "CA":
                ca_coords.append(xyz)
                ca_chains.append(chain)

    if n_models == 0 and n_atom_records:
        # A file with no MODEL records but with atoms is one model.
        n_models = 1

    backbone = REQUIRED_BACKBONE + OPTIONAL_BACKBONE
    missing: dict[str, list[str]] = {a: [] for a in backbone}
    for (chain, resseq, icode), atoms in residues.items():
        label = f"{chain}:{resseq}{icode}"
        for atom in backbone:
            if atom not in atoms:
                missing[atom].append(label)

    return StructureScan(
        path=str(path),
        n_models=n_models,
        n_atom_records=n_atom_records,
        n_residues=len(residues),
        chain_order=tuple(chain_order),
        ca_coords=tuple(ca_coords),
        ca_chains=tuple(ca_chains),
        atom_names=frozenset(atom_names),
        missing_backbone={a: tuple(v) for a, v in missing.items() if v},
        numbering_problems=_numbering_problems(list(residues), revisited),
        malformed_lines=tuple(malformed),
    )


def _numbering_problems(
    keys: list[tuple[str, int, str]], revisited: list[str] | None = None
) -> tuple[str, ...]:
    """Residue numbering complaints, in file order.

    Within one chain the residue numbers must not go backwards and must not jump
    by more than :data:`MAX_RESNUM_JUMP`. Both failures mean the same thing in
    practice: the file is not one continuous chain of residues, so the renumbered
    1..N positions ``--allowed-residues`` speaks in -- and the concatenation
    RPXdock builds -- do not mean what the caller thinks.
    """
    problems: list[str] = [
        f"residue {label} is written in two non-adjacent blocks of ATOM records"
        for label in (revisited or [])
    ]
    last: dict[str, tuple[int, str]] = {}
    for chain, resseq, icode in keys:
        prev = last.get(chain)
        last[chain] = (resseq, icode)
        if prev is None:
            continue
        prev_seq, prev_icode = prev
        if resseq < prev_seq:
            problems.append(
                f"chain {chain!r}: residue numbering goes backwards "
                f"({prev_seq}{prev_icode} then {resseq}{icode})"
            )
        elif resseq - prev_seq > MAX_RESNUM_JUMP:
            problems.append(
                f"chain {chain!r}: residue numbering jumps {prev_seq} -> {resseq} "
                f"({resseq - prev_seq} > {MAX_RESNUM_JUMP})"
            )
    return tuple(problems)


def validate_structure(scan: StructureScan, design: str = "") -> list[str]:
    """Refuse a file RPXdock cannot meaningfully dock; return non-fatal warnings.

    Raises :class:`InputStructureError` for: no parseable ``ATOM`` records,
    unparseable ``ATOM`` records, more than one model, a residue missing ``N``,
    ``CA`` or ``C``, or residue numbering that goes backwards / jumps absurdly.

    Returns a list of warning strings (currently only about a missing ``O``, which
    willutil guesses with ``add_bb_o_guess``). The caller prints them so they land
    in the submit log.
    """
    where = f"{_label(design)}{scan.path}"

    if scan.n_atom_records == 0:
        raise InputStructureError(
            f"{where} contains no parseable ATOM records in its first model. "
            f"RPXdock reads the backbone from ATOM lines only -- a file that is "
            f"empty, HETATM-only, or not a PDB at all looks exactly like this. "
            f"Check that the input column points at the structure you meant."
        )

    if scan.n_models > 1:
        raise InputStructureError(
            f"{where} holds {scan.n_models} models. RPXdock docks ONE rigid body "
            f"and reads only the first model, so the other {scan.n_models - 1} "
            f"would be silently dropped and the score would be for model 1 alone. "
            f"Split the models and dock the one you mean."
        )

    if scan.malformed_lines:
        raise InputStructureError(
            f"{where} has {len(scan.malformed_lines)} ATOM record(s) that do not "
            f"parse as fixed-width PDB (line "
            f"{_listed([str(n) for n in scan.malformed_lines])}). A truncated or "
            f"hand-edited PDB reaches willutil's parser and produces coordinates "
            f"nobody can account for."
        )

    broken = {a: v for a, v in scan.missing_backbone.items() if a in REQUIRED_BACKBONE}
    if broken:
        detail = "; ".join(
            f"{atom} missing from {len(res)} residue(s) ({_listed(res)})"
            for atom, res in sorted(broken.items())
        )
        raise InputStructureError(
            f"{where} is missing required backbone atoms: {detail}. RPXdock scores "
            f"residue pairs off the N/CA/C/O backbone and derives secondary "
            f"structure from it (willutil's pure-Python DSSP, no PyRosetta), so a "
            f"CA-only or gapped-backbone file either crashes in body.py "
            f"('body is all loops and not sub-body!!') or scores ~0 everywhere "
            f"with no error. O is the one backbone atom that may be absent -- "
            f"willutil guesses it."
        )

    if scan.numbering_problems:
        raise InputStructureError(
            f"{where} has unusable residue numbering: "
            f"{_listed(scan.numbering_problems)}. RPXdock renumbers the whole "
            f"input 1..N internally, so a file whose residues are out of order or "
            f"split across blocks is concatenated in FILE order -- the positions "
            f"--allowed-residues speaks in, and the chain RPXdock docks, are then "
            f"not the ones you counted."
        )

    warnings: list[str] = []
    if missing_o := scan.missing_backbone.get("O"):
        warnings.append(
            f"WARNING {where} has {len(missing_o)} residue(s) with no backbone O "
            f"({_listed(missing_o)}). Not fatal: willutil fills it with "
            f"wu.chem.add_bb_o_guess. It does mean the secondary structure, and "
            f"therefore the motif score, is partly computed on guessed atoms."
        )
    return warnings


def check_chain_count(
    scan: StructureScan,
    architecture: str,
    protocol: str,
    design: str = "",
) -> None:
    """Refuse an input with more chains than this architecture family can mean.

    The limit is per family, because the same chain count means different things:

    * **cyclic / cyclic_stack (``C2``..``C17``, ``CxSTACK``) -> at most 1 chain.**
      Cyclic docking requires a single-chain MONOMER. RPXdock collapses every chain
      into one chain renumbered from 1 (``Body.set_pose_info``), so a 2-chain dimer
      docked into ``C2`` becomes one concatenated pseudo-chain of double the
      length -- not a dock of the dimer, and not a dock anyone asked for.
    * **onecomp (``Dx_y`` and the cages ``T2 T3 O2 O3 O4 I2 I3 I5``) -> at most 2
      chains.** 2 is legitimate here: a ``Dx_2`` scaffold is a dimer sitting on the
      2-fold.

    Raises :class:`ChainCountError` naming the design, the observed chain count and
    the architecture. Returns ``None``: there is no warning case left -- the one
    that used to warn (2 chains on a cyclic architecture) is now refused outright.
    """
    where = f"{_label(design)}{scan.path}"
    n = scan.n_chains
    chains = ", ".join(repr(c) for c in scan.chain_order)
    # The measured evidence that makes this a submit-time error rather than a
    # run-time surprise. Repeated in both messages because each is read alone.
    hang = (
        "Refused at submit time rather than in the container because a 2-chain, "
        "154-residue input was measured to hang for over 10 minutes in willutil's "
        "per-atom PDB parser (willutil/pdb/pdbfile.py:295) before any docking "
        "began, producing no .exit file at all; single-chain 100- and 119-residue "
        "inputs dock in 58-75 s."
    )

    if protocol in CYCLIC_PROTOCOLS:
        if n > MAX_CHAINS_CYCLIC:
            raise ChainCountError(
                f"{where} has {n} chains ({chains}), but --architecture "
                f"{architecture} is cyclic and cyclic docking requires a "
                f"SINGLE-CHAIN MONOMER. RPXdock collapses every chain of its input "
                f"into one chain renumbered from 1 (Body.set_pose_info), so a "
                f"{n}-chain input docked into {architecture} becomes one "
                f"concatenated pseudo-chain of {n}x the length -- it is not the "
                f"{architecture} dock you asked for, and it would still come back "
                f"with a plausible score. Extract one protomer with `chainsel` and "
                f"dock that. {hang}"
            )
        return

    if n > MAX_CHAINS_ONECOMP:
        raise ChainCountError(
            f"{where} has {n} chains ({chains}), but --architecture "
            f"{architecture} takes at most {MAX_CHAINS_ONECOMP}. RPXdock collapses "
            f"every chain of its input into one chain renumbered from 1 "
            f"(Body.set_pose_info), so it would dock one long concatenation rather "
            f"than your {n} protomers, and report a plausible score for it. Feed "
            f"the asymmetric unit, pre-aligned with its symmetry axis on Z -- "
            f"RPXdock regenerates the oligomer itself. Two chains are allowed here "
            f"only because a Dx_2 scaffold is a dimer sitting on the 2-fold. Use "
            f"`chainsel` to extract what you mean. {hang}"
        )


def read_ca_atoms(path: Path) -> tuple[list[tuple[float, float, float]], list[str]]:
    """CA coordinates and chain IDs of the first model (``.gz`` handled).

    Kept as a named function because ``rpxdock_worker.py`` computes
    ``input_com_dist`` and ``n_chains_in`` from exactly these two lists, and those
    are collected columns -- the parsing rule behind them must not drift. Plain
    Python lists, not numpy, so this module stays importable anywhere.
    """
    scan = scan_structure(path)
    return list(scan.ca_coords), list(scan.ca_chains)
