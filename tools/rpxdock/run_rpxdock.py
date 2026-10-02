#!/usr/bin/env python3
"""
Submit an array (SLURM or Modal) that rigid-body docks each scaffold into a
ONE-COMPONENT symmetric architecture with RPXdock, and mints one child row per
kept pose.

RPXdock (Sheffler et al., PLoS Comput Biol 2022, 10.1371/journal.pcbi.1010680)
enumerates the rigid-body placements compatible with a target point-group symmetry
and scores each one with precomputed residue-pair motif tables ("hscore"): a dock
scores well when the new symmetric interface it creates looks like interfaces seen
in natural structures. It answers a question no other tool here answers -- *given
this monomer, is there a placement that makes a good C2 (or T3, or D3_2) interface,
and where is it?* -- and it answers it by producing NEW STRUCTURES, which is why
this is a ``create`` tool: one scaffold row fans out to N dock rows in a child
table.

PREMISE AND SCOPE -- read this before trusting a number
=======================================================

* **One component only.** v1 drives ``--inputs1`` and nothing else. That covers
  cyclic (``C2``..``C17``, and the ``CxSTACK`` variants) and the one-component
  cages and dihedrals (``T2 T3 O2 O3 O4 I2 I3 I5``, ``Dx_y``). Two- and
  three-component docking (``T32``, ``I32``, ``AXLE_*``, ``PLUG_*``, ``ASYM``,
  the multi-component layers) is NOT wired: those protocols need a second and
  third input list that this tool has no way to express, so they are refused by
  name at manifest-build time rather than failing confusingly in a container.
  One-component LAYERS are refused too, for a source-level reason:
  ``DockSpec1CompLayer`` indexes ``default_lattice_axes``, which holds only
  ``P6_632`` (three components) and ``P4M_4`` (a *mirror* layer, routed to
  ``DockSpec1CompMirrorLayer``) -- so there is no reachable non-mirror
  one-component layer architecture, and the mirror one has no validated dump path
  here.

* **The input must be centred on the origin, and for D*/cage architectures it must
  be the ASYMMETRIC UNIT of a cyclic oligomer, pre-aligned with its own symmetry
  axis on Z** -- not the full oligomer, which RPXdock regenerates itself. Feed a
  whole assembly, or an off-origin monomer, and the dock still runs and still
  reports scores; they are just scores for a placement nobody asked about.
  ``--recenter-input`` fixes the centring (and the collected ``input_com_dist``
  tells you whether it was needed), but NOTHING here can tell you the symmetry
  axis was wrong. That is the premise, and it is on the caller.

* **No PyRosetta.** The image deliberately omits it, which is the configuration
  RPXdock's own CLI defaults to (``--use_rosetta`` is ``store_true``, i.e. False).
  Secondary structure therefore comes from willutil's pure-Python DSSP
  (``wu.dssp`` on the backbone N/CA/C/O), not Rosetta's. The collected
  ``frac_helix``/``frac_sheet``/``frac_loop`` report that assignment.

  NOTE, measured: these fractions are DESCRIPTIVE, not a gate. It was previously
  claimed here that because the default ``ilv_h`` tables are built from HELIX
  PAIRS, a non-helical scaffold "scores ~0 everywhere with no error". That is
  FALSE -- a ``frac_helix`` 0.00 / ``frac_sheet`` 0.65 scaffold scored ``rpx``
  78.90 under ``ilv_h`` (the best in its batch, above an 81%-helix fixture at
  77.64) and 107.74 under ``afilmv_ehl``. The ``_h`` describes how the tables were
  GENERATED; they are applied SS-INDEPENDENTLY. Do not filter on ``frac_helix``.
  An ALL-loop body is still a real problem, but RPXdock refuses that outright at
  ``body.py:151``. What PyRosetta would have added:
  full-atom poses, the helix-termini accessibility options
  (``--term_access*``/``--termini_dir*``), and Rosetta's own DSSP.

* **One chain for a cyclic dock; at most two for a cage or dihedral.** Every
  check below runs at SUBMIT TIME, in the workstation, before a single container
  starts -- because the failure it prevents is not a crash but a ten-minute
  silence. Measured: a 2-chain, 154-residue input sat in willutil's per-atom PDB
  parser (``willutil/pdb/pdbfile.py:295``) for over ten minutes per task, with no
  ``.exit`` file and an empty design dir, before any rigid-body sampling began;
  single-chain inputs of 100 and 119 residues dock in 58-75 s. And it would have
  been wrong anyway: ``Body.set_pose_info`` collapses every chain into one chain
  renumbered from 1, so a 2-chain dimer docked into ``C2`` is one concatenated
  pseudo-chain of double the length. So the limit is per FAMILY:
  **cyclic / cyclic_stack refuse more than 1 chain**, and **the one-component
  cages and dihedrals refuse more than 2** (2 is legitimate: a ``Dx_2`` scaffold
  is a dimer on the 2-fold). Alongside that, the file RPXdock will actually read
  is opened and checked for parseable ``ATOM`` records, the N/CA/C backbone (a
  missing ``O`` is only a warning -- willutil guesses it), a single model, and
  sane residue numbering. See ``rpxdock_structure.py``.

* **Only CIF is converted.** ``.pdb`` and ``.pdb.gz`` go to RPXdock untouched --
  it reads gzipped PDB natively, and its own shipped fixture is
  ``C3_1na0-1_1.pdb.gz``. ``prosapia.utils.ensure_pdb`` short-circuits on
  ``suffix == ".pdb"`` only, so calling it unconditionally round-tripped every
  ``.pdb.gz`` through gemmi for nothing. ``.cif``/``.cif.gz``/``.mmcif`` are
  converted; anything else is a named submit-time error.

* **Motif scores are table lookups, comparable within a batch at one
  ``--hscore-files`` setting and not across settings.** The collected
  ``hscore`` column records which table produced each number so a later reader
  cannot mix two scales by accident.

``default_input_column`` is the literal string ``"not applicable"`` -- the sentinel
``usalign``/``chainsel``/``cms`` use. There is no honest default: the scaffold may
come from ``rfdiffusion3_path``, ``boltz_path`` or ``chainsel_path``, and picking
one would make the others fail *silently* (a column absent from the frame is a
no-op for ``filter_ready``, so every row would look ready and then read a missing
cell). So **-i/--input-column is effectively required**.

Residue selections use the prosapia positions mini-language, resolved per design up
the table lineage, and written out as the whitespace-separated residue file
RPXdock's ``--allowed_residues1`` expects::

    --allowed-residues '20:60'                  # literal, 1-indexed
    --allowed-residues '{helix1_start}:{helix1_end},{helix2_start}:{helix2_end}'

Anything without a dedicated flag goes through ``--set``, forwarded verbatim to
``python -m rpxdock``'s own CLI (``--set '--max_longaxis_dot_z 0.7'``), the same
escape hatch ``proteinmpnn`` offers. One option is deliberately NOT given a flag:
``--score_only_sspair`` raises ``AttributeError: module 'numpy' has no attribute
'bool'`` under the image's ``numpy<2`` pin (``Body.filter_pairs`` uses the alias
numpy removed in 1.24 and restored in 2.0), and that pin is there because numpy 2
breaks the cage/dihedral sampler outright. Reach it through ``--set`` only if you
are prepared for that.

Usage:
    # C2 from a table of rfd3 backbones, 5 docks kept per scaffold
    sapia run rpxdock outputs/RUN -t table0 -i rfdiffusion3_path \\
        --architecture C2 --nout-top 5 -g 0

    # dihedral from pre-aligned C3 asymmetric units, finer search
    sapia run rpxdock outputs/RUN -t table1 -i chainsel_path \\
        --architecture D3_3 --beam-size 300000 --nout-top 3 -g 0
"""

import json
import shlex
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from prosapia.core import CommonArgs, ManifestCtx
from prosapia.core.executors import volume_path
from prosapia.utils import ensure_pdb, parse_positions

from .rpxdock_structure import (
    ChainCountError,
    InputStructureError,
    check_chain_count,
    input_kind,
    scan_structure,
    validate_structure,
)

# Re-exported so a caller (or a test) can catch the submit-time refusals without
# knowing which module defines them. Both are ValueError subclasses.
__all__ = [
    "ArchitectureError",
    "ChainCountError",
    "InputStructureError",
    "NO_DEFAULT_COLUMN",
    "add_run_rpxdock_args",
    "build_rpxdock_manifest",
    "check_input_structure",
    "classify_architecture",
    "parse_set_flags",
    "resolve_input_pdb",
]

# The sentinel default_input_column (see the module docstring): matches no column,
# so the builder can refuse the run with a real message instead of silently
# submitting rows whose input cell does not exist.
NO_DEFAULT_COLUMN = "not applicable"

# Layout this tool imposes on its out_dir. rpxdock_worker.py and
# collect_rpxdock.py read the same convention -- keep the three in step.
CONFIG_DIRNAME = "configs"
RESIDUES_DIRNAME = "allowed_residues"

# The one-component cage/dihedral point groups DockSpec1CompCage accepts
# (rpxdock/search/dockspec.py:41). Checked here so a typo is a submit-time error
# naming the valid set, not an AssertionError in a container.
ONECOMP_CAGE_SYMS = ("T2", "T3", "O2", "O3", "O4", "I2", "I3", "I5")
ONECOMP_DIHEDRAL_SYMS = ("D2", "D3", "D4", "D5", "D6", "D8")

# Cyclic orders RPXdock documents (README docks C3 and C17).
CYCLIC_MIN, CYCLIC_MAX = 2, 17

DOCKING_METHODS = ("hier", "grid")

# rpxdock CLI options this tool owns through a dedicated flag. --set may not
# repeat any of them: two sources for one option is a silent coin-flip.
OWNED_RPX_OPTIONS = frozenset(
    {
        "architecture",
        "inputs",
        "inputs1",
        "inputs2",
        "inputs3",
        "allowed_residues",
        "allowed_residues1",
        "hscore_files",
        "hscore_data_dir",
        "docking_method",
        "max_trim",
        "beam_size",
        "recenter_input",
        "score_only_ss",
        "weight_rpx",
        "weight_ncontact",
        "max_bb_redundancy",
        "use_orig_coords",
        "output_prefix",
        "dump_pdbs",
        "nout_top",
        "nout_each",
        "nout_debug",
        "save_results_as_tarball",
        "save_results_as_pickle",
        "suppress_dump_results",
        "overwrite_existing_results",
        "dont_store_body_in_results",
        "use_rosetta",
    }
)


class ArchitectureError(ValueError):
    """An architecture this tool refuses: multi-component, or malformed. Raised at
    manifest-build time so the run never reaches a container."""


class RpxdockArgs(CommonArgs):
    architecture: str
    nout_top: int
    hscore_files: str
    hscore_data_dir: str
    docking_method: str
    max_trim: int
    beam_size: int
    recenter_input: bool
    score_only_ss: str
    weight_rpx: float
    weight_ncontact: float
    max_bb_redundancy: float
    allowed_residues: str
    use_orig_coords: bool
    set: list[str]


def add_run_rpxdock_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        help="Symmetric architecture to dock into, in RPXdock's own spelling. "
        "ONE-COMPONENT architectures only: 'C2'..'C17' (and 'CxSTACK'), the "
        "one-component cages "
        + " ".join(ONECOMP_CAGE_SYMS)
        + ", and the dihedrals 'Dx_y' with x in "
        + " ".join(s[1] for s in ONECOMP_DIHEDRAL_SYMS)
        + " and y either 2 or x (e.g. 'D3_2', 'D3_3'). Multi-component "
        "architectures (T32, O43, I32, ASYM, AXLE_*, PLUG_*, layers) need a "
        "second/third input list this tool does not wire, and are refused by name "
        "at submit time. Cages and dihedrals require the input to be the "
        "ASYMMETRIC UNIT of a cyclic oligomer with its symmetry axis on Z.",
    )
    parser.add_argument(
        "--nout-top",
        type=int,
        default=10,
        help="Docks to keep per scaffold -- the number of CHILD ROWS one parent row "
        "produces, best-scoring first. Default 10. Fewer may come back: RPXdock "
        "filters redundant poses (--max-bb-redundancy) before this cut, so a "
        "scaffold with few distinct good placements yields fewer rows, and one with "
        "none yields none. There is deliberately no --nout-each: this tool docks "
        "exactly one scaffold per task, so RPXdock's 'top across all docks' and "
        "'top for each dock' are the same set.",
    )
    parser.add_argument(
        "--hscore-files",
        type=str,
        default="ilv_h",
        help="Motif-table set to score with (RPXdock's --hscore_files): an alias "
        "naming a subdirectory of --hscore-data-dir. "
        "THE ALIAS MUST MATCH THE SECONDARY STRUCTURE OF THE DESIGNS -- see the "
        "rpxdock skill, which records the current campaign's choice and when to "
        "revisit it. The default 'ilv_h' is HELIX-ORIENTED (ILV residues, helix "
        "pairs only, SS-independent, ~365 MB) and is the wrong table set for "
        "beta-containing designs; for those pass 'afilmv_ehl' (5.70 GB, "
        "SS-DEPENDENT, all SS types -- raise --mem well above 16G and expect a long "
        "per-task table load). Also on the Volume but DO NOT USE, UNTESTED: "
        "'ailv_h' (adds A; ~1.4 GB -- 475 MB of .txz plus 923 MB of .txz.pickle "
        "sidecars, which take precedence and may not unpickle here). "
        "SCORES ARE NOT COMPARABLE ACROSS ALIASES -- the collected "
        "'hscore' column records which one produced each number. "
        "'small_ilv_h' is a 860 KB TEST fixture shipped inside the package; use it "
        "with --hscore-data-dir pointed at the package's data/hscore dir for a smoke "
        "test only, never for a reported score.",
    )
    parser.add_argument(
        "--hscore-data-dir",
        type=str,
        default="/rpxdock_files",
        help="Directory the --hscore-files alias is looked up under. Default "
        "'/rpxdock_files', which is RPXdock's own default AND the mount point of "
        "the 'rpxdock-hscore' Modal Volume, so the default needs no flag on Modal.",
    )
    parser.add_argument(
        "--docking-method",
        type=str,
        choices=DOCKING_METHODS,
        default="hier",
        help="Search strategy: 'hier' (default) is RPXdock's hierarchical search, "
        "coarse-to-fine with a beam; 'grid' is a flat enumeration at "
        "--set '--grid_resolution_cart_angstroms N' resolution. 'hier' is what the "
        "paper uses and what --beam-size tunes.",
    )
    parser.add_argument(
        "--max-trim",
        type=int,
        default=0,
        help="Residues RPXdock may trim from a terminus to relieve a clash "
        "(RPXdock's --max_trim). Default 0, which disables trimming entirely and is "
        "markedly faster. When >0 the surviving range is collected as reslb/resub "
        "and the dumped PDB contains only that range, so the dock structure may be "
        "SHORTER than the scaffold -- check reslb/resub before comparing lengths.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=100_000,
        help="Samples carried into each stage of the hierarchical search after the "
        "first (RPXdock's --beam_size). Default 100000. The main runtime knob: "
        "raising it searches more thoroughly and costs proportionally more. Ignored "
        "by --docking-method grid.",
    )
    parser.add_argument(
        "--recenter-input",
        action="store_true",
        help="Translate the input so its backbone centre of mass sits at the origin "
        "before docking. RPXdock does NOT centre inputs by default and its sampling "
        "assumes an origin-centred body, so an off-origin scaffold is docked in a "
        "frame nobody intended. Turning this on fixes that, but the transforms "
        "RPXdock reports are then relative to the RECENTRED pose -- if you plan to "
        "use those transforms, pre-centre the input instead. The collected "
        "'input_com_dist' (measured on the file, before any recentring) and "
        "'recentered' say which regime a row is in.",
    )
    parser.add_argument(
        "--score-only-ss",
        type=str,
        default="EHL",
        help="Restrict scoring to residues of these secondary-structure types "
        "(RPXdock's --score_only_ss; any of E, H, L). Default 'EHL' = no "
        "restriction. Note the default ilv_h TABLES already contain helix pairs "
        "only, so 'H' is usually redundant with them and this flag matters most "
        "with the SS-dependent afilmv_ehl set.",
    )
    parser.add_argument(
        "--weight-rpx",
        type=float,
        default=1.0,
        help="Weight of the motif (rpx) term in the combined score "
        "(RPXdock's --weight_rpx). Default 1.0. The collected 'score' is the "
        "weighted combination; 'rpx' and 'ncontact' are each reported unweighted "
        "alongside it, so changing a weight does not make the components "
        "unreadable.",
    )
    parser.add_argument(
        "--weight-ncontact",
        type=float,
        default=0.01,
        help="Weight of the contact-count term in the combined score "
        "(RPXdock's --weight_ncontact). Default 0.01.",
    )
    parser.add_argument(
        "--max-bb-redundancy",
        type=float,
        default=3.0,
        help="Minimum backbone separation (roughly a non-aligned RMSD, A) between "
        "two kept docks of one scaffold (RPXdock's --max_bb_redundancy). Default "
        "3.0. This is what decides how DIFFERENT the --nout-top rows are from each "
        "other: lower it to keep near-duplicates, raise it for more diverse docks "
        "and fewer rows.",
    )
    parser.add_argument(
        "--allowed-residues",
        type=str,
        default="",
        help="Residues allowed to form the docked interface, in the prosapia "
        "positions mini-language (1-indexed, ',' between fragments, 'start:end' "
        "inclusive, '{column}' islands resolved per design up the lineage). Empty "
        "(default) allows every residue. Written out as the whitespace-separated "
        "residue file RPXdock's --allowed_residues1 reads. Use it to keep a motif, "
        "a binding site or a terminus out of the new symmetric interface. NOTE the "
        "numbering is the position IN THE FILE (1..N), not an author residue number "
        "-- RPXdock renumbers every input from 1 internally.",
    )
    parser.add_argument(
        "--use-orig-coords",
        action="store_true",
        help="Write the input's ORIGINAL atoms into the dumped docks. Off by "
        "default, matching RPXdock: a dumped dock is then a reduced "
        "N/CA/C/O/CB + CEN representation (CEN is a pseudo-atom, not a real one), "
        "which is fine for a backbone-level look and for redesign, but is NOT an "
        "all-atom structure and will mislead any tool that counts atoms or packs "
        "side chains. Turn this on when the docks feed something that needs the "
        "real side chains.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FLAG",
        help="Raw flag forwarded verbatim to `python -m rpxdock` (repeatable), for "
        "the ~200 options without a dedicated flag here. E.g. "
        "--set '--max_longaxis_dot_z 0.7', --set '--max_cluster 1000', "
        "--set '--grid_resolution_cart_angstroms 2'. Tokenised with shlex, so "
        "quoting works. A flag already owned by a dedicated option above is "
        "refused: two sources for one option is a silent coin-flip.",
    )


def classify_architecture(architecture: str) -> tuple[str, int]:
    """``'C2' -> ('cyclic', 2)``; ``'D3_2' -> ('onecomp', 2)``; ``'T3' -> ('onecomp', 3)``.

    Mirrors ``rpxdock/app/dock.py:main``'s dispatch and ``DockSpec1CompCage``'s
    assertions, so that an architecture this tool cannot drive is refused HERE,
    with a message naming what it would have needed, instead of dying inside a
    container on an AssertionError or on a missing ``--inputs2``.

    Returns ``(protocol, nfold)`` where protocol is ``'cyclic'``, ``'cyclic_stack'``
    or ``'onecomp'`` and nfold is the symmetry order of the axis being docked
    about. Raises ``ArchitectureError`` for anything else.
    """
    arch = architecture.strip().upper()
    if not arch:
        raise ArchitectureError("--architecture is empty.")

    def _multi(what: str) -> ArchitectureError:
        return ArchitectureError(
            f"--architecture {arch!r} is a {what} architecture. This tool is "
            f"ONE-COMPONENT only: it drives RPXdock's --inputs1 and has no way to "
            f"express the second (and third) input list those protocols need. "
            f"Supported: C2..C17 (and CxSTACK), the one-component cages "
            f"{' '.join(ONECOMP_CAGE_SYMS)}, and the dihedrals Dx_y "
            f"(x in {' '.join(s[1] for s in ONECOMP_DIHEDRAL_SYMS)}, y = 2 or x)."
        )

    if arch == "ASYM":
        raise _multi("two-monomer asymmetric")
    if arch.startswith("PLUG"):
        raise _multi("plug (monomer into a fixed oligomer)")
    if arch.startswith("AXLE_"):
        raise _multi("two-component axle")

    if arch.startswith("C"):
        stack = arch.endswith("STACK")
        base = arch[: -len("STACK")] if stack else arch
        try:
            nfold = int(base[1:])
        except ValueError:
            raise ArchitectureError(
                f"--architecture {arch!r} looks cyclic but {base[1:]!r} is not an "
                f"integer; write e.g. 'C2', 'C3', 'C17' (or 'C3STACK')."
            ) from None
        if not CYCLIC_MIN <= nfold <= CYCLIC_MAX:
            raise ArchitectureError(
                f"--architecture {arch!r}: cyclic order {nfold} is outside the "
                f"range RPXdock is used over here (C{CYCLIC_MIN}..C{CYCLIC_MAX})."
            )
        return ("cyclic_stack" if stack else "cyclic"), nfold

    if len(arch) == 2 or (arch[0] == "D" and len(arch) > 2 and arch[2] == "_"):
        if arch[:2] not in ONECOMP_CAGE_SYMS + ONECOMP_DIHEDRAL_SYMS:
            raise ArchitectureError(
                f"--architecture {arch!r} is not a one-component point group "
                f"RPXdock knows. One-component cages: "
                f"{' '.join(ONECOMP_CAGE_SYMS)}; dihedrals: Dx_y with x in "
                f"{' '.join(s[1] for s in ONECOMP_DIHEDRAL_SYMS)}."
            )
        if arch[0] == "D":
            # DockSpec1CompCage requires the Dx_y form explicitly: a bare 'D3'
            # does not say which axis the scaffold sits on.
            if len(arch) != 4 or arch[2] != "_":
                raise ArchitectureError(
                    f"--architecture {arch!r}: a dihedral must be written 'Dx_y', "
                    f"where y is the symmetry of the SCAFFOLD -- 2 (a dimer on a "
                    f"2-fold) or x (an x-mer on the main axis). E.g. 'D3_2' or "
                    f"'D3_3'."
                )
            nfold = int(arch[3])
            if nfold not in (2, int(arch[1])):
                raise ArchitectureError(
                    f"--architecture {arch!r}: for D{arch[1]} the scaffold symmetry "
                    f"y must be 2 or {arch[1]}, got {nfold}."
                )
            return "onecomp", nfold
        return "onecomp", int(arch[1])

    if arch.startswith("F"):
        raise _multi("discrete n-sided (multi-component)")

    if arch.startswith("P"):
        raise ArchitectureError(
            f"--architecture {arch!r} is a layer/wallpaper architecture, which this "
            f"tool does not support. RPXdock's one-component layer spec indexes "
            f"`default_lattice_axes`, which holds only P6_632 (three components) "
            f"and P4M_4 (a MIRROR layer, handled by a different sampler) -- so "
            f"there is no reachable non-mirror one-component layer, and the mirror "
            f"one has no validated structure-dump path here."
        )

    raise _multi("two-component cage")


def parse_set_flags(tokens: list[str]) -> list[str]:
    """``['--max_cluster 1000']`` -> ``['--max_cluster', '1000']``, guarding
    against a flag this tool already owns through a dedicated option."""
    argv: list[str] = []
    for token in tokens:
        parts = shlex.split(token)
        if not parts:
            continue
        if not parts[0].startswith("--"):
            raise ValueError(
                f"--set {token!r} must start with a long flag, e.g. "
                f"--set '--max_longaxis_dot_z 0.7'."
            )
        option = parts[0].lstrip("-")
        if option in OWNED_RPX_OPTIONS:
            raise ValueError(
                f"--set {token!r} sets --{option}, which this tool already controls "
                f"through its own flag (or owns outright for bookkeeping). Setting "
                f"it twice means whichever RPXdock sees last wins, silently. Use "
                f"the dedicated flag."
            )
        argv.extend(parts)
    return argv


def write_allowed_residues(path: Path, positions: list[list[int]]) -> None:
    """Write RPXdock's --allowed_residues1 file: whitespace-separated 1-indexed
    residue numbers.

    The positions mini-language speaks in per-chain groups, but a Body is a single
    chain by construction (``Body.set_pose_info`` sets ``chain`` to a constant and
    renumbers 0..N-1), so several groups would be meaningless here -- hence the
    single-group requirement enforced by the caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(p) for p in positions[0]) + "\n")


def resolve_input_pdb(src: Path, run_dir: Path, design: str = "") -> Path:
    """The file that will actually be handed to RPXdock, converting only if needed.

    ``.pdb`` and ``.pdb.gz`` are returned unchanged: RPXdock reads gzipped PDB
    natively (its own fixture is ``C3_1na0-1_1.pdb.gz``), and
    ``prosapia.utils.ensure_pdb`` would otherwise round-trip a ``.pdb.gz`` through
    gemmi because its ``suffix`` is ``".gz"``, not ``".pdb"``. mmCIF forms go
    through ``ensure_pdb`` as before -- willutil's reader has no mmCIF path at all.
    Any other suffix raises :class:`InputStructureError` naming the accepted forms.

    Pure except for the conversion itself, so the routing decision is testable via
    :func:`rpxdock_structure.input_kind` with no gemmi and no run_dir.
    """
    if input_kind(src, design) == "pdb":
        return src
    return ensure_pdb(src, run_dir)


def check_input_structure(
    path: Path, architecture: str, protocol: str, design: str = ""
) -> list[str]:
    """Validate the file RPXdock will read; return the warnings to print.

    A single pure function over a file path, so it can be exercised with a
    handwritten PDB and no image: it scans the file once
    (:func:`rpxdock_structure.scan_structure`), refuses an unusable structure
    (:func:`~rpxdock_structure.validate_structure` -- no ATOM records, several
    models, a broken N/CA/C backbone, nonsense residue numbering) and refuses a
    chain count this architecture family cannot mean
    (:func:`~rpxdock_structure.check_chain_count` -- >1 chain for cyclic, >2 for a
    cage or dihedral).

    Raises ``InputStructureError`` or ``ChainCountError``. Warnings (currently only
    a missing backbone ``O``) are returned, not raised, and the caller prints them
    to stdout so they land in the submit log.
    """
    scan = scan_structure(path)
    warnings = validate_structure(scan, design)
    check_chain_count(scan, architecture, protocol, design)
    return warnings


def build_rpxdock_manifest(ctx: ManifestCtx[RpxdockArgs]) -> list[tuple[str, ...]]:
    args = ctx.args
    args.gpus_per_task = 0  # CPU-only tool -- callers need no -g 0

    # Refuse an unsupported architecture before anything else: it is the one
    # mistake that would otherwise burn a whole array.
    protocol, nfold = classify_architecture(args.architecture)
    architecture = args.architecture.strip().upper()

    extra_argv = parse_set_flags(args.set)

    if args.nout_top < 1:
        raise ValueError(
            f"--nout-top must be at least 1 (got {args.nout_top}); a run that keeps "
            f"no dock would mint an empty child table."
        )

    # No honest default input column (see module docstring): refuse loudly rather
    # than submit rows whose input cell does not exist.
    column = args.input_column
    if column == NO_DEFAULT_COLUMN or column not in ctx.df.columns:
        available = ", ".join(
            str(c) for c in ctx.df.columns if str(c).endswith("_path")
        )
        raise ValueError(
            f"rpxdock has no default input column: pass -i/--input-column with the "
            f"structure column holding the scaffold to dock (got {column!r}, which "
            f"table '{args.table}' does not have). Structure columns available: "
            f"{available or '(none)'}."
        )

    config_dir = ctx.out_dir / CONFIG_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)

    # Run-wide rpxdock argv. Per-design pieces (--inputs1, --output_prefix,
    # --allowed_residues1) are appended by the worker from its config.
    common_argv: list[str] = [
        "--architecture",
        architecture,
        "--hscore_files",
        args.hscore_files,
        "--hscore_data_dir",
        args.hscore_data_dir,
        "--docking_method",
        args.docking_method,
        "--max_trim",
        str(args.max_trim),
        "--beam_size",
        str(args.beam_size),
        "--score_only_ss",
        args.score_only_ss.upper(),
        "--weight_rpx",
        str(args.weight_rpx),
        "--weight_ncontact",
        str(args.weight_ncontact),
        "--max_bb_redundancy",
        str(args.max_bb_redundancy),
        # Bookkeeping this tool owns: the worker dumps structures itself from the
        # in-memory Result (see rpxdock_worker.py), so RPXdock's own pdb dumping
        # stays off, and a rerun must not be refused for an existing tarball.
        "--save_results_as_tarball",
        "True",
        "--save_results_as_pickle",
        "False",
        "--overwrite_existing_results",
    ]
    if args.recenter_input:
        common_argv.append("--recenter_input")
    if args.use_orig_coords:
        common_argv.append("--use_orig_coords")
    common_argv += extra_argv

    ready = ctx.ready
    manifest_rows: list[tuple[str, ...]] = []
    for name in ready.index:
        name = cast(str, name)
        src = Path(str(ready.at[name, column]))
        if not src.exists():
            print(f"{name}: MISSING {src} (skipping)")
            continue
        # PDB (plain or gzipped) goes through untouched; mmCIF is converted, cached
        # under run_dir/.cif_to_pdb. Anything else raises.
        input_pdb = resolve_input_pdb(src, args.run_dir, name)

        # Check the file RPXdock will ACTUALLY read, here, in the workstation.
        # A structural problem or a chain count the architecture cannot mean is
        # caller misuse of this tool's premise, so it raises and takes the whole
        # run with it -- the same treatment as a multi-group --allowed-residues,
        # and the opposite of a missing input file (skipped above), which is a
        # gap in the table rather than a wrong instruction. Raising is the point:
        # the alternative is ten silent minutes per task inside willutil.
        for warning in check_input_structure(input_pdb, architecture, protocol, name):
            print(warning)

        residues_file = ""
        if args.allowed_residues.strip():
            try:
                positions = parse_positions(args.allowed_residues, ctx.lookup, name)
            except ValueError as e:
                # One unresolvable row shouldn't sink the array: warn and skip.
                print(f"{name}: {e} (skipping)")
                continue
            if len(positions) != 1:
                raise ValueError(
                    f"--allowed-residues names {len(positions)} chain groups "
                    f"(separated by '/'), but RPXdock treats an input as a single "
                    f"chain renumbered from 1. Give one group."
                )
            if not positions[0]:
                print(f"{name}: --allowed-residues resolved to no residue (skipping)")
                continue
            path = ctx.out_dir / RESIDUES_DIRNAME / f"{name}.res"
            write_allowed_residues(path, positions)
            residues_file = str(volume_path(path))

        config = {
            "name": name,
            "architecture": architecture,
            "protocol": protocol,
            "nfold": nfold,
            "nout_top": args.nout_top,
            "hscore": args.hscore_files,
            "recentered": bool(args.recenter_input),
            "allowed_residues_file": residues_file,
            "argv": common_argv,
        }
        config_json = config_dir / f"{name}.json"
        config_json.write_text(json.dumps(config, indent=2))

        manifest_rows.append(
            (
                name,
                str(volume_path(input_pdb)),
                str(volume_path(config_json)),
            )
        )

    return manifest_rows
