#!/usr/bin/env python3
"""
Extract a named subset of chains from one structure into a new structure file,
optionally merging several source chains into one output chain.

This is the per-array-task step of the chainsel tool (and works standalone for a
single structure). It exists because every structure comparison in the workbench is
whole-file: ``usalign --mm 1`` on a binder complex is dominated by the large fixed
target and reads ~1.0 whatever the binder did, and ``--ter 2`` isolates only the
FIRST chain while a binder is the LAST. Writing the binder chain to its own file
makes binder-to-binder comparison (``usalign --mm 0``) possible.

The second job is the SPLIT PROTOMER: a target trimmed of a membrane belt is given
to a predictor as two entities per protomer, so the prediction holds one protomer as
two chains. ``--groups 'A+C:A,B+D:B'`` writes those back as one chain per protomer,
which is what ``ringfit --target-chains A,B`` needs before it can report
``bsa_t1``/``bsa_t2``/``bridge_ratio`` at all.

The one rule that matters: **a requested chain that is not in the structure is an
error, not a shorter file.** A silently partial extraction would produce a perfectly
well-formed PDB that every downstream number (TM-score, RMSD, BSA, energy) would then
be computed on -- wrong, plausible, and undetectable. So a missing chain is reported
as data (``status`` = ``error: chain E not in <file> (have: A,B,C)``) and no output
file is written. That holds for every chain named in a merge group too.

RESIDUE ORDER inside a merged chain is the order the group names its sources -- all
of A's residues, then all of C's -- and is never re-sorted by residue number: two
segments of a split protomer carry independent numbering from the predictor (each
1..N), so sorting would interleave them catastrophically.

NUMBERING is exactly one of three regimes:

* default: source residue numbers and insertion codes are preserved. If a MERGED
  chain then holds a duplicated (number, icode), that design is an ERROR -- a file
  whose residue numbers repeat inside one chain silently loses residues in any
  consumer that indexes by number (``ringfit --resnum-match resnum`` does);
* ``--renumber``: each OUTPUT chain is renumbered 1..N in written order, insertion
  codes dropped -- continuous across a merge seam;
* ``--renumber-from 'A:19-59,C:107-155'``: each SOURCE chain's segment is renumbered
  consecutively from its own first number, insertion codes dropped. When a ``-last``
  is given it is checked against the residues actually written, so a segment of the
  wrong length is an error rather than a quietly shifted numbering. It assumes each
  segment is internally gapless.

Reads PDB or mmCIF transparently (gemmi). Alternative conformations and hydrogens are
dropped on read, so the extraction is deterministic regardless of what the source
modelled. Only the FIRST model is used: a multi-model file (NMR ensemble, some
predictors' output) would otherwise write a chain several times over.

``--strip-het`` (the default; ``--keep-het`` disables it) keeps only residues gemmi
tabulates as amino acids or nucleic acids, which drops waters, ions and ligands while
keeping modified polymer residues such as MSE.

Writes a one-row TSV (name, status, path, n_chains, n_res, chains, n_atoms) that
collect_chainsel.py merges back into the table. ``n_chains`` counts OUTPUT chains (so
a merge lowers it), while ``n_res``/``n_atoms`` count everything written (so a merge
does not change them). Errors are recorded as data (a status starting with 'error:')
rather than only crashing, so partial array runs still collect.

Normally invoked per array task by the chainsel tool -- run ``sapia run chainsel``
rather than calling this directly.

Usage (standalone, single structure):
    python tools/chainsel/chainsel_worker.py \\
        --name design_0 --src design_0.cif --chains E --rename-to A \\
        --out design_0_binder.pdb --result-tsv design_0.tsv

    python tools/chainsel/chainsel_worker.py \\
        --name design_0 --src design_0.cif \\
        --groups 'A+C:A,B+D:B,E:E' \\
        --renumber-from 'A:19-59,C:107-155,B:19-59,D:107-155' \\
        --out design_0_merged.pdb --result-tsv design_0.tsv
"""

import argparse
import csv
import sys
from pathlib import Path

import gemmi

# Metric columns of the per-design TSV, in order (bare names: collect_chainsel.py
# hands them to the driver, which leaf-prefixes them to chainsel_<name>).
METRIC_COLUMNS = ["n_chains", "n_res", "chains", "n_atoms"]
RESULT_COLUMNS = ["name", "status", "path", *METRIC_COLUMNS]

# One output chain: source chain IDs in write order, and the ID written.
Group = tuple[list[str], str]


def split_list(value: str) -> list[str]:
    """Comma-joined list -> stripped, non-empty tokens."""
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def parse_groups(value: str) -> list[Group]:
    """'A+C:A,B:B' -> [(['A', 'C'], 'A'), (['B'], 'B')]."""
    groups: list[Group] = []
    for spec in split_list(value):
        if spec.count(":") != 1:
            raise ValueError(
                f"--groups entry {spec!r} must be '<chain>[+<chain>...]:<out_id>'"
            )
        sources_str, out_id = (tok.strip() for tok in spec.split(":"))
        sources = [tok.strip() for tok in sources_str.split("+") if tok.strip()]
        if not sources or not out_id:
            raise ValueError(f"--groups entry {spec!r} is missing a chain or an id")
        groups.append((sources, out_id))
    if not groups:
        raise ValueError("--groups is empty")
    return groups


def resolve_groups(groups: str, chains: str, rename_to: str) -> list[Group]:
    """The selection as output groups.

    ``--groups`` is what chainsel.sh passes (the manifest carries the canonical
    form). ``--chains``/``--rename-to`` are kept for standalone use and become
    groups of one.
    """
    if groups.strip():
        return parse_groups(groups)
    kept = split_list(chains)
    renamed = split_list(rename_to)
    if not kept:
        raise ValueError("--chains is empty")
    if renamed and len(renamed) != len(kept):
        raise ValueError(
            f"--rename-to has {len(renamed)} IDs but --chains has {len(kept)}"
        )
    out_ids = renamed or kept
    return [([chain], out_id) for chain, out_id in zip(kept, out_ids)]


def parse_starts(value: str) -> dict[str, tuple[int, int]]:
    """'A:19-59,C:107' -> {'A': (19, 59), 'C': (107, 0)} (0 = no end given)."""
    starts: dict[str, tuple[int, int]] = {}
    for spec in split_list(value):
        if spec.count(":") != 1:
            raise ValueError(
                f"--renumber-from entry {spec!r} must be '<chain>:<first>[-<last>]'"
            )
        chain, rng = (tok.strip() for tok in spec.split(":"))
        first_str, _, last_str = rng.partition("-")
        first = int(first_str)
        last = int(last_str) if last_str else 0
        starts[chain] = (first, last)
    return starts


def load_structure(path: Path) -> gemmi.Structure:
    """Read a PDB or mmCIF file into a single-conformer, hydrogen-free structure."""
    if not path.exists():
        raise FileNotFoundError(f"structure missing: {path}")
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if len(st) == 0:
        raise ValueError(f"no model in {path}")
    return st


def is_polymer_residue(res: gemmi.Residue) -> bool:
    """True for a tabulated amino-acid or nucleic-acid residue.

    Used as the --strip-het test instead of the HETATM flag, which would throw away
    modified polymer residues (MSE and friends are HETATM records but are part of
    the chain).
    """
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info and (info.is_amino_acid() or info.is_nucleic_acid()))


def chain_residues(
    model: gemmi.Model, chain_id: str, strip_het: bool
) -> list[gemmi.Residue]:
    """Residues of every chain object named ``chain_id``, in file order.

    A PDB split by TER, or an mmCIF whose waters hang off the polymer's auth chain,
    can present one chain name as several chain objects; they are concatenated here
    rather than the first one silently winning.
    """
    residues: list[gemmi.Residue] = []
    for chain in model:
        if chain.name != chain_id:
            continue
        for res in chain:
            if strip_het and not is_polymer_residue(res):
                continue
            residues.append(res)
    return residues


def check_unique_numbering(chain: gemmi.Chain, sources: list[str], src: Path) -> None:
    """A merged chain must not repeat a (residue number, insertion code).

    Two segments of a split protomer each arrive numbered 1..N, so preserving the
    source numbering across a merge collides. The collision is invisible in the
    file itself but silently drops residues in any consumer that indexes residues
    by number -- ringfit's ``--resnum-match resnum`` builds exactly such an index.
    """
    keys = [f"{res.seqid.num}{str(res.seqid.icode).strip()}" for res in chain]
    seen: set[str] = set()
    dups: list[str] = []
    for key in keys:
        if key in seen and key not in dups:
            dups.append(key)
        seen.add(key)
    if dups:
        shown = ",".join(dups[:5]) + ("..." if len(dups) > 5 else "")
        raise ValueError(
            f"merged chain {chain.name} of {src.name} ({'+'.join(sources)}) has "
            f"duplicate residue numbers ({shown}): the segments carry independent "
            f"numbering. Pass --renumber (1..N across the merged chain) or "
            f"--renumber-from '<chain>:<first>[-<last>],...' (auth numbering per "
            f"segment)"
        )


def extract(
    src: Path,
    groups: list[Group],
    strip_het: bool,
    renumber: bool,
    starts: dict[str, tuple[int, int]],
) -> tuple[gemmi.Structure, list[str], int, int]:
    """One structure -> (extracted structure, written chain IDs, n_res, n_atoms).

    ``n_res``/``n_atoms`` count everything written, over all output chains; the
    number of output chains is ``len(groups)``, which a merge makes smaller than
    the number of source chains.

    Raises on anything unusable (the caller records the message as data): a chain
    that is absent, one that survives HET stripping with no residues left, a
    segment whose length contradicts its --renumber-from range, or a merged chain
    whose residue numbers collide.
    """
    st = load_structure(src)
    model = st[0]
    present = [chain.name for chain in model]

    out_model = gemmi.Model(1)
    out_ids: list[str] = []
    n_res = 0
    n_atoms = 0

    for sources, out_id in groups:
        out_chain = gemmi.Chain(out_id)
        for chain_id in sources:
            if chain_id not in present:
                raise ValueError(
                    f"chain {chain_id} not in {src.name} (have: {','.join(present)})"
                )
            residues = chain_residues(model, chain_id, strip_het)
            if not residues:
                hint = (
                    " after stripping HETATM/ligands/waters (--keep-het)"
                    if strip_het
                    else ""
                )
                raise ValueError(
                    f"chain {chain_id} of {src.name} has no residues{hint}"
                )

            first = None
            if starts:
                if chain_id not in starts:
                    raise ValueError(
                        f"--renumber-from gives no start for kept chain {chain_id}"
                    )
                first, last = starts[chain_id]
                if last and len(residues) != last - first + 1:
                    raise ValueError(
                        f"chain {chain_id} of {src.name} has {len(residues)} "
                        f"residues but --renumber-from says {first}-{last} "
                        f"({last - first + 1})"
                    )

            # Written in the order the group lists its sources, never re-sorted
            # by residue number.
            for i, res in enumerate(residues):
                copy = res.clone()
                if first is not None:
                    copy.seqid = gemmi.SeqId(first + i, " ")
                elif renumber:
                    # Continuous across the whole output chain, so a merged
                    # A+C comes out 1..90 rather than 1..41 then 1..49.
                    copy.seqid = gemmi.SeqId(len(out_chain) + 1, " ")
                out_chain.add_residue(copy)
                n_atoms += len(copy)
            n_res += len(residues)

        if len(sources) > 1:
            check_unique_numbering(out_chain, sources, src)
        out_model.add_chain(out_chain)
        out_ids.append(out_id)

    if n_res == 0:
        raise ValueError(f"nothing extracted from {src.name}")

    out_st = gemmi.Structure()
    out_st.name = src.stem
    out_st.spacegroup_hm = st.spacegroup_hm
    out_st.cell = st.cell
    out_st.add_model(out_model)
    out_st.setup_entities()
    return out_st, out_ids, n_res, n_atoms


def write_structure(st: gemmi.Structure, out: Path, out_format: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out_format == "cif":
        st.make_mmcif_document().write_file(str(out))
    else:
        st.write_pdb(str(out))


def write_result(
    result_tsv: Path, name: str, status: str, path: str, metrics: dict
) -> None:
    """One-row TSV. A metric that did not apply is written as NA (an empty cell),
    never as 0 -- 0 chains and 0 residues are real, alarming values."""
    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(result_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(RESULT_COLUMNS)
        writer.writerow(
            [name, status, path]
            + [
                "" if metrics.get(c) is None else str(metrics[c])
                for c in METRIC_COLUMNS
            ]
        )


class Args(argparse.Namespace):
    name: str
    src: Path
    chains: str
    rename_to: str
    groups: str
    out_format: str
    keep_het: bool
    renumber: bool
    renumber_from: str
    out: Path
    result_tsv: Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract (and optionally merge) a subset of chains."
    )
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--src", type=Path, required=True, help="Source structure (PDB or mmCIF)."
    )
    ap.add_argument(
        "--groups",
        default="",
        help="Canonical selection '<chain>[+<chain>...]:<out_id>', comma-joined, "
        "e.g. 'A+C:A,B+D:B'. What chainsel.sh passes; supersedes "
        "--chains/--rename-to.",
    )
    ap.add_argument(
        "--chains",
        default="",
        help="Chain IDs to keep, comma-joined and ordered (standalone use).",
    )
    ap.add_argument(
        "--rename-to",
        default="",
        help="New chain IDs, same order and number as --chains. Empty keeps them.",
    )
    ap.add_argument("--out-format", choices=("pdb", "cif"), default="pdb")
    ap.add_argument(
        "--keep-het",
        action="store_true",
        help="Keep HETATM/ligands/waters. Default strips them.",
    )
    ap.add_argument(
        "--renumber",
        action="store_true",
        help="Renumber each OUTPUT chain from 1, continuous across a merge seam. "
        "Default preserves the numbering.",
    )
    ap.add_argument(
        "--renumber-from",
        default="",
        help="Per-source-chain auth numbering '<chain>:<first>[-<last>]', "
        "comma-joined, e.g. 'A:19-59,C:107-155'. Excludes --renumber.",
    )
    ap.add_argument(
        "--out", type=Path, required=True, help="Extracted structure to write."
    )
    ap.add_argument(
        "--result-tsv", type=Path, required=True, help="Per-design result TSV to write."
    )
    args = ap.parse_args(namespace=Args())

    try:
        if args.renumber and args.renumber_from.strip():
            raise ValueError("--renumber and --renumber-from are mutually exclusive")
        groups = resolve_groups(args.groups, args.chains, args.rename_to)
        starts = parse_starts(args.renumber_from)
        st, out_ids, n_res, n_atoms = extract(
            args.src, groups, not args.keep_het, args.renumber, starts
        )
        write_structure(st, args.out, args.out_format)
        metrics = {
            "n_chains": len(out_ids),
            "n_res": n_res,
            "chains": ",".join(out_ids),
            "n_atoms": n_atoms,
        }
        print(
            f"{args.name}: wrote {args.out.name} -- {len(out_ids)} chain(s) "
            f"{','.join(out_ids)}, {n_res} residues, {n_atoms} atoms"
        )
        write_result(args.result_tsv, args.name, "OK", str(args.out), metrics)
    except Exception as e:  # noqa: BLE001 - errors are recorded as data
        print(f"{args.name}: ERROR {e}", file=sys.stderr)
        # No partial file is left behind: a half-written extraction would be read
        # by the next tool as if it were complete.
        try:
            args.out.unlink(missing_ok=True)
        except OSError:
            pass
        write_result(args.result_tsv, args.name, f"error: {e}", "", {})


if __name__ == "__main__":
    main()
