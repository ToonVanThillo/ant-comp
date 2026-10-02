#!/usr/bin/env python3
"""
Per-design post-step for hbdesigner: turn one design dir into the one-row-per-rank
TSV that collect_hbdesigner.py folds into the table.

Three jobs, in order of how much they matter:

1. RECORD A RUN THAT PRODUCED NOTHING. ``rank_and_save`` returns early -- writing
   no PDB and no CSV -- when no network survives scoring or symmetrization, and
   the process still exits 0. Left alone that reads downstream as a design with no
   children, i.e. as nothing at all. This writes a single rank-0 row with an
   ``error:`` status instead, so the table says so.

2. TRANSLATE THE NETWORK INTO PROTEINMPNN'S COORDINATES. HBDesigner reports its
   network as PDB chain+resnum+aa (``A12S:A16T``). ProteinMPNN's
   ``--fixed-positions`` counts 1..L WITHIN each parsed chain. Those agree only
   when the PDB happens to be numbered 1..L per chain. The mapping is therefore
   computed from the output PDB's own residue order, and reported with it:

     * ``resnum_offset``   per network chain, ``index - resnum`` when constant
                           over that chain, else ``<chain>:var``
     * ``resnum_shift_max`` max |index - resnum| over the network residues; 0 means
                           the resnums in ``network`` ARE the mpnn positions
     * a hard check that the residue sitting at each mapped position really is the
       amino acid HBDesigner said it was -- a mismatch is an ``error:`` status, not
       a plausible-looking number.

3. VERIFY THE GRAFT. When chains were omitted from design they come back
   poly-glycine and the task script runs upstream's graft_seq.py. This reports
   whether that ran (``grafted``) and how much of the grafted chains now matches
   the input (``graft_identity``; 1.0 = fully restored).

Residues counted are ATOM-record amino acids, in file order, merged across repeated
chain blocks -- the same set ProteinMPNN's parser indexes. HETATM (ligands, waters,
and the virtual guide atom HBDesigner adds with --guide_res) is ignored throughout.

Usage (called by hbdesigner.sh; standalone for debugging):
    python hbdesigner_worker.py --name design_0 --design-dir out/design_0 \\
        --ref-pdb out/design_0/design_0.pdb --graft-chains B \\
        --graft-dir out/design_0/grafted --run-rc 0 --result-tsv out/design_0.tsv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import gemmi

# Columns of the per-design TSV. `name` and `rank` key the row; everything after
# `path` is collected as a leaf-prefixed column (collect_hbdesigner.py).
RESULT_COLUMNS = [
    "name",
    "rank",
    "status",
    "path",
    "hb_score_full",
    "hb_score_hb",
    "avg_burial",
    "saturation",
    "buried_heavy_unsats",
    "buried_unsat_hpol",
    "network",
    "network_seq",
    "n_network_res",
    "network_chains",
    "fixed_positions",
    "fixed_idx",
    "resnum_offset",
    "resnum_shift_max",
    "grafted",
    "graft_identity",
    "n_res_total",
]

# Upstream's stats CSV header -> our bare column name. Verified against
# `rank_and_save`, which writes exactly these columns (plus a pandas index).
CSV_FIELDS = {
    "HB_Score_full": "hb_score_full",
    "HB_Score_hb": "hb_score_hb",
    "Avg_Burial": "avg_burial",
    "saturation": "saturation",
    "buried_heavy_unsats": "buried_heavy_unsats",
    "buried_unsat_Hpol": "buried_unsat_hpol",
    "network": "network",
}

# One entry of the colon-separated `network` string: chain, residue number, and
# the one-letter code of the residue placed there (get_network_res).
NETWORK_RE = re.compile(r"^([A-Za-z])(-?\d+)([A-Z])$")


class ResidueMap:
    """A structure's amino-acid residues, per chain, in file order.

    ``index[(chain, resnum)]`` is the 1-based position of that residue within its
    chain -- what ProteinMPNN calls position N -- and ``seq[chain]`` its one-letter
    sequence. Repeated chain blocks are merged under one chain id, HETATM is
    skipped, and insertion codes are NOT part of the key (HBDesigner folds them
    into the residue number when it writes a PDB, so they cannot be matched back).
    """

    def __init__(self, path: Path):
        st = gemmi.read_structure(str(path))
        if len(st) == 0:
            raise ValueError(f"{path.name}: no model")
        self.chain_order: list[str] = []
        self.seq: dict[str, str] = {}
        self.resnums: dict[str, list[int]] = {}
        self.index: dict[tuple[str, int], int] = {}
        for chain in st[0]:
            for res in chain:
                if res.het_flag != "A":
                    continue
                info = gemmi.find_tabulated_residue(res.name)
                if info is None or not info.is_amino_acid():
                    continue
                name = chain.name
                if name not in self.seq:
                    self.chain_order.append(name)
                    self.seq[name] = ""
                    self.resnums[name] = []
                self.seq[name] += (info.one_letter_code or "x").upper()
                self.resnums[name].append(res.seqid.num)
                self.index[(name, res.seqid.num)] = len(self.resnums[name])

    @property
    def n_res(self) -> int:
        return sum(len(v) for v in self.resnums.values())

    def offset(self, chain: str) -> str:
        """``index - resnum`` for this chain when it is constant, else 'var'."""
        offsets = {i + 1 - num for i, num in enumerate(self.resnums[chain])}
        return str(offsets.pop()) if len(offsets) == 1 else "var"


def parse_network(network: str) -> list[tuple[str, int, str]]:
    """'A16T:A12S' -> [('A', 16, 'T'), ('A', 12, 'S')] (order as written)."""
    entries = []
    for token in network.split(":"):
        token = token.strip()
        if not token:
            continue
        m = NETWORK_RE.match(token)
        if m is None:
            raise ValueError(
                f"unparseable network entry {token!r} (expected '<chain><resnum><aa>')"
            )
        entries.append((m.group(1), int(m.group(2)), m.group(3)))
    if not entries:
        raise ValueError("empty network string")
    return entries


def network_columns(network: str, rmap: ResidueMap) -> dict[str, object]:
    """Map the network onto ProteinMPNN positions and report the mapping's trust.

    Sorted by chain (in PDB order) then by position, so ``fixed_positions`` reads
    the way a human writes one. Sorting is safe here: ProteinMPNN preserves the
    order it is given but only ``--tied-positions`` is order-sensitive.
    """
    entries = parse_network(network)
    per_chain: dict[str, list[tuple[int, str, int]]] = {}
    shifts: list[int] = []
    for chain, resnum, aa in entries:
        idx = rmap.index.get((chain, resnum))
        if idx is None:
            raise ValueError(
                f"network residue {chain}{resnum} is not in the output PDB "
                f"(chains present: {','.join(rmap.chain_order) or 'none'})"
            )
        actual = rmap.seq[chain][idx - 1]
        if actual != aa:
            raise ValueError(
                f"network says {chain}{resnum} is {aa} but the output PDB has "
                f"{actual} at that position -- the residue mapping is wrong, so "
                f"fixed_positions would fix the wrong residues"
            )
        per_chain.setdefault(chain, []).append((idx, aa, resnum))
        shifts.append(abs(idx - resnum))

    chains = [c for c in rmap.chain_order if c in per_chain]
    for chain in chains:
        per_chain[chain].sort()

    flat = [(chain, idx, aa) for chain in chains for idx, aa, _ in per_chain[chain]]
    return {
        "network_seq": "".join(aa for _, _, aa in flat),
        "n_network_res": len(flat),
        "network_chains": ",".join(chains),
        # ProteinMPNN --fixed-positions syntax: '/' between chains (one group per
        # chain in --chains-to-design order), ',' between positions.
        "fixed_positions": "/".join(
            ",".join(str(idx) for idx, _, _ in per_chain[chain]) for chain in chains
        ),
        # The same positions flat, for the per-row {expr} hand-off (fix1..fixK).
        "fixed_idx": ",".join(str(idx) for _, idx, _ in flat),
        "resnum_offset": ",".join(f"{c}:{rmap.offset(c)}" for c in chains),
        "resnum_shift_max": max(shifts),
    }


def graft_columns(
    out_map: ResidueMap, ref_map: ResidueMap, graft_chains: list[str]
) -> dict[str, object]:
    """Did the graft put the omitted chains back, and how completely?"""
    if not graft_chains:
        return {"grafted": "no", "graft_identity": None}
    matched = total = 0
    for chain in graft_chains:
        if chain not in out_map.seq:
            raise ValueError(f"grafted chain {chain} is missing from the output PDB")
        if chain not in ref_map.seq:
            raise ValueError(f"grafted chain {chain} is missing from the input PDB")
        out_seq, ref_seq = out_map.seq[chain], ref_map.seq[chain]
        if len(out_seq) != len(ref_seq):
            raise ValueError(
                f"grafted chain {chain} has {len(out_seq)} residues but the input "
                f"has {len(ref_seq)}"
            )
        matched += sum(a == b for a, b in zip(out_seq, ref_seq))
        total += len(ref_seq)
    return {
        "grafted": "yes",
        "graft_identity": round(matched / total, 4) if total else None,
    }


def write_result(result_tsv: Path, rows: list[dict]) -> None:
    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(result_tsv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=RESULT_COLUMNS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in RESULT_COLUMNS})


def read_stats(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def collect_design(args: argparse.Namespace) -> list[dict]:
    design_dir = Path(args.design_dir)
    stats_csv = design_dir / f"{args.name}_HBDes_stats.csv"
    graft_chains = [c.strip() for c in args.graft_chains.split(",") if c.strip()]

    if args.run_rc != 0:
        return [
            {
                "name": args.name,
                "rank": 0,
                "status": f"error: run_hbdesigner exited {args.run_rc}",
            }
        ]

    # The zero-output success: upstream returned early without writing anything.
    if not stats_csv.is_file():
        return [
            {
                "name": args.name,
                "rank": 0,
                "status": (
                    "error: no networks (hbdesigner exited 0 without writing "
                    "a stats CSV -- nothing passed scoring or symmetrization)"
                ),
            }
        ]

    ref_map = ResidueMap(Path(args.ref_pdb))
    rows: list[dict] = []
    for raw in read_stats(stats_csv):
        rank = int(raw["Rank"])
        row: dict = {"name": args.name, "rank": rank, "status": "OK"}
        pdb_name = raw.get("Output_PDB") or f"{args.name}_HBDes_rank_{rank}.pdb"
        # When grafting ran, the grafted file is the one the next tool must use.
        pdb_path = (Path(args.graft_dir) if graft_chains else design_dir) / pdb_name
        try:
            for src, dst in CSV_FIELDS.items():
                if src not in raw:
                    raise ValueError(f"stats CSV has no column {src!r}")
                row[dst] = raw[src]
            if not pdb_path.is_file():
                raise ValueError(f"missing output structure {pdb_path.name}")
            out_map = ResidueMap(pdb_path)
            if out_map.n_res != ref_map.n_res:
                raise ValueError(
                    f"output has {out_map.n_res} residues, input has "
                    f"{ref_map.n_res} -- the scaffold was not preserved"
                )
            row["n_res_total"] = out_map.n_res
            row.update(network_columns(raw["network"], out_map))
            row.update(graft_columns(out_map, ref_map, graft_chains))
            row["path"] = str(pdb_path)
        except Exception as e:  # noqa: BLE001 - errors are recorded as data
            print(f"{args.name} rank {rank}: ERROR {e}", file=sys.stderr)
            row["status"] = f"error: {e}"
            row["path"] = ""
        rows.append(row)

    if not rows:
        return [
            {
                "name": args.name,
                "rank": 0,
                "status": "error: no networks (stats CSV is empty)",
            }
        ]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise one hbdesigner design dir.")
    ap.add_argument("--name", required=True, help="Design (parent row) name.")
    ap.add_argument("--design-dir", required=True, help="This design's output dir.")
    ap.add_argument(
        "--ref-pdb",
        required=True,
        help="The PDB handed to HBDesigner: the graft reference and the residue-"
        "count reference.",
    )
    ap.add_argument(
        "--graft-chains",
        default="",
        help="Comma-joined chains that were grafted back ('' = none grafted).",
    )
    ap.add_argument("--graft-dir", default="", help="Dir holding the grafted PDBs.")
    ap.add_argument(
        "--run-rc", type=int, default=0, help="Exit code of run_hbdesigner."
    )
    ap.add_argument("--result-tsv", required=True, help="Per-design TSV to write.")
    args = ap.parse_args()

    rows = collect_design(args)
    write_result(Path(args.result_tsv), rows)
    ok = sum(1 for r in rows if r["status"] == "OK")
    print(f"{args.name}: wrote {len(rows)} row(s), {ok} OK -> {args.result_tsv}")


if __name__ == "__main__":
    main()
