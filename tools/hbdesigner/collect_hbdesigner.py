#!/usr/bin/env python3
"""
Collect hbdesigner outputs into the child table it reserved.

Reads the one-row-per-rank TSV ``<run_dir>/<table>/hbdesigner/<design>.tsv`` written
by hbdesigner_worker.py and mints one child row per kept network, keyed
``<parent>_hb<rank>`` (rank 1 = HBDesigner's best) and carrying ``parent_name`` for
lineage. Nothing from the ancestors is copied -- resolve it with ``lookup`` when you
need it.

Columns (leaf-prefixed ``hbdesigner_``):

    path                  the designed structure. The GRAFTED file when chains were
                          grafted back, otherwise the raw rank PDB. Poly-glycine
                          everywhere except the network -- by design, not corruption.
    rank                  1 = best by HBDesigner's own ordering (fewest buried
                          unsats, then highest saturation, then lowest HB_Score_full)
    hb_score_full         ref2015 energy of the network, per network residue,
                          relative to the same backbone as poly-glycine. MORE
                          NEGATIVE IS BETTER.
    hb_score_hb           the same difference under an hbond-only score function
                          (fa_rep 0.55 + hbond_sc + hbond_bb_sc). More negative is
                          better; this is the one that is about hydrogen bonds.
    avg_burial            mean sidechain-neighbour count over the network residues.
                          Higher = more buried. A buried network is the point.
    saturation            fraction of the network's polar groups that are satisfied
                          (range 0-2). Higher is better; --min-sat filtered on it.
    buried_heavy_unsats   buried unsatisfied heavy atoms left in the network
                          (--max-buns, default 0 = none tolerated). Lower is better.
    buried_unsat_hpol     buried unsatisfied polar hydrogens (--max-buphs, default
                          5). Lower is better.
    network               HBDesigner's own description of the network, as
                          ``<chain><resnum><aa>`` joined by ':', e.g. 'A12S:A16T'.
                          PDB NUMBERING. Includes --anchor-res residues.
    network_seq           the network's amino acids in fixed_positions order ('STN')
    n_network_res         how many residues the network has. Usually --n-res, but
                          anchors and symmetrization can change it -- read it, do
                          not assume it.
    network_chains        chains the network touches, comma-joined, in PDB order.
                          This is what --chains-to-design must list for
                          fixed_positions to line up.
    fixed_positions       the network in ProteinMPNN's --fixed-positions syntax
                          ('12,16/5': ',' between positions, '/' between chains, one
                          group per chain in network_chains order). 1-BASED WITHIN
                          EACH CHAIN -- mapped through the output PDB's residue
                          order, never copied from the resnums in `network`.
    fix1 .. fixK          the same positions one per column, so a per-row hand-off
                          can be written as --fixed-positions '{hbdesigner_fix1},
                          {hbdesigner_fix2}/' (prosapia's {expr} islands resolve
                          integers only, so a whole string column cannot be used).
    resnum_offset         per network chain, ``mpnn position - PDB resnum`` when it
                          is constant over that chain, else '<chain>:var'. All
                          zeroes means the PDB was numbered 1..L per chain and
                          `network`'s resnums ARE the mpnn positions.
    resnum_shift_max      max |mpnn position - PDB resnum| over the network
                          residues. 0 = no renumbering happened. The filterable
                          form of the trust check above.
    grafted               'yes' if the omitted chains were restored from the input
                          PDB (upstream graft_seq.py), 'no' otherwise.
    graft_identity        fraction of the grafted chains' residues that now match
                          the input. 1.0 = fully restored; NA when nothing was
                          grafted. Below 1.0 means the graft did not do its job --
                          treat the structure as wrong, not merely imperfect.
    n_res_total           residues in the output structure (checked equal to the
                          input's; a mismatch is an error status, not a column).

Two things to read carefully:

  * A ``<parent>_hb0`` row is a FAILURE RECORD, not a design: HBDesigner can find
    no network and still exit 0, writing neither a PDB nor a CSV. That row carries
    an ``error:`` status, an empty path and NA metrics, so the parent's failure is
    visible in the table instead of looking like a design that was never attempted.
    Never read it as a design; always gate on ``hbdesigner_status == 'OK'``.
  * The fan-out is NOT ``--top-k`` per parent. Upstream keeps
    ``min(top_k, n_surviving)``, so a parent may contribute 5 rows, 1 row, or only
    the ``_hb0`` failure record.

A design whose TSV is absent altogether (task still queued, killed, or never
submitted) is SKIPPED with a message -- no evidence is not the same as failure.

Usage:
    sapia collect hbdesigner outputs/<run> --table <the table the run reserved>
"""

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from prosapia.core import CollectCtx, CollectEach, Collected, DesignCtx

FLOAT_COLUMNS = [
    "hb_score_full",
    "hb_score_hb",
    "avg_burial",
    "saturation",
    "graft_identity",
]
INT_COLUMNS = [
    "rank",
    "buried_heavy_unsats",
    "buried_unsat_hpol",
    "n_network_res",
    "resnum_shift_max",
    "n_res_total",
]
STR_COLUMNS = [
    "network",
    "network_seq",
    "network_chains",
    "fixed_positions",
    "resnum_offset",
    "grafted",
]
# `fixed_idx` is exploded into fix1..fixK instead of being collected as a column.
DATA_COLUMNS = FLOAT_COLUMNS + INT_COLUMNS + STR_COLUMNS


def _value(row: dict[str, str], key: str) -> str | None:
    """The cell, or None when absent or blank (NA, never a silent zero)."""
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def _fields(row: dict[str, str]) -> dict[str, Any]:
    """One TSV row -> collected columns. A blank stays NA rather than becoming 0."""
    data: dict[str, Any] = {c: None for c in DATA_COLUMNS}
    for col in FLOAT_COLUMNS:
        if (value := _value(row, col)) is not None:
            data[col] = float(value)
    for col in INT_COLUMNS:
        if (value := _value(row, col)) is not None:
            # Written through pandas upstream, so an int can arrive as '0.0'.
            data[col] = int(float(value))
    for col in STR_COLUMNS:
        if (value := _value(row, col)) is not None:
            data[col] = value
    if (idx := _value(row, "fixed_idx")) is not None:
        for i, position in enumerate(idx.split(","), start=1):
            data[f"fix{i}"] = int(position)
    return data


def collect_hbdesigner(ctx: CollectCtx) -> CollectEach:
    """Per-parent hbdesigner collector. hbdesigner is a create tool: the framework
    iterates the ready parents and this mints one child row per kept network
    (``<parent>_hb<rank>``), carrying ``parent`` for lineage. The framework stamps
    status/path/parent_name from each Collected. Rows are rebuilt from disk on every
    collect, so re-collecting is idempotent."""

    def one(d: DesignCtx) -> Iterable[Collected]:
        tsv_path = ctx.out_dir / f"{d.name}.tsv"
        if not tsv_path.is_file():
            # No result file at all: the task did not run or did not finish. That is
            # not a design failure, so nothing is minted -- re-collect after the
            # task completes.
            print(f"{d.name}: no result TSV at {tsv_path.name} (skipping)")
            return

        with open(tsv_path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        for row in rows:
            rank = int(float(_value(row, "rank") or 0))
            status = _value(row, "status") or "error: no status recorded"
            path = _value(row, "path") or ""
            yield Collected(
                # rank 0 never collides with a real design: upstream ranks from 1.
                name=f"{d.name}_hb{rank}",
                parent=d.name,
                path=path,
                status=status,
                # Whatever the worker managed to record is kept even on an error
                # row (blanks collect as NA); the status is the gate, not the data.
                data=_fields(row),
            )

    return one
