"""Collect Proton-PottsMPNN designs into a child table.

Each task wrote ``<out_dir>/<parent_name>/designs.tsv`` -- one row per design, already
carrying every metric (the worker computes them while the featurised context is still in
memory). This collector is therefore a reader, not a calculator: it mints one child row
per design and leaf-prefixes the columns.

Row keys are ``<parent>_p<i>`` over the file's row order, which is the engine's own
``sorted_by_energy`` order (lowest Potts energy first) -- so ``_p0`` is the stablest
design of that backbone, NOT the most pH-selective one. Rank by the column you care
about rather than by the suffix.

Columns, and how to read them
-----------------------------
Science:
  sequence          1-letter binder sequence -- what you fold (via mkcomplex -> boltz)
  extended_tokens   the parallel 3-letter + protonation-state string (… ASP-P … HIS-P …)
  potts_energy      whole-system Potts Hamiltonian; LOWER = stabler
  selective_energy  sum over centres of (e_protonated - e_deprotonated); LOWER = the
                    protonated microstate is more preferred, i.e. more pH-selective
  global_dh         binder-wide H(all titratable protonated) - H(all deprotonated);
                    LOWER = the whole binder leans low-pH, not just the centres
  combined_lambda   the trade-off this design was optimised at (0 stability, 1 selectivity)
  pareto            True if no other design of the SAME backbone beats it on both energies
  centers           pinned centres as 'resnum:STATE;…' in INPUT-structure numbering
  center_seqpos     the same centres as 1-based positions in `sequence`
  n_centers         how many were pinned

Trust (read these before the science):
  centers_verified  fraction of pinned centres whose token in `extended_tokens` really is
                    the pinned microstate. < 1.0 => the design does not carry the centre
                    it was optimised for and its selective_energy is meaningless
  resnum_offset     res_id of the binder's first residue minus 1. 0 means the chain was
                    renumbered from 1, so `centers` are positions, not target numbering
  n_mut             mutations against the sequence the redesign started from. 0 = the
                    seed echoed back, not a design
  seq_rec           identity to that same starting sequence
  seed_source       'native' (the input structure's sequence) or 'inverse' (--seed-column)
  binder_chain      which chain was redesigned
  binder_len        length of `sequence`
  n_designable      positions the optimiser was allowed to touch

THE ENERGIES ARE PER-BACKBONE. They are model energies z-scored within one featurised
backbone, so comparing potts_energy across different parents ranks nothing. Filter and
rank within a parent.

Usage:
    sapia collect protonpottsmpnn outputs/<run> --table <the table the run reserved>
"""

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from prosapia.core import Collected, CollectCtx, CollectEach, DesignCtx

# Written by the worker; the only file this collector reads.
DESIGNS_TSV = "designs.tsv"

# Columns whose values are numbers, so the table can be filtered arithmetically rather
# than lexically. Anything absent or unparseable becomes NA (never 0 -- a 0 energy would
# read as a real, and very good, score).
NUMERIC: dict[str, type] = {
    "potts_energy": float,
    "selective_energy": float,
    "global_dh": float,
    "combined_lambda": float,
    "centers_verified": float,
    "seq_rec": float,
    "n_centers": int,
    "n_designable": int,
    "binder_len": int,
    "resnum_offset": int,
    "n_mut": int,
    "sample": int,
}

# Everything else kept as-is, in the order the table should show it.
TEXT = (
    "design_id",
    "sequence",
    "extended_tokens",
    "centers",
    "center_seqpos",
    "binder_chain",
    "seed_source",
)


def _coerce(row: "pd.Series") -> dict[str, Any]:
    """One TSV row -> the bare-named columns the driver will leaf-prefix."""
    data: dict[str, Any] = {}
    for column in TEXT:
        value = row.get(column)
        data[column] = "" if value is None or pd.isna(value) else str(value)
    for column, caster in NUMERIC.items():
        value = row.get(column)
        if value is None or pd.isna(value) or str(value) == "":
            data[column] = pd.NA
            continue
        try:
            data[column] = caster(float(value))
        except (TypeError, ValueError):
            data[column] = pd.NA
    pareto = row.get("pareto")
    data["pareto"] = pd.NA if pareto is None or pd.isna(pareto) else str(pareto) == "True"
    return data


def collect_protonpottsmpnn(ctx: CollectCtx) -> CollectEach:
    """Per-parent collector. protonpottsmpnn is a create tool: the framework iterates the
    ready parents and this mints one child row (``<parent>_p<i>``) per design, carrying
    ``parent`` for lineage. Child rows are rebuilt from disk, so re-running is idempotent.
    """

    def one(d: DesignCtx) -> Iterable[Collected]:
        tsv_path: Path = ctx.out_dir / d.name / DESIGNS_TSV
        if not tsv_path.is_file():
            # No output at all: the task never reached the worker. The framework marks
            # the PARENT row missing; nothing is minted.
            print(f"{d.name}: no {DESIGNS_TSV} found (skipping)")
            return

        frame = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
        if frame.empty:
            print(f"{d.name}: {DESIGNS_TSV} is empty (skipping)")
            return

        # The worker's failure file: a single row carrying only `status`. Mint ONE child
        # row that records the error, so the failure is visible in the table rather than
        # being an absence someone has to notice.
        if "sequence" not in frame.columns:
            status = str(frame.iloc[0].get("status", "error: unknown"))
            yield Collected(
                name=f"{d.name}_p0",
                parent=d.name,
                path=tsv_path,
                status=status,
                data={},
            )
            return

        for i, (_, row) in enumerate(frame.iterrows()):
            status = str(row.get("status") or "OK")
            yield Collected(
                name=f"{d.name}_p{i}",
                parent=d.name,
                path=tsv_path,
                status=status,
                data=_coerce(row),
            )

    return one
