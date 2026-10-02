#!/usr/bin/env python3
"""
Collect ifacegeom results into the table.

Reads the per-design <name>.tsv files that ifacegeom_worker.py wrote under
<run_dir>/<table>/ifacegeom[_<label>]/ and merges them back as <prefix>_status /
<prefix>_path (the per-residue TSV: side, label, chain, resnum, resname, selected,
selected_by, min_heavy_dist, min_cb_dist -- the evidence behind the lists) plus one
<prefix>_<field> column per metric:

    method              'vector' or 'heavy' -- which criterion produced the lists
    binder_chains       the two sides actually used (target_chains is resolved here
    target_chains       when --target-chains was 'auto')
    binder_len          amino acids per side
    target_len
    binder_res          BINDER interface residues as 'A:12,A:15,...' -- the epitope
                        footprint, carried as a COLUMN so it survives the generation
                        boundary into the dock and network tables
    n_binder_res
    binder_res_seq      their one-letter codes, in the same order (lets a later
                        graft be checked by residue identity)
    binder_his          histidines among them, and how many
    n_binder_his
    target_res          TARGET interface residues, same format
    n_target_res
    target_his          histidines in the target epitope, and how many -- the
    n_target_his        residues whose contacts change when the pH drops
    binder_com          mass-weighted centres as 'x,y,z' (this file's frame)
    binder_iface_com    centre of the binder-side interface residues = the epitope
                        COM, the reference point dimerfit measures against later
    iface_com           centre of both sides' interface residues = the axis endpoint
    axis_len            |iface_com - binder_com|, the axis the projections use
    nterm_res           which residues the termini are ('A:1' / 'A:115'), so every
    cterm_res           projection below is checkable against the structure by hand
    nterm_proj          SIGNED projection (A) of the terminus CA on the
    cterm_proj          binder_com -> iface_com axis, measured from binder_com:
                        > 0 = the interface side of the plane, < 0 = the far side.
                        cterm_proj is the C-terminal-tag criterion; nterm_proj is
                        the partner-reach one. Both are RECORDED, not filtered on
                        -- the requirement is two-sided.
    nterm_iface_dist    terminus CA to iface_com (A)
    cterm_iface_dist
    nterm_iface_min_dist  terminus CA to the NEAREST epitope CA (A) -- the number
    cterm_iface_min_dist  that actually answers "would the tag sit in the epitope"
    seconds             wall time of the design's measurement

NA (empty) rather than 0 marks a field that did not apply: the design failed, no TSV
was written, or the geometry was undefined (no interface residues, so no axis).

A status of 'warn: no interface residues' means the two chain sets never came within
the cutoffs. That is almost always the wrong --binder-chains/--target-chains rather
than a real non-contact; the lengths and chain names are still recorded so the
mistake is visible, and the row stays out of any `== "OK"` selection.

Usage:
    sapia collect ifacegeom outputs/20261002_dimer_binder --table table0
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd
from prosapia.core import CollectCtx, CollectEach, Collected, DesignCtx

# Bare column names; the driver leaf-prefixes them (ifacegeom_<name>).
FLOAT_COLUMNS = [
    "axis_len",
    "nterm_proj",
    "cterm_proj",
    "nterm_iface_dist",
    "cterm_iface_dist",
    "nterm_iface_min_dist",
    "cterm_iface_min_dist",
    "seconds",
]
INT_COLUMNS = [
    "binder_len",
    "target_len",
    "n_binder_res",
    "n_binder_his",
    "n_target_res",
    "n_target_his",
]
STR_COLUMNS = [
    "method",
    "binder_chains",
    "target_chains",
    "binder_res",
    "binder_res_seq",
    "binder_his",
    "target_res",
    "target_his",
    "binder_com",
    "binder_iface_com",
    "iface_com",
    "nterm_res",
    "cterm_res",
]
RESULT_COLUMNS = FLOAT_COLUMNS + INT_COLUMNS + STR_COLUMNS

# Every string column is read as a string: a residue list like 'A:12,A:15' and a
# coordinate like '1.0,2.0,3.0' must never be guessed at by pandas' type inference.
STR_DTYPES = {c: str for c in STR_COLUMNS}


def _empty() -> dict[str, Any]:
    return {c: None for c in RESULT_COLUMNS}


def _present(row: pd.Series, col: str) -> Any | None:
    """The cell's value, or None when the column is absent, NaN or blank."""
    value = row.get(col)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return value


def _fields(row: pd.Series) -> dict[str, Any]:
    """One result row -> collected columns, NA-safe (a missing or blank cell is a
    field that did not apply, not a zero)."""
    data = _empty()
    for col in FLOAT_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = float(value)
    for col in INT_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = int(float(value))
    for col in STR_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = str(value)
    return data


def collect_ifacegeom(ctx: CollectCtx) -> CollectEach:
    """Per-design ifacegeom collector. The framework iterates ready designs and stamps
    status/path (keyed by the tool leaf); this reads one design's one-row <name>.tsv
    and adds the interface-geometry columns. Variants -- another structure column or
    another chain split off the same table -- are distinguished via --dir-label,
    matching the output dir."""

    def one(d: DesignCtx) -> Iterable[Collected]:
        tsv_path = ctx.out_dir / f"{d.name}.tsv"
        if not tsv_path.is_file():
            yield Collected(status="missing", path="", data=_empty())
            return

        result_df = pd.read_csv(tsv_path, sep="\t", dtype=STR_DTYPES)
        if result_df.empty:
            yield Collected(status="error: empty tsv", path="", data=_empty())
            return

        row = result_df.iloc[0]
        status = str(row["status"])

        # 'warn: ...' means the measurement ran but found no interface -- keep the
        # lengths and chain names, which are what make a chain-selection mistake
        # obvious, while the status keeps the row out of any `== "OK"` selection.
        if status != "OK" and not status.startswith("warn:"):
            yield Collected(status=status, path="", data=_empty())
            return

        path = _present(row, "path")
        yield Collected(
            status=status,
            path="" if path is None else str(path),
            data=_fields(row),
        )

    return one
