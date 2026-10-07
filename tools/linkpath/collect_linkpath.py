#!/usr/bin/env python3
"""
Collect linkpath results into the table it ran against.

Reads the per-design <name>.tsv files that linkpath_worker.py wrote under
<run_dir>/<table>/linkpath[_<label>]/ and merges them back as <prefix>_status /
<prefix>_path (the route as a chain of PDB pseudo-atoms, chain X forward and chain Y
reverse -- for looking at in PyMOL, not for designing against) plus one
<prefix>_<field> column per metric:

  provenance / trust -- a distance is meaningless without the parameters and the
  endpoints it was measured between, so they are columns, not run notes:
    from_res, to_res    the anchor residues as 'A:111' / 'B:1'
    from_atom, to_atom  the backbone atom actually used ('C' or 'CA'; 'N' or 'CA').
                        A 'CA' here means the residue had no C/N -- the distance is
                        then ~1.3 A off the chemically right endpoint
    from_chain, to_chain  resolved: 'auto' never reaches the table
    obstacle_chains     THE auditable column: the RESOLVED chain list the route had
                        to go around. Never 'all' -- always what 'all' resolved to,
                        so a later reader can tell whether the target was in there
    n_obstacle_atoms    how many atoms that was
    probe, spacing      echoed, because they set the answer (see the skill)
    grid_shape          'nx x ny x nz' voxels
    direct_clear        was the straight anchor-to-anchor vector unobstructed?
                        Measured analytically against every obstacle atom EXCEPT the
                        two anchor residues' own, which the segment necessarily
                        starts and ends inside

  geometry:
    straight_dist       anchor atom to anchor atom (backbone C -> backbone N), the
                        chemically correct endpoints for a linker. A LOWER BOUND
    straight_ca_dist    CA to CA -- the BRIDGE to dimerfit's retired link_dist,
                        which was CA-CA. Compare old rows to this column, not to
                        straight_dist (they differ by ~1 A) and never to path_dist
    path_dist           the A* route length, anchor to anchor, including the short
                        hops from each anchor atom to its first free voxel
    detour_ratio        path_dist / straight_dist (1.0 = the straight line was free)
    path_found          False means NO FREE ROUTE at this probe -- a RESULT, not a
                        failure (see below)
    min_clearance       closest approach of the route to an atom SURFACE (A),
                        sampled along the route, outside the anchor bubbles. A route
                        that honestly stays in solvent-accessible space never comes
                        closer than `probe`; well below it means the 26-connected
                        grid cut a corner, which makes path_dist too SHORT. Below
                        probe - 0.25 the status becomes 'warn: ...'
    rev_path_dist       the reverse route (C-term of to_chain -> N-term of
    path_asymmetry      from_chain) and |path_dist - rev_path_dist|. Under exact C2
                        the two routes are the same route, so this is a free trust
                        metric: it measures the grid's own discretisation error.
                        Measured ~0.16 A on an exact C2 dimer at the defaults
    n_res_min           ceil(path_dist / taut_rise): a taut, strained linker
    taut_rise           the A/residue constant that produced it (default 3.5)
    n_res_relaxed       ceil(path_dist / relaxed_rise): a relaxed coil. Design
                        against this one
    relaxed_rise        the A/residue constant that produced it (default 2.1). The
                        rises are columns for the same reason probe and spacing are:
                        a residue count cannot be re-derived without them, and 2.1
                        A/res is a CONVENTION about how taut a coil may be, not a
                        physical constant. All four are NA under
                        --no-residue-estimate, and when no route was found
    note                free text: why no path was found, mostly
    seconds             wall time of the design's measurement

NA (empty) rather than 0 marks a field that did not apply: the design failed, no TSV
was written, or the geometry was undefined. A blank must never be read as a passing
zero -- in particular a missing path_dist is not "zero distance", it is "no route".

The 'no path found' contract
----------------------------
If A* finds no free route, the design is UNLINKABLE at that probe radius. That is a
finding, not a tool failure, so:

    status      stays 'OK'
    path_found  False
    path_dist, detour_ratio, n_res_min, n_res_relaxed   NA
    note        'no free path at probe 2.0 / spacing 1.0'

Do not read `status == "OK"` as "a route exists" -- filter on `path_found`. Using an
'error:' status here would conflate "this design cannot be linked" with "the tool
broke", and a standard trust filter would silently delete a real result.

A status of 'warn: ...' means the numbers were computed but the route's measured
clearance says the grid cut a corner; the metrics are kept so the suspicion can be
judged from the table, and the prefix keeps the row out of any `== "OK"` selection.
Genuine failures (a missing chain, a chain with no amino acids, an unreadable file,
'auto' on a structure that is not two protein chains, a violated
path_dist >= straight_dist invariant) get an 'error:' status with every metric NA.

Usage:
    sapia collect linkpath outputs/20261002_143419_dimer_phase2 --table table1
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd
from prosapia.core import CollectCtx, CollectEach, Collected, DesignCtx

# Bare column names; the driver leaf-prefixes them (linkpath_<name>).
FLOAT_COLUMNS = [
    "probe",
    "spacing",
    "straight_dist",
    "straight_ca_dist",
    "path_dist",
    "detour_ratio",
    "min_clearance",
    "rev_path_dist",
    "path_asymmetry",
    # The rise constants ride beside the counts they produced (see the worker): a
    # residue count cannot be re-derived from the table without them, and the rise
    # is a convention, not a physical constant. NA whenever the count is NA.
    "taut_rise",
    "relaxed_rise",
    "seconds",
]
INT_COLUMNS = [
    "n_obstacle_atoms",
    "n_res_min",
    "n_res_relaxed",
]
BOOL_COLUMNS = [
    "direct_clear",
    "path_found",
]
STR_COLUMNS = [
    "from_res",
    "to_res",
    "from_atom",
    "to_atom",
    "from_chain",
    "to_chain",
    "obstacle_chains",
    "grid_shape",
    "note",
]
RESULT_COLUMNS = FLOAT_COLUMNS + INT_COLUMNS + BOOL_COLUMNS + STR_COLUMNS

# Every string column is read as a string: a chain id like '1', a residue label like
# 'A:111' and a grid shape like '86x68x69' must never be guessed at by pandas' type
# inference. The booleans are read as strings too and converted explicitly, so a
# literal 'False' can never arrive as the truthy string it would otherwise be.
STR_DTYPES = {c: str for c in STR_COLUMNS + BOOL_COLUMNS}


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
    for col in BOOL_COLUMNS:
        if (value := _present(row, col)) is not None:
            text = str(value).strip().lower()
            if text not in ("true", "false"):
                raise ValueError(
                    f"{col!r} should be True or False, got {value!r} -- the worker "
                    f"and this collector disagree about the TSV format."
                )
            data[col] = text == "true"
    for col in STR_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = str(value)
    return data


def collect_linkpath(ctx: CollectCtx) -> CollectEach:
    """Per-design linkpath collector. The framework iterates ready designs and stamps
    status/path (keyed by the tool leaf); this reads one design's one-row <name>.tsv
    and adds the provenance, geometry and trust columns. Variants -- another
    structure column, another obstacle set, another probe -- are distinguished via
    --dir-label, matching the output dir."""

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

        # An 'error:' row keeps every metric NA -- the message (which names the
        # reason: the missing chain, the chain count, the violated invariant) is the
        # only thing worth carrying.
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
