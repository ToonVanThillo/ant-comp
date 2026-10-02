#!/usr/bin/env python3
"""
Collect RPXdock results into the child dock table.

``rpxdock_worker.py`` writes one ``<out_dir>/<scaffold>.tsv`` per scaffold, with
one ROW PER KEPT DOCK, and the structures themselves under
``<out_dir>/<scaffold>/<scaffold>_d<rank>.pdb``. This is a ``create`` tool: each of
those rows becomes a child row whose ``parent_name`` is the scaffold it was docked
from, so one parent fans out to up to ``--nout-top`` children.

Row names are ``<scaffold>_d<rank>``, rank 1 = best scoring, which is deterministic
for a given run and sorts the way a reader expects. The underlying ``model`` index
into RPXdock's own Result is collected alongside, because that -- not the rank --
is what ``_Result.txz`` is indexed by.

Columns (the driver leaf-prefixes each to ``rpxdock_<name>``):

  THE DOCK
    score       the weighted RPXdock score this row is ranked by (higher better).
                A motif-table lookup: comparable within one batch at one
                --hscore-files setting, NOT across settings or against any
                physical energy.
    rpx         the unweighted motif component of that score
    ncontact    the unweighted contact-count component (residue-centroid pairs
                within --max-pair-dist)
    model       index into the run's _Result.txz -- the stable identity of the pose
    rank        1 = best scoring of this scaffold's kept docks
    reslb/resub 0-indexed first/last residue of the scaffold actually present in
                the dock. With --max-trim 0 these are 0 and n_res-1; with trimming
                on, the dumped structure is SHORTER than the scaffold.
    disp        displacement along the symmetry axis (one-component cages and
                dihedrals only; NA for cyclic)

  WHAT PRODUCED IT
    architecture   the symmetry docked into, e.g. 'C2', 'D3_2'
    nfold          order of the axis docked about
    hscore         the motif-table set used. Carried on every row precisely so two
                   batches scored against different tables can never be compared by
                   accident.
    scaffold_path  the structure actually docked (after any CIF->PDB conversion)
    result_path    the _Result.txz, for rescoring or re-dumping upstream-side
    n_docks        poses RPXdock returned for this scaffold after redundancy
                   filtering, before the --nout-top cut. Same on every row of a
                   scaffold. n_docks == nout_top means the cut bound, and there may
                   be more worth keeping.

  TRUST METRICS -- read these before trusting a score
    n_res            residues in the Body RPXdock scored
    n_chains_in      chains in the input FILE. RPXdock collapses every chain into
                     one and renumbers from 1, so anything >1 means it docked a
                     concatenation. For a cyclic dock of a monomer that is wrong;
                     for a cage/dihedral the input should be a single-chain
                     asymmetric unit, so >1 is wrong there too.
    frac_helix       secondary-structure composition of that same Body, as assigned
    frac_sheet       by willutil's pure-Python DSSP (no PyRosetta in the image).
    frac_loop        The default ilv_h tables score HELIX PAIRS ONLY -- a scaffold
                     read as mostly loop scores ~0 with no error, and frac_loop is
                     the only thing that says so.
    input_com_dist   distance (A) of the input file's CA centre of mass from the
                     origin, measured BEFORE any recentring. RPXdock's samplers
                     assume an origin-centred body; a large value with
                     recentered == False means the dock happened in a frame nobody
                     intended.
    recentered       whether --recenter-input was applied. When True the transforms
                     RPXdock reports are relative to the recentred pose.
    hscore_seconds   time spent loading the motif tables
    seconds          wall time of the whole scaffold's dock

A scaffold whose dock FAILED mints no rows at all -- there is no entity to key them
to. The failure is printed here and preserved in that scaffold's ``.tsv``
(``status: error: ...``), and the task itself exits non-zero so the scheduler shows
it. To audit a run, compare the child table's distinct ``parent_name`` values
against the parent table's rows: a parent with no child either failed or produced
no non-clashing dock.

NA (empty) rather than 0 marks a field that does not apply -- ``disp`` on a cyclic
dock, every field of a failed scaffold. A score of 0 is a real (bad) value.

Usage:
    sapia collect rpxdock outputs/RUN --table <the table the run reserved>
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd
from prosapia.core import CollectCtx, CollectEach, Collected, DesignCtx

# Bare column names; the driver leaf-prefixes them (rpxdock_<name>).
FLOAT_COLUMNS = [
    "score",
    "rpx",
    "ncontact",
    "disp",
    "frac_helix",
    "frac_sheet",
    "frac_loop",
    "input_com_dist",
    "hscore_seconds",
    "seconds",
]
INT_COLUMNS = [
    "model",
    "rank",
    "nfold",
    "reslb",
    "resub",
    "n_docks",
    "n_res",
    "n_chains_in",
]
STR_COLUMNS = ["architecture", "hscore", "scaffold_path", "result_path"]
BOOL_COLUMNS = ["recentered"]
RESULT_COLUMNS = FLOAT_COLUMNS + INT_COLUMNS + STR_COLUMNS + BOOL_COLUMNS


def _present(row: pd.Series, col: str) -> Any | None:
    """The cell's value, or None when the column is absent, NaN or blank."""
    value = row.get(col)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return value


def _fields(row: pd.Series) -> dict[str, Any]:
    """One worker row -> collected columns, NA-safe (a missing or blank cell is a
    field that did not apply, not a zero)."""
    data: dict[str, Any] = {c: None for c in RESULT_COLUMNS}
    for col in FLOAT_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = float(value)
    for col in INT_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = int(float(value))
    for col in STR_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = str(value)
    for col in BOOL_COLUMNS:
        if (value := _present(row, col)) is not None:
            data[col] = str(value).strip().lower() in ("true", "1", "yes")
    return data


def collect_rpxdock(ctx: CollectCtx) -> CollectEach:
    """Per-scaffold RPXdock collector.

    A create tool: the framework iterates the ready PARENT rows and this mints one
    child row per kept dock, carrying ``parent`` for lineage. A scaffold with no
    successful dock yields no rows (and says so on stdout). The framework stamps
    status/path/parent_name from each Collected.
    """

    def one(d: DesignCtx) -> Iterable[Collected]:
        tsv_path = ctx.out_dir / f"{d.name}.tsv"
        if not tsv_path.is_file():
            print(f"{d.name}: no result TSV at {tsv_path}, skipping")
            return

        df = pd.read_csv(tsv_path, sep="\t", dtype={"name": str, "status": str})
        if df.empty:
            print(f"{d.name}: empty result TSV, skipping")
            return

        # The worker writes a single name-less row when the whole scaffold failed.
        docked = df[df["name"].notna() & (df["name"].astype(str).str.strip() != "")]
        if docked.empty:
            status = str(df.iloc[0].get("status", "error: no dock rows"))
            print(f"{d.name}: {status} -- no dock rows (no child row minted)")
            return

        n = 0
        for _, row in docked.iterrows():
            child = str(row["name"])
            status = str(row["status"])
            path = _present(row, "path")
            if status != "OK" or path is None:
                # A dock row that is not OK has no structure; emitting it with a
                # zeroed score would read as a real, bad dock.
                print(f"{d.name}: {child} {status}")
                yield Collected(
                    name=child,
                    parent=d.name,
                    status=status,
                    path="",
                    data={c: None for c in RESULT_COLUMNS},
                )
                n += 1
                continue
            yield Collected(
                name=child,
                parent=d.name,
                status=status,
                path=str(path),
                data=_fields(row),
            )
            n += 1
        print(f"{d.name}: OK ({n} dock(s))")

    return one
