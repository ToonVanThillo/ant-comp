#!/usr/bin/env python3
"""
Submit an array (SLURM or Modal) measuring each binder/target complex's interface
footprint and the geometry of its termini relative to that interface.

Each task runs ifacegeom_worker.py on a batch of designs: it selects the interface
residues on both sides, records the binder-side list (the epitope -- the footprint
every later step of the dimer campaign needs) and the target-side list with its
histidine count, then projects both termini onto the binder-COM -> interface-COM
axis. Use ``sapia collect ifacegeom`` to merge the metrics back into the table.

``action: update`` -- this is a property of a complex that already exists, so it
annotates the table in place rather than minting a generation. Its columns then
reach every later table through lineage, which is the whole point: an epitope list
in a sidecar file does not survive the generation boundary into the dock and
network tables, and that is exactly where it is needed.

Run it on whichever structure column holds the **complex** (``-i``): the ingested
BindCraft2 output for the inherited pool, or a ``boltz_path`` to measure the
interface of a prediction instead of the designed pose. Each file must contain
both binder and target -- there is no interface in a monomer. Label variants with
``-l`` so the columns do not collide.

``default_input_column`` is the ``"not applicable"`` sentinel (as in cms, usalign
and chainsel): there is no honest default structure column, and a wrong one fails
silently, so the builder refuses the run unless ``-i`` names a column the table has.

Several designs are packed into one task (``--designs-per-task``): the work is pure
geometry and runs in well under a second per design, while a container cold start
costs ~10 s, so a task per design would spend almost all of its time starting up.

Usage:
    sapia run ifacegeom outputs/20261002_dimer_binder \
        --table table0 \
        --input-column input_path \
        --binder-chains A --target-chains auto \
        -g 0
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from prosapia.core import CommonArgs, ManifestCtx
from prosapia.core.executors import volume_path

# The sentinel default_input_column (see the module docstring): matches no column,
# so the builder's check below always fires unless -i was given.
NO_DEFAULT_COLUMN = "not applicable"
METHODS = ("vector", "heavy")


class IfacegeomArgs(CommonArgs):
    binder_chains: str
    target_chains: str
    method: str
    contact_cutoff: float
    cb_dist_cut: float
    vector_dist_cut: float
    vector_angle_cut: float
    designs_per_task: int


def add_run_ifacegeom_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--binder-chains",
        type=str,
        default="A",
        help="The binder's chain IDs, comma-joined. Default 'A'. A chain named here "
        "but ABSENT from a design is an error for that design, never a smaller "
        "selection -- a silently narrower binder would make every number wrong.",
    )
    parser.add_argument(
        "--target-chains",
        type=str,
        default="auto",
        help="The target's chain IDs, comma-joined. 'auto' (default) takes every "
        "protein chain that is not a binder chain, which is right for a complex "
        "holding exactly one binder and one target of however many chains.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=METHODS,
        default="vector",
        help="How interface residues are selected. 'vector' (default) reproduces the "
        "idea of Rosetta's InterGroupInterfaceByVector: a residue is at the interface "
        "either because it touches the other side (--contact-cutoff) or because its "
        "CA->CB vector points at it (--vector-dist-cut / --vector-angle-cut). 'heavy' "
        "keeps the contact test alone. The vector test is what excludes residues that "
        "are merely nearby while facing away -- grafting those rotamers back later "
        "would be wasted work.",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=5.5,
        help="Heavy-atom distance (A) below which two residues count as in contact; "
        "the sole criterion under --method heavy, and the 'nearby' shortcut under "
        "'vector'. Default 5.5, Rosetta's nearby_atom_cut. Note this is an ATOM-to-"
        "atom distance: the ~8-11 A numbers usually quoted for "
        "InterGroupInterfaceByVector are its CB-CB cutoffs (--vector-dist-cut, "
        "--cb-dist-cut), not this one.",
    )
    parser.add_argument(
        "--cb-dist-cut",
        type=float,
        default=11.0,
        help="CB-CB distance (A) beyond which a residue pair is not considered at "
        "all. Default 11.0 (Rosetta's cb_dist_cut).",
    )
    parser.add_argument(
        "--vector-dist-cut",
        type=float,
        default=9.0,
        help="CB-CB distance (A) within which the pointing test applies. Default 9.0 "
        "(Rosetta's vector_dist_cut). Ignored under --method heavy.",
    )
    parser.add_argument(
        "--vector-angle-cut",
        type=float,
        default=75.0,
        help="Maximum angle (degrees) between a residue's CA->CB direction and the "
        "direction to the partner's CB for it to count as pointing at the other "
        "side. Default 75.0 (Rosetta's vector_angle_cut). Ignored under "
        "--method heavy.",
    )
    parser.add_argument(
        "--designs-per-task",
        type=int,
        default=200,
        help="Designs measured per task (default 200). The work is well under a "
        "second per design, so this exists to amortise the container cold start, not "
        "the computation. Lower it for more parallelism.",
    )


def _split(value: str) -> list[str]:
    """Comma-joined list -> stripped, non-empty tokens."""
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def build_ifacegeom_manifest(ctx: ManifestCtx[IfacegeomArgs]) -> list[tuple[str, ...]]:
    ctx.args.gpus_per_task = 0  # CPU-only tool

    binder = _split(ctx.args.binder_chains)
    if not binder:
        raise ValueError("--binder-chains must name at least one chain.")
    target_spec = ctx.args.target_chains.strip()
    if target_spec.lower() != "auto":
        target = _split(target_spec)
        if not target:
            raise ValueError("--target-chains must name at least one chain, or 'auto'.")
        if overlap := sorted(set(binder) & set(target)):
            raise ValueError(
                f"chains {overlap} are in both --binder-chains and --target-chains; "
                f"an interface needs two disjoint sides."
            )
    if ctx.args.designs_per_task < 1:
        raise ValueError("--designs-per-task must be >= 1.")
    if not 0.0 < ctx.args.vector_angle_cut <= 180.0:
        raise ValueError("--vector-angle-cut must be in (0, 180] degrees.")
    if ctx.args.vector_dist_cut > ctx.args.cb_dist_cut:
        raise ValueError(
            f"--vector-dist-cut ({ctx.args.vector_dist_cut}) is above --cb-dist-cut "
            f"({ctx.args.cb_dist_cut}), which prefilters it: the pointing test would "
            f"never see the pairs between the two."
        )

    column = ctx.args.input_column
    if column == NO_DEFAULT_COLUMN or column not in ctx.df.columns:
        available = ", ".join(str(c) for c in ctx.df.columns if str(c).endswith("_path"))
        raise ValueError(
            f"ifacegeom has no default input column: pass -i/--input-column with the "
            f"column holding the binder/target COMPLEX (got {column!r}, which table "
            f"'{ctx.args.table}' does not have). Structure columns available: "
            f"{available or '(none)'}."
        )

    ready = ctx.ready
    members: list[tuple[str, str]] = []
    for name in ready.index:
        name = cast(str, name)
        src = Path(str(ready.at[name, column]))
        if not src.exists():
            print(f"{name}: MISSING {src} (skipping)")
            continue
        # No CIF->PDB staging: the worker reads either format with gemmi.
        members.append((name, str(volume_path(src))))

    # One sub-manifest per task, and a top-level row per task pointing at it.
    tasks_dir = ctx.out_dir / "ifacegeom_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    x = ctx.args.designs_per_task
    manifest_rows: list[tuple[str, ...]] = []
    for t, i in enumerate(range(0, len(members), x)):
        task_file = tasks_dir / f"task_{t}.tsv"
        with open(task_file, "w") as f:
            for name, src in members[i : i + x]:
                f.write(f"{name}\t{src}\n")
        manifest_rows.append(
            (
                str(volume_path(task_file)),
                ",".join(binder),
                # 'auto' or an explicit list; never empty either way.
                target_spec if target_spec.lower() == "auto" else ",".join(_split(target_spec)),
                ctx.args.method,
                str(ctx.args.contact_cutoff),
                str(ctx.args.cb_dist_cut),
                str(ctx.args.vector_dist_cut),
                # Kept last: always non-empty.
                str(ctx.args.vector_angle_cut),
            )
        )

    return manifest_rows
