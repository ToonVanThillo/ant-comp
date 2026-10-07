#!/usr/bin/env python3
"""
Submit an array (SLURM or Modal) measuring, per design, the shortest route between
two chain termini that stays out of the protein.

Premise, which is the tool's whole contract: *measure the shortest route between two
chain termini that stays out of the protein -- the length a flexible linker would
actually have to span.* A straight-line C-term-to-N-term distance is only a LOWER
BOUND; if the direct vector passes through the protein body the linker has to go
around, and this tool computes the shortest path through solvent-accessible space.

Each task runs linkpath_worker.py on a batch of designs. Per design it

  1. reads the structure and builds an occupancy grid over the chosen OBSTACLE
     chains (blocked = within vdW + ``--probe`` of an atom),
  2. picks the two anchor atoms (``--from-chain``/``--to-chain``, C and N backbone
     atoms of the terminal -- or named -- residues),
  3. A*-searches the free voxels between them and reports the route length, the
     detour over the straight line, and the linker residue count that implies,
  4. optionally routes the REVERSE pair as well, whose difference from the forward
     route is a free trust metric under C2 symmetry,
  5. writes the route as a PDB of pseudo-atoms for PyMOL.

``action: update`` -- this is a property of structures that already exist, so it
annotates the table in place rather than minting a generation. ``linkpath_path`` is
the pseudo-atom route, for inspection; nothing downstream designs against it.

What it does NOT do
-------------------
* It does **not** model the linker. No sequence, no conformational sampling, no
  energy: ``n_res_min``/``n_res_relaxed`` are a contour-length division, nothing more.
* It does **not** move anything. The structure is rigid; real termini are not.
* It does **not** decide the obstacle set. ``--obstacle-chains`` is the scientific
  content of the measurement and the resolved list is written back as a column.
* It is **not** a linker designer and **not** a loop closure test.
* It does **not** supersede ``dimerfit``'s straight-line ``link_dist`` -- it bounds
  it from above. ``straight_ca_dist`` is collected specifically so the two are
  comparable (see the skill).

``default_input_column`` is the ``"not applicable"`` sentinel (as in dimerfit,
ifacegeom, cms and chainsel): there is no honest default structure column, and a
wrong one fails silently, so the builder refuses the run unless ``-i`` names a column
the table has.

Several designs are packed into one task (``--designs-per-task``): measured at
~0.16 s per design on a 216-residue C2 dimer at the default grid, while a container
cold start costs ~10 s.

Usage:
    sapia run linkpath outputs/20261002_143419_dimer_phase2 \
        --table table1 \
        --input-column dimerfit_path \
        --from-chain auto --to-chain auto \
        --obstacle-chains all --probe 2.0
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from prosapia.core import CommonArgs, ManifestCtx
from prosapia.core.executors import volume_path
from prosapia.utils import resolve_template

# The sentinel default_input_column (see the module docstring): matches no column, so
# the builder's check below always fires unless -i was given.
NO_DEFAULT_COLUMN = "not applicable"
ENDS = ("C", "N")


class LinkpathArgs(CommonArgs):
    from_chain: str
    to_chain: str
    from_res: str
    to_res: str
    from_end: str
    to_end: str
    obstacle_chains: str
    include_h: bool
    protein_only: bool
    spacing: float
    probe: float
    pad: float
    carve: float
    taut_rise: float
    relaxed_rise: float
    both_directions: bool
    residue_estimate: bool
    model: int
    designs_per_task: int


def add_run_linkpath_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--from-chain",
        type=str,
        default="auto",
        help="Chain donating the --from-end terminus (the C-terminus by default). "
        "'auto' (default) requires EXACTLY TWO protein chains and takes the first; "
        "on any other chain count 'auto' is an error for that design, never a guess. "
        "Default 'auto'.",
    )
    parser.add_argument(
        "--to-chain",
        type=str,
        default="auto",
        help="Chain accepting the --to-end terminus (the N-terminus by default). "
        "'auto' (default) takes the SECOND protein chain, under the same "
        "exactly-two-chains rule as --from-chain.",
    )
    parser.add_argument(
        "--from-res",
        type=str,
        default="",
        help="Residue NUMBER to anchor on in --from-chain, instead of that chain's "
        "own terminus. Accepts the {expr} mini-language (e.g. '{binder_len}'), "
        "resolved per design up the lineage. Default: the last amino acid of the "
        "chain (for --from-end C).",
    )
    parser.add_argument(
        "--to-res",
        type=str,
        default="",
        help="Residue NUMBER to anchor on in --to-chain. Accepts {expr}. Default: "
        "the first amino acid of the chain (for --to-end N).",
    )
    parser.add_argument(
        "--from-end",
        type=str,
        choices=ENDS,
        default="C",
        help="Which terminus of --from-chain the route leaves from: 'C' uses the "
        "backbone C atom (falling back to CA), 'N' the backbone N. Default 'C' -- a "
        "fusion linker grows out of a C-terminus.",
    )
    parser.add_argument(
        "--to-end",
        type=str,
        choices=ENDS,
        default="N",
        help="Which terminus of --to-chain the route arrives at. Default 'N'.",
    )
    parser.add_argument(
        "--obstacle-chains",
        type=str,
        default="all",
        help="THE scientifically load-bearing flag: the chains whose atoms the route "
        "has to go around, comma-joined. 'all' (default) is every chain in the input "
        "file, which is what you want when the file holds exactly the bodies the "
        "linker must avoid. Name a subset to route a dimer through only its own two "
        "protomers while a target sits in the same file -- but understand that a "
        "narrower obstacle set only ever makes the answer SHORTER. The resolved list "
        "is written back as linkpath_obstacle_chains; a chain named but absent is an "
        "error for that design, never a smaller selection.",
    )
    parser.add_argument(
        "--include-h",
        action="store_true",
        help="Treat hydrogens as obstacles too. Off by default -- most structures "
        "here are heavy-atom only, and a mixed set would make designs incomparable.",
    )
    parser.add_argument(
        "--protein-only",
        action="store_true",
        help="Ignore everything that is not an amino acid (ligands, cofactors, ions, "
        "nucleic acids) when building the obstacle grid. OFF by default: a cofactor "
        "in the way is in the way, and turning this on can only SHORTEN the route. "
        "Waters are never obstacles either way.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Grid spacing in A (default 1.0). This is the precision/cost lever: the "
        "voxel count -- and the runtime -- goes as 1/spacing^3. 1.5 is ~2x faster "
        "and ~3 A noisier on a 50 A route (measured); below 0.75 the cost rises "
        "faster than the answer improves.",
    )
    parser.add_argument(
        "--probe",
        type=float,
        default=2.0,
        help="Probe radius in A: how fat the thing being routed is (default 2.0). "
        "This is a deliberate campaign decision and differs from the 1.4 A WATER "
        "probe of the linker_path.py prototype: a polypeptide backbone is thicker "
        "than a water molecule, and a 1.4 A probe lets the route thread crevices a "
        "real chain cannot enter, which UNDERSTATES the detour. Measured on a C2 "
        "barnase dimer: probe 1.4 gave 50.5 A, probe 2.0 gave 52.8 A for the same "
        "termini. Use 1.4 only to reproduce the prototype.",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=15.0,
        help="Padding in A around the structure's bounding box (default 15), so the "
        "route can leave the surface and go around the outside. Raise it if a route "
        "looks pinned to the surface.",
    )
    parser.add_argument(
        "--carve",
        type=float,
        default=3.0,
        help="Radius in A of the bubble freed around each anchor atom so the chain "
        "can leave its own terminus (default 3.0). NOTE: the effective radius is "
        "raised to at least vdW(anchor) + --probe (3.70 A for a backbone C at probe "
        "2.0), because otherwise the anchor atom -- which sits at the bubble's "
        "centre -- wraps its own bubble in an unbroken blocked shell and the route "
        "can only escape by a grid artifact.",
    )
    parser.add_argument(
        "--taut-rise",
        type=float,
        default=3.5,
        help="A per residue of a FULLY EXTENDED polypeptide (default 3.5), the "
        "divisor behind n_res_min. That is an absolute floor: a linker at contour "
        "length is a taut string, not a linker.",
    )
    parser.add_argument(
        "--relaxed-rise",
        type=float,
        default=2.1,
        help="A per residue of a RELAXED coil held at <=60%% of contour length "
        "(default 2.1), the divisor behind n_res_relaxed. This is the number to "
        "design against.",
    )
    parser.add_argument(
        "--no-both-directions",
        dest="both_directions",
        action="store_false",
        help="Skip the reverse route (C-terminus of --to-chain -> N-terminus of "
        "--from-chain). It is ON by default because under exact C2 symmetry the two "
        "routes must be equal, so their difference (linkpath_path_asymmetry) is a "
        "FREE trust metric that costs one extra A* and catches a broken grid, a "
        "broken frame or a structure that is not actually symmetric.",
    )
    parser.set_defaults(both_directions=True)
    parser.add_argument(
        "--no-residue-estimate",
        dest="residue_estimate",
        action="store_false",
        help="Record distances only: n_res_min and n_res_relaxed come back NA. Use "
        "it when the contour-length conversion is not the question and you do not "
        "want a divided number read as a design specification.",
    )
    parser.set_defaults(residue_estimate=True)
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="Model index for a multi-model file (default 0, the first).",
    )
    parser.add_argument(
        "--designs-per-task",
        type=int,
        default=50,
        help="Designs measured per task (default 50). Measured at ~0.16 s per design "
        "for a 216-residue dimer at --spacing 1.0, so this exists to amortise the "
        "container cold start. Lower it if you raise --spacing's cost (0.5 A is ~5x "
        "slower) or the structures are much larger.",
    )


def build_linkpath_manifest(ctx: ManifestCtx[LinkpathArgs]) -> list[tuple[str, ...]]:
    ctx.args.gpus_per_task = 0  # CPU-only tool

    if ctx.args.designs_per_task < 1:
        raise ValueError("--designs-per-task must be >= 1.")
    for flag, value in (
        ("--spacing", ctx.args.spacing),
        ("--probe", ctx.args.probe),
        ("--carve", ctx.args.carve),
        ("--taut-rise", ctx.args.taut_rise),
        ("--relaxed-rise", ctx.args.relaxed_rise),
    ):
        if value <= 0:
            raise ValueError(f"{flag} must be > 0 (got {value}).")
    if ctx.args.pad < 0:
        raise ValueError(f"--pad must be >= 0 (got {ctx.args.pad}).")
    if ctx.args.relaxed_rise > ctx.args.taut_rise:
        raise ValueError(
            f"--relaxed-rise ({ctx.args.relaxed_rise}) must not exceed --taut-rise "
            f"({ctx.args.taut_rise}): a relaxed coil covers LESS distance per "
            f"residue than a fully extended one, so n_res_relaxed must be the "
            f"larger count."
        )

    from_chain = ctx.args.from_chain.strip()
    to_chain = ctx.args.to_chain.strip()
    if not from_chain or not to_chain:
        raise ValueError("--from-chain and --to-chain must each name a chain or 'auto'.")
    if (
        from_chain.lower() != "auto"
        and from_chain == to_chain
        and ctx.args.from_end == ctx.args.to_end
    ):
        raise ValueError(
            f"--from-chain and --to-chain are both {from_chain!r} and both ends are "
            f"{ctx.args.from_end!r}: that routes a terminus to itself."
        )

    obstacles = ctx.args.obstacle_chains.strip()
    if obstacles.lower() != "all":
        chains = [tok.strip() for tok in obstacles.split(",") if tok.strip()]
        if not chains:
            raise ValueError(
                "--obstacle-chains must name at least one chain, or 'all'."
            )
        obstacles = ",".join(chains)

    column = ctx.args.input_column
    if column == NO_DEFAULT_COLUMN or column not in ctx.df.columns:
        available = ", ".join(str(c) for c in ctx.df.columns if str(c).endswith("_path"))
        raise ValueError(
            f"linkpath has no default input column: pass -i/--input-column with the "
            f"column holding the STRUCTURE to route through (got {column!r}, which "
            f"table '{ctx.args.table}' does not have). Structure columns available: "
            f"{available or '(none)'}."
        )

    ready = ctx.ready
    members: list[tuple[str, str, str, str]] = []
    for name in ready.index:
        name = cast(str, name)
        src = Path(str(ready.at[name, column]))
        if not src.exists():
            print(f"{name}: MISSING {src} (skipping)")
            continue
        # {expr} is resolved HERE, at submit time, so a bad column name fails before
        # anything is queued. '-' is the worker's sentinel for "the chain's own
        # terminus"; a manifest field is never empty.
        from_res = resolve_template(ctx.args.from_res, ctx.lookup, name).strip() or "-"
        to_res = resolve_template(ctx.args.to_res, ctx.lookup, name).strip() or "-"
        members.append((name, str(volume_path(src)), from_res, to_res))

    # One sub-manifest per task, and a top-level row per task pointing at it. The row
    # carries the task's design COUNT so the worker can refuse a half-visible task
    # file instead of quietly running short (see the guard in linkpath.sh).
    tasks_dir = ctx.out_dir / "linkpath_tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    x = ctx.args.designs_per_task
    manifest_rows: list[tuple[str, ...]] = []
    for t, i in enumerate(range(0, len(members), x)):
        batch = members[i : i + x]
        task_file = tasks_dir / f"task_{t}.tsv"
        with open(task_file, "w") as f:
            for row in batch:
                f.write("\t".join(row) + "\n")
        manifest_rows.append(
            (
                str(volume_path(task_file)),
                str(len(batch)),
                from_chain,
                to_chain,
                ctx.args.from_end,
                ctx.args.to_end,
                obstacles,
                "1" if ctx.args.include_h else "0",
                "1" if ctx.args.protein_only else "0",
                str(ctx.args.spacing),
                str(ctx.args.probe),
                str(ctx.args.pad),
                str(ctx.args.carve),
                str(ctx.args.taut_rise),
                str(ctx.args.relaxed_rise),
                "1" if ctx.args.both_directions else "0",
                "1" if ctx.args.residue_estimate else "0",
                # Kept last: always non-empty, so the manifest line never ends on an
                # empty field.
                str(ctx.args.model),
            )
        )

    return manifest_rows
