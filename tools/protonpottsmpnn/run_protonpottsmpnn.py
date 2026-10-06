"""Submit tasks running Proton-PottsMPNN pH-switch redesign, spawning a child table.

Proton-PottsMPNN (Jacobsen et al. 2026, https://github.com/christian-creator/ProtonPottsMPNN)
is PottsMPNN with an explicit PROTONATION-STATE alphabet: histidine is ``HIS-P``
(charged) vs ``HIS-S`` (neutral), acids are ``ASP-P``/``GLU-P`` (protonated, neutral
COOH) vs ``ASP-D``/``GLU-D`` (deprotonated, -1). Because the learned Potts energy is
protonation-aware, the design engine can PIN protonated centres and redesign their
neighbourhood so that binding switches with pH.

So this is a sequence designer in the same slot as ``proteinmpnn`` / ``atomium`` -- it
reads a backbone and mints child rows carrying a designed binder sequence -- but it
optimises a DIFFERENT objective: a weighted trade-off between stability and pH
selectivity, one design per lambda.

    O = (1 - lambda) * zscore(H_stab) + lambda * zscore(sum selective)

``--num-designs`` sweeps lambda from ``--lambda-min`` (pure stability) to
``--lambda-max`` (pure selectivity), which is exactly the Pareto sweep in the repo's
``inference/design_ph.py``. The collector marks the Pareto-optimal designs of each
backbone, so one run gives you the whole trade-off curve and a filterable front.

PREMISE -- read this before trusting any number
-----------------------------------------------
  * The input structure is a BINDER + TARGET COMPLEX, and ``--binder-chain`` names the
    binder. Only the binder chain is redesigned; every other chain is held fixed and
    seen by the encoder as context. On a single-chain structure the tool still runs,
    but ``--placement-region interface`` has NO candidate positions (the interface mask
    is built from binder-CA/target-CA contacts) and the design is a plain monomer
    redesign, not a binder redesign. The builder raises rather than let that pass.
  * The protonation labels the model conditions on come from the EV6 FLAML labeller,
    which reads H-bond geometry from HBPLUS. HBPLUS is an external C binary that is NOT
    pip-installable and is NOT bundled -- see ``modal_image.py``. The repo's README
    claims design does not need it; that is WRONG. ``prepare_potts_input`` builds the
    inference pipeline through ``get_protonation_state_transforms``, which runs
    ``CalculateHbondsPlus`` unconditionally (``pipelines/potts_mpnn.py:269``), so every
    design call shells out to HBPLUS. Verified by reading the pipeline, not by prose.
  * Energies are MODEL energies on the INPUT backbone, in arbitrary units, and are only
    comparable WITHIN one backbone. ``potts_energy`` of design A on backbone 1 says
    nothing against design B on backbone 2 -- the z-scoring that makes lambda meaningful
    is per-backbone. Rank within a parent, never across the whole table.
  * This tool DESIGNS. It does not fold, and it does not measure binding. A low
    ``selective_energy`` is a model's claim that the protonated microstate is preferred
    at the pinned centres -- not evidence of a pH switch. Fold with ``boltz`` (via
    ``mkcomplex``) and score the interface with ``cms`` / ``pyrosetta`` as usual.

What this tool deliberately does NOT wrap
-----------------------------------------
  * ``labeller/`` -- standalone protonation labelling of a PDB. Different question
    (annotate a structure, not design a sequence); would be an ``update`` tool.
  * ``scoring/`` + ``inference/fold_rf3.py`` -- RF3 folding and pH-bond / charge-clash
    read-outs of a fold. RF3's ~3 GB weights are not shipped, and folding is ``boltz``'s
    job in this workspace.
  * ``benchmarks/``, ``training/`` -- not pipeline steps.

Centre mini-language (``--explicit-centers``): ``,`` separates centres, each is
``<resnum>:<STATE>``, and the residue number may be a ``{...}`` island resolved up the
lineage with ``+ - * //`` arithmetic (the usual prosapia expression mini-language):

    --explicit-centers '45:HIS-P,78:ASP-P'
    --explicit-centers '{motif_end}:HIS-P'       # per-design, from a parent column

Residue numbers are the INPUT STRUCTURE's own ``res_id`` values, NOT 1..L sequence
positions -- the engine matches centres by ``res_id`` on the parsed atom array. A
generator that renumbered the binder from 1 will therefore take ``45`` to mean the 45th
residue; the collector reports ``resnum_offset`` so you can tell which world you are in.
With no ``--explicit-centers``, ``--center-types`` gives the COMPOSITION to place and
``--placement-by`` chooses the positions.

Usage:
    # 8 designs along the stability<->selectivity front, centres placed at the interface
    sapia run protonpottsmpnn outputs/<run> -t table0 --table-label ph \\
        --binder-chain A --num-designs 8 --placement-region interface

    # pin the centres by hand, redesign around them
    sapia run protonpottsmpnn outputs/<run> -t table0 \\
        --explicit-centers '45:HIS-P,78:ASP-P' --num-designs 4

    # redesign ProteinMPNN sequences instead of the native backbone sequence
    sapia run protonpottsmpnn outputs/<run> -t table1 -i rfdiffusion3_path \\
        --seed-column proteinmpnn_sequence
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

from prosapia.core import CommonArgs, ManifestCtx
from prosapia.core.executors import volume_path
from prosapia.utils import ensure_pdb, polymer_chain_names, resolve_template

# The protonated microstates the v6 checkpoint knows, and the deprotonated contrast
# each one is scored against. v6 has NO HID/HIE tautomers -- neutral His is HIS-S.
DEP_MAP: dict[str, list[str]] = {
    "HIS-P": ["HIS-S"],
    "ASP-P": ["ASP-D"],
    "GLU-P": ["GLU-D"],
}
CENTER_STATES = tuple(DEP_MAP)

# Only the AMBIGUOUS microstates are forbidden: a residue the labeller's five folds
# disagree about (-A) is not a state the design should ever commit to.
FORBIDDEN_TOKENS = ["HIS-A", "ASP-A", "GLU-A", "UNK"]

PLACEMENT_REGIONS = ("all", "interface", "core", "surface")
PLACEMENT_BY = ("scan_potts", "scan_mpnn", "random")

# Where the shipped v6 checkpoint lives in the Modal image (see modal_image.py). The
# checkpoint is 21 MB and ships IN the repo, so there is no weights Volume.
DEFAULT_CHECKPOINT = (
    "/opt/protonpottsmpnn/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt"
)


class ProtonPottsMPNNArgs(CommonArgs):
    binder_chain: str
    num_designs: int
    samples_per_design: int
    lambda_min: float
    lambda_max: float
    center_types: str
    explicit_centers: str
    placement_region: str
    placement_by: str
    neighbour_k: int
    max_mutations: int
    block_size: int
    temperature: float
    seed: int
    seed_column: str
    checkpoint: str
    n_jobs: int
    set: list[str]


def _parse_center_types(spec: str) -> list[str]:
    """``'HIS-P,ASP-P'`` -> the exact composition to place, validated against v6."""
    types = [t.strip() for t in spec.split(",") if t.strip()]
    if not types:
        raise ValueError(
            "--center-types: at least one protonated state is required "
            f"(choose from {', '.join(CENTER_STATES)}), or pass --explicit-centers"
        )
    for t in types:
        if t not in CENTER_STATES:
            raise ValueError(
                f"--center-types: {t!r} is not a protonated state the v6 checkpoint "
                f"knows; choose from {', '.join(CENTER_STATES)}. (The DEPROTONATED "
                f"contrasts HIS-S/ASP-D/GLU-D are what each one is scored against and "
                f"are never placed directly.)"
            )
    return types


def _parse_explicit_centers(spec: str, lookup, name: str) -> list[dict[str, Any]]:
    """``'45:HIS-P,{motif_end}:ASP-P'`` -> the engine's ``explicit_centers`` list.

    Residue numbers go through ``resolve_template`` so a ``{...}`` island reads a column
    up this design's lineage. Resolution happens HERE, at manifest-build time, so a typo
    raises before a single container starts.
    """
    if not spec.strip():
        return []
    centers: list[dict[str, Any]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"--explicit-centers: malformed centre {token!r} "
                f"(expected '<resnum>:<STATE>', e.g. '45:HIS-P')"
            )
        resnum_spec, state = token.rsplit(":", 1)
        state = state.strip()
        if state not in CENTER_STATES:
            raise ValueError(
                f"--explicit-centers: {state!r} is not a protonated state the v6 "
                f"checkpoint knows; choose from {', '.join(CENTER_STATES)}"
            )
        resolved = resolve_template(resnum_spec.strip(), lookup, name)
        try:
            res_id = int(resolved)
        except ValueError:
            raise ValueError(
                f"--explicit-centers: residue number {resolved!r} (from {resnum_spec!r}) "
                f"is not an integer for design {name}"
            )
        centers.append({"res_id": res_id, "protonation_type": state})
    return centers


def _parse_regions(spec: str) -> list[str]:
    regions = [r.strip() for r in spec.split(",") if r.strip()]
    if not regions:
        raise ValueError("--placement-region: at least one region is required")
    for r in regions:
        if r not in PLACEMENT_REGIONS:
            raise ValueError(
                f"--placement-region: {r!r} is not a region; choose from "
                f"{', '.join(PLACEMENT_REGIONS)}"
            )
    return regions


def _parse_set(tokens: list[str]) -> dict[str, Any]:
    """``--set block_max_rounds=20`` -> ``{'block_max_rounds': 20}``.

    An escape hatch onto ``PHDesignCriteria`` fields this tool does not expose. Values
    are JSON-parsed when possible so numbers, booleans and lists survive; anything else
    stays a string. The worker validates the field name against the dataclass, so a typo
    fails loudly on the first task rather than being silently ignored.
    """
    extra: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(
                f"--set: malformed token {token!r} (expected 'field=value', "
                f"e.g. --set block_max_rounds=20)"
            )
        key, value = token.split("=", 1)
        key = key.strip()
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = value
    return extra


def _lambdas(args: ProtonPottsMPNNArgs) -> list[float]:
    """The lambda ladder: ``--num-designs`` points from lambda-min to lambda-max.

    One design per lambda is the repo's own sweep. A single design takes lambda-min
    (NOT the midpoint) so ``--num-designs 1`` is a plain, reproducible stability design
    unless the caller says otherwise.
    """
    n = args.num_designs
    if n < 1:
        raise ValueError(f"--num-designs: must be at least 1, got {n}")
    lo, hi = args.lambda_min, args.lambda_max
    for label, value in (("--lambda-min", lo), ("--lambda-max", hi)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label}: must be in [0, 1], got {value}")
    if hi < lo:
        raise ValueError(f"--lambda-max ({hi}) must be >= --lambda-min ({lo})")
    if n == 1 or hi == lo:
        return [round(lo, 4)] * n
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 4) for i in range(n)]


def _seed_sequence(ctx: ManifestCtx[ProtonPottsMPNNArgs], name: str) -> str:
    """The binder sequence the redesign starts from, or "" for the native one.

    ``--seed-column`` is resolved up the lineage. A named column that resolves to
    nothing is an error, not a silent fallback to native -- the two give different
    designs and the table would not record which one happened.
    """
    column = ctx.args.seed_column
    if not column:
        return ""
    value = ctx.lookup(name, column)
    if value is None or (isinstance(value, float) and value != value) or str(value) == "":
        raise ValueError(
            f"--seed-column {column!r} is empty for design {name}. Either the column "
            f"does not exist on this row's lineage or the upstream step did not "
            f"succeed; drop the flag to redesign from the native backbone sequence."
        )
    return str(value)


def build_protonpottsmpnn_manifest(
    ctx: ManifestCtx[ProtonPottsMPNNArgs],
) -> list[tuple[str, ...]]:
    # The design engine is CPU-only by construction: run_ph_redesign fans the lambda
    # sweep across a fork pool, and forking after CUDA init is unsafe (the engine only
    # parallelises on a CPU device). Forcing 0 here means callers never need -g 0.
    ctx.args.gpus_per_task = 0

    args = ctx.args

    # Guardrails: everything cheap, everything before submission.
    lambdas = _lambdas(args)
    regions = _parse_regions(args.placement_region)
    extra = _parse_set(args.set)
    if args.placement_by not in PLACEMENT_BY:
        raise ValueError(
            f"--placement-by: {args.placement_by!r} is not a strategy; choose from "
            f"{', '.join(PLACEMENT_BY)}"
        )
    center_types = [] if args.explicit_centers else _parse_center_types(args.center_types)
    if args.samples_per_design < 1:
        raise ValueError(
            f"--samples-per-design: must be at least 1, got {args.samples_per_design}"
        )

    configs_dir = ctx.out_dir / "protonpottsmpnn_configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple[str, ...]] = []
    for design_name in sorted(cast(str, n) for n in ctx.ready.index):
        input_path = Path(str(ctx.ready.at[design_name, args.input_column]))
        pdb_src = ensure_pdb(input_path, args.run_dir)

        # The binder/target premise, checked per design against the real structure.
        # polymer_chain_names reads the file, so this also catches a path that exists
        # but holds no polymer.
        chains = polymer_chain_names(pdb_src)
        if args.binder_chain not in chains:
            raise ValueError(
                f"{design_name}: --binder-chain {args.binder_chain!r} is not a polymer "
                f"chain of {pdb_src} (it has {', '.join(chains) or 'none'}). Name the "
                f"chain you want redesigned; every other chain is held fixed as target "
                f"context."
            )
        if len(chains) < 2 and "interface" in regions:
            raise ValueError(
                f"{design_name}: --placement-region includes 'interface' but "
                f"{pdb_src} has only chain {args.binder_chain!r}. The interface mask is "
                f"built from binder-CA/target-CA contacts, so it would be EMPTY and no "
                f"centre could be placed there. Pass a binder+target complex, or use "
                f"--placement-region all/core/surface."
            )

        config = {
            "name": design_name,
            "pdb": str(volume_path(pdb_src)),
            "binder_chain": args.binder_chain,
            "checkpoint": args.checkpoint or DEFAULT_CHECKPOINT,
            "out_dir": str(volume_path(ctx.out_dir / design_name)),
            "lambdas": lambdas,
            "samples_per_design": args.samples_per_design,
            "center_types": center_types,
            "explicit_centers": _parse_explicit_centers(
                args.explicit_centers, ctx.lookup, design_name
            ),
            "dep_map": DEP_MAP,
            "forbidden_tokens": FORBIDDEN_TOKENS,
            "placement_region": regions,
            "placement_by": args.placement_by,
            "neighbour_k": args.neighbour_k,
            "max_mutations": args.max_mutations,
            "block_size": args.block_size,
            "temperature": args.temperature,
            "seed": args.seed,
            "seed_sequence": _seed_sequence(ctx, design_name),
            "n_jobs": args.n_jobs,
            "extra": extra,
        }
        config_path = configs_dir / f"{design_name}.json"
        config_path.write_text(json.dumps(config, indent=2))
        manifest_rows.append((design_name, str(volume_path(config_path))))

    return manifest_rows


def add_run_protonpottsmpnn_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--binder-chain",
        type=str,
        default="A",
        help="Chain to redesign (default A). Every OTHER chain in the structure is held "
        "fixed and seen by the encoder as target context -- so this flag is what makes "
        "the run a binder redesign rather than a monomer redesign. Must be a polymer "
        "chain of the input, checked per design before submission.",
    )
    parser.add_argument(
        "--num-designs",
        type=int,
        default=8,
        help="Designs per backbone (default 8). Each is one point on the lambda ladder "
        "from --lambda-min to --lambda-max, i.e. one stability/selectivity trade-off -- "
        "this is the repo's own Pareto sweep, and the collector marks which designs are "
        "Pareto-optimal within each backbone.",
    )
    parser.add_argument(
        "--samples-per-design",
        type=int,
        default=1,
        help="Stochastic draws at EACH lambda (default 1). The block-descent readout is "
        "near-deterministic at the default --temperature, so >1 mostly pays off once you "
        "raise the temperature. Total rows per backbone = --num-designs x this.",
    )
    parser.add_argument(
        "--lambda-min",
        type=float,
        default=0.0,
        help="Low end of the lambda ladder (default 0.0 = pure stability). lambda is the "
        "RELATIVE weight of selectivity against stability in O = (1-l)*z(H_stab) + "
        "l*z(sum selective); both terms are z-scored per backbone, so it is a true "
        "relative weight and not an energy.",
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1.0,
        help="High end of the lambda ladder (default 1.0 = pure pH selectivity). Set "
        "equal to --lambda-min to run every design at one fixed lambda.",
    )
    parser.add_argument(
        "--center-types",
        type=str,
        default="HIS-P,ASP-P,GLU-P",
        help="The exact COMPOSITION of protonated centres to place, comma-separated "
        "(default 'HIS-P,ASP-P,GLU-P' = one of each). Repeat a state to place several "
        "('HIS-P,HIS-P'). Placement chooses WHERE via --placement-by/--placement-region; "
        "this only fixes WHAT. Valid states: HIS-P, ASP-P, GLU-P -- the deprotonated "
        "contrasts (HIS-S/ASP-D/GLU-D) are what each is scored against, never placed. "
        "Ignored when --explicit-centers is given.",
    )
    parser.add_argument(
        "--explicit-centers",
        type=str,
        default="",
        help="Pin the centres BY HAND instead of letting placement choose: comma-separated "
        "'<resnum>:<STATE>' (e.g. '45:HIS-P,78:ASP-P'). Residue numbers are the INPUT "
        "STRUCTURE's own res_id values, not 1..L sequence positions, and may be {...} "
        "islands resolved up the lineage with + - * // arithmetic (e.g. "
        "'{motif_end}:HIS-P'). Overrides --center-types. Empty (default): place "
        "automatically.",
    )
    parser.add_argument(
        "--placement-region",
        type=str,
        default="all",
        help="Where centres may be placed, comma-separated subset of "
        "all,interface,core,surface (default all). 'interface' = binder residues with a "
        "target CA within 6 A; 'core'/'surface' split on residue RASA at 0.2. "
        "'interface' REQUIRES a binder+target complex -- the builder raises on a "
        "single-chain input rather than let an empty mask pass silently.",
    )
    parser.add_argument(
        "--placement-by",
        type=str,
        default="scan_potts",
        choices=list(PLACEMENT_BY),
        help="How candidate positions are ranked within the region (default scan_potts): "
        "'scan_potts' ranks by the Potts gap dE(protonated) - dE(deprotonated), "
        "'scan_mpnn' by the decoder log-likelihood gap, 'random' uniformly.",
    )
    parser.add_argument(
        "--neighbour-k",
        type=int,
        default=16,
        help="Per-centre cap on how many coupled kNN neighbours become designable "
        "(default 16). This is the redesign EXTENT around each centre; 0 = the full "
        "neighbourhood.",
    )
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=20,
        help="Hard cap on the TOTAL designable positions across all centres (default 20), "
        "i.e. a direct mutation budget. 0 = no cap. Keep it low to stay close to a "
        "validated parent sequence; raise it to let the engine rebuild the pocket.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=3,
        help="Block-descent block size (default 3 = triples). Each block enumerates all "
        "V**block_size joint assignments, so cost grows exponentially: 1 = greedy ICM, "
        "2 = pairwise, 3 = triples. Only the stability term has within-block pairwise "
        "coupling, so this changes the answer only through stability.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="Block readout temperature (default 0.05, near-deterministic). 0 = argmin; "
        ">0 samples the block from softmax(-J/T). Raise it together with "
        "--samples-per-design to get diversity at a fixed lambda.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default 0). Unlike atomium, 0 is a real seed here, not "
        "'pick one at random'.",
    )
    parser.add_argument(
        "--seed-column",
        type=str,
        default="",
        help="Table column holding the binder sequence to redesign FROM, resolved up the "
        "lineage (e.g. proteinmpnn_sequence, atomium_sequence). Empty (default): start "
        "from the input structure's own native sequence. A named column that is empty on "
        "a row is an error, not a silent fallback -- the two starting points give "
        "different designs and the table has to record which one ran.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=f"Path to the Proton-PottsMPNN checkpoint. Empty (default) uses the v6 "
        f"checkpoint shipped in the repo at {DEFAULT_CHECKPOINT}. The checkpoint's "
        f"extended_vocab MUST be v6 or the 30-token weight load fails.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="CPU workers the lambda sweep is fanned across within one task "
        "(default 0 = use --cpus-per-task). The backbone is featurised ONCE and the "
        "workers share it read-only via fork, so this is close to free parallelism.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Set any other PHDesignCriteria field verbatim (repeatable), e.g. "
        "--set block_max_rounds=20 --set sweep_order=knn --set self_weight=0.5. Values "
        "are JSON-parsed when possible. An unknown field name fails the task loudly "
        "rather than being ignored.",
    )
