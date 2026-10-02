#!/usr/bin/env python3
"""
Submit an array (SLURM or Modal) that designs BURIED HYDROGEN-BOND NETWORKS onto
each input backbone with HBDesigner, spawning a new child table.

PREMISE -- read this before using the numbers it writes.

HBDesigner (RosettaCommons/HBDesigner, MIT; PyRosetta is free for academic use) is
a GNN + PyRosetta algorithm that picks a small set of positions on an EXISTING
backbone and assigns polar residues there such that they form one connected,
buried hydrogen-bond network. It designs ``--n-res`` positions (2-6) and NOTHING
else: every other position in the output structure is written as GLYCINE. The
output is therefore a *network stub on a backbone*, not a foldable sequence. It is
an input to sequence design, not a replacement for it -- the intended chain is

    rfdiffusion3 -> hbdesigner -> proteinmpnn (network positions fixed) -> boltz

and the ``hbdesigner_fixed_positions`` / ``hbdesigner_fix<i>`` columns exist to
make that hand-off exact.

Its numbers are meaningless if:

  * you read the output as a designed protein. It is poly-glycine outside the
    network (``to_pdb(unk_to_gly=True)``, verified in ``rank_and_save``). Anything
    that computes on the sequence -- a predictor, a composition filter, a
    hydrophobicity score -- will see poly-Gly and be wrong, not broken-looking.
  * the backbone has no buried volume. HBDesigner buries a network; on a backbone
    with no core (a short helix, an extended peptide) it will find nothing and the
    run collects as ``error: no networks`` rather than as a bad score.
  * you designed one side of an interface without grafting the partner back in
    (see ``--graft-chains``) -- the partner chain returns as poly-glycine and any
    downstream interface metric is computed against a chain that is not there.

action ``create``: each input backbone yields up to ``--top-k`` RANKED designs,
each a different network, i.e. new entities -- so it mints a child table with rows
``<parent>_hb1 .. <parent>_hb<k>`` (mirroring proteinmpnn's ``<parent>_f<i>``).
Rank 1 is the best by HBDesigner's own ordering (fewest buried unsats, then
highest saturation, then lowest HB_Score_full).

WHAT IT DOES NOT DO: it does not design the rest of the sequence, it does not
relax or validate the fold, it does not predict a structure, and it does not score
an interface. It has no opinion on whether the network is worth having -- the
filters it applies (``--max-buns``, ``--min-sat``, ...) are the only judgement in
the tool, and they are yours to set.

RESIDUE NUMBERING -- the one thing that silently goes wrong here.
HBDesigner speaks PDB chain+resnum (``--guide-res A12,B13``, ``--anchor-res B5``,
and the ``network`` string it reports, e.g. ``A12S:A16T``). ProteinMPNN speaks
1-based index WITHIN each parsed chain. They agree only when the input PDB is
numbered 1..L per chain -- which rfdiffusion3 output happens to be, and a trimmed
crystal target is not. The collector therefore never copies HBDesigner's resnums
into ``fixed_positions``: it maps them through the output PDB's own residue order
and reports the mapping as ``resnum_offset`` / ``resnum_shift_max`` so you can see
whether a shift happened.

Usage:
    # monomer: 3-residue network on each rfd3 backbone, 5 ranked designs each
    sapia run hbdesigner outputs/<run> --table table0 \\
        --n-res 3 --n-samples 200 --top-k 5

    # one-sided interface: network must include target residue B5, chain B is not
    # designable, and chain B's real sequence is grafted back afterwards
    sapia run hbdesigner outputs/<run> --table table0 \\
        -i rfdiffusion3_path \\
        --n-res 3 --n-samples 200 --anchor-res B5 --omit-chains B
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from prosapia.core import CommonArgs, ManifestCtx
from prosapia.core.executors import volume_path
from prosapia.utils import ensure_pdb, expand_chain_spec, resolve_template

# Mirrors RESOURCES["cpu"] in modal_image.py. Used only when neither --n-workers
# nor -c/--cpus-per-task is given, so the packing pool matches the container.
DEFAULT_CPUS = 16

# --graft-chains sentinel: graft exactly the chains --omit-chains made poly-Gly.
GRAFT_AUTO = "auto"
GRAFT_NONE = "none"

DESIGN_MODELS = ("design_002", "design_020")


class HBDesignerArgs(CommonArgs):
    design_model: str
    cpu: bool
    n_workers: int | None
    n_samples: int
    top_k: int
    n_res: int
    t_range: list[float]
    min_burial: float
    guide_res: str
    guide_radius: float
    guide_seq: str
    max_buns: int
    max_buphs: int
    min_sat: float
    max_hb_energy: float
    max_hb_score: float
    min_core_res: int
    anchor_res: str
    symm_chains: str
    symm_file: str
    sel_chains: str
    omit_chains: str
    omit_aa: str
    seed: int | None
    graft_chains: str
    set: list[str]


def add_run_hbdesigner_args(parser: ArgumentParser) -> None:
    # --- model / sampling -------------------------------------------------
    parser.add_argument(
        "--design-model",
        choices=DESIGN_MODELS,
        default="design_020",
        help="Which design checkpoint to sample from. 'design_020' (default, "
        "moderate noise) is the general-purpose model; 'design_002' (low noise) "
        "is more conservative. Both ship in the image.",
    )
    parser.add_argument(
        "--n-res",
        type=int,
        choices=range(2, 7),
        default=2,
        metavar="{2..6}",
        help="Size of the designed network, in residues (2-6, default 2). This is "
        "the ONLY thing hbdesigner designs -- every other position comes back as "
        "glycine. Bigger networks pack far less often: upstream's advice is "
        "n_res=2 -> --n-samples 100, 3 -> 200, 4/5 -> 500, 6 -> 1000.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Networks sampled before packing/scoring (default 100). Only a "
        "fraction survive packing -- upstream reports ~10 of 200 as typical -- so "
        "this is the main knob when a design returns nothing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many ranked designs to KEEP per input backbone (default 5). You "
        "may get FEWER: upstream takes min(top_k, n_surviving), so the child table "
        "is not n_parents * top_k rows. Never assume a fixed fan-out.",
    )
    parser.add_argument(
        "--t-range",
        type=float,
        nargs=2,
        default=[0.1, 1.0],
        metavar=("LO", "HI"),
        help="Sampling temperature range (default 0.1 1.0). Lower = more "
        "conservative, higher = more diverse. Passed through as --T_range.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Off by default. Note upstream's own caveat: with "
        "parallel packing and Rosetta scoring, runs are NOT bit-reproducible even "
        "with a seed.",
    )

    # --- where the network may go ----------------------------------------
    parser.add_argument(
        "--min-burial",
        type=float,
        default=0.0,
        help="Minimum burial (Rosetta sidechain-neighbour count) for a position to "
        "be designable (default 0.0 = no constraint). Raise it to force the "
        "network into the core.",
    )
    parser.add_argument(
        "--min-core-res",
        type=int,
        default=0,
        help="Minimum number of CORE residues a network must contain (default 0).",
    )
    parser.add_argument(
        "--guide-res",
        default="",
        help="Residues whose Cb centroid a virtual guide atom is placed at, so the "
        "network forms near them. PDB chain+resnum, comma-joined: 'A3,A26'. "
        "{expr} islands are resolved per design up the lineage (e.g. "
        "'A{motif_end}'). NOTE: PDB numbering, NOT the 1..L-per-chain indexing "
        "proteinmpnn's --fixed-positions uses.",
    )
    parser.add_argument(
        "--guide-radius",
        type=float,
        default=1e6,
        help="Hard cap (Angstrom) on the Cb distance from the guide atom for a "
        "designable position. Default 1e6 = off. Only meaningful with --guide-res.",
    )
    parser.add_argument(
        "--anchor-res",
        default="",
        help="Residues that EVERY returned network must contain, PDB chain+resnum, "
        "comma-joined: 'B5'. {expr} islands resolved per design. This is what makes "
        "one-sided interface design work: anchor on the target residue you want "
        "hydrogen-bonded. Must be fewer than --n-res, and anchors must be polar "
        "(upstream rejects hydrophobic anchors). Anchors keep their input identity, "
        "so they appear in the reported network alongside the designed positions.",
    )

    # --- what the network may be -----------------------------------------
    parser.add_argument(
        "--guide-seq",
        default="",
        help="Per-position amino-acid conditioning, comma-joined, one entry per "
        "--n-res position: 'S,N,T' (exactly those), 'X,T,X' (at least one THR), "
        "'S,N|Q,T' (S, then N or Q, then T). Default = all X (unconstrained). "
        "Upstream's advice: SER/THR networks pack far more often, so 'S,X,X' is a "
        "cheap success-rate win when the identities do not matter.",
    )
    parser.add_argument(
        "--omit-aa",
        default="",
        help="Amino acids the network may NOT use, comma-joined: 'R,K'. Passed "
        "through as --omit_AA. Must not contradict --guide-seq.",
    )

    # --- chains -----------------------------------------------------------
    parser.add_argument(
        "--omit-chains",
        default="",
        help="Chains HBDesigner SEES but may not take network residues from "
        "(except via --anchor-res). Chain mini-language: ':' is an inclusive "
        "letter range, ',' separates ('B', 'A:C', 'A,C'). This is the one-sided "
        "interface-design flag. The omitted chain still comes back POLY-GLYCINE -- "
        "see --graft-chains, which by default repairs exactly these chains.",
    )
    parser.add_argument(
        "--sel-chains",
        default="",
        help="Chains to run HBDesigner on; every other chain is REMOVED from the "
        "input before design and concatenated back AFTER it, keeping its own "
        "sequence. Chain mini-language. Different from --omit-chains: here the "
        "model never sees the other chains at all, so it cannot avoid clashing "
        "with them. Note the concatenation appends the unused chains at the end, "
        "so the output chain ORDER may differ from the input's.",
    )
    parser.add_argument(
        "--symm-chains",
        default="",
        help="Symmetrize the designed network across chains after design: 'A,B' "
        "ties A with B; 'A,B;C,D' ties two pairs independently. Off by default. "
        "Experimental upstream, and it can drop every network (see the zero-output "
        "trap in the skill).",
    )
    parser.add_argument(
        "--symm-file",
        default="",
        help="Rosetta .symm file for STRICT symmetry (made with "
        "make_symmdef_file.pl). Requires --symm-chains. The path must be readable "
        "from the task container -- put it inside the run_dir on the Volume.",
    )

    # --- acceptance filters (the only judgement in the tool) ---------------
    parser.add_argument(
        "--max-buns",
        type=int,
        default=0,
        help="Max buried unsatisfied HEAVY atoms allowed in a network (default 0, "
        "i.e. none). Passed as --max_BUNs. Raising it is more permissive.",
    )
    parser.add_argument(
        "--max-buphs",
        type=int,
        default=5,
        help="Max buried unsatisfied POLAR HYDROGENS allowed (default 5). Passed "
        "as --max_BUPHs.",
    )
    parser.add_argument(
        "--min-sat",
        type=float,
        default=0.5,
        help="Minimum network saturation, in [0, 2] (default 0.5). Higher is "
        "STRICTER -- the opposite direction from the two --max-* filters.",
    )
    parser.add_argument(
        "--max-hb-energy",
        type=float,
        default=0.0,
        help="Rosetta score:hb_max_energy -- the per-hydrogen-bond energy ceiling "
        "used while scoring (default 0.0).",
    )
    parser.add_argument(
        "--max-hb-score",
        type=float,
        default=0.0,
        help="Max HB_Score a returned network may have (default 0.0). More "
        "negative is stricter.",
    )

    # --- resources / post-processing / escape hatch ------------------------
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker processes for CPU packing (DataLoader workers, and the pool "
        "used for lazy symmetrization). Default: match the task's CPU count "
        f"(-c/--cpus-per-task, else {DEFAULT_CPUS}). MUST be >= 1: upstream builds "
        "its DataLoader with persistent_workers=True, which raises on "
        "num_workers=0, so '--n-workers 0' is not a way to turn parallelism off.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU (loads the *_cpu.yaml configs) and request no "
        "GPU. Much slower; for debugging or a GPU-less backend.",
    )
    parser.add_argument(
        "--graft-chains",
        default=GRAFT_AUTO,
        help="After design, restore the original sequence AND sidechains of these "
        f"chains from the tool's own input PDB (upstream's graft_seq.py). "
        f"'{GRAFT_AUTO}' (default) grafts exactly the chains named by "
        f"--omit-chains, and does nothing otherwise; '{GRAFT_NONE}' never grafts; "
        "or give an explicit chain list. WITHOUT this, a one-sided interface "
        "design hands the next tool a target chain that is poly-glycine. Grafting "
        "only overwrites positions that came back as glycine, so designed and "
        "anchor residues are preserved. Refused together with --sel-chains, whose "
        "chain reordering would make the graft copy residues onto the wrong "
        "positions.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="TOKEN",
        help="Forward a raw run_hbdesigner flag verbatim, repeatable: "
        "--set --design_model_ckpt --set /runs/<path>.pt. The escape hatch for "
        "upstream flags this tool does not name (the two custom-checkpoint flags). "
        "Tokens may not contain whitespace.",
    )


def _chain_list(spec: str, flag: str) -> str:
    """Chain mini-language -> the comma-joined list HBDesigner wants ('A,C')."""
    spec = spec.strip()
    if not spec:
        return ""
    try:
        chains = expand_chain_spec(spec)
    except ValueError as e:
        raise ValueError(f"{flag}: {e}") from None
    if len(set(chains)) != len(chains):
        raise ValueError(f"{flag} names a chain twice: {chains}")
    return ",".join(chains)


def _symm_chain_groups(spec: str) -> str:
    """'A,B;C,D' -> each ';' group expanded through the chain mini-language."""
    spec = spec.strip()
    if not spec:
        return ""
    groups = [g for g in (part.strip() for part in spec.split(";")) if g]
    if not groups:
        raise ValueError("--symm-chains is empty after parsing")
    return ";".join(_chain_list(group, "--symm-chains") for group in groups)


def _resolve_graft_chains(args: HBDesignerArgs, omit_chains: str) -> str:
    """Which chains the task script grafts back, as a comma-joined list ('' = none)."""
    spec = args.graft_chains.strip()
    if spec.lower() == GRAFT_NONE:
        return ""
    chains = omit_chains if spec.lower() == GRAFT_AUTO else _chain_list(spec, "--graft-chains")
    if chains and args.sel_chains.strip():
        raise ValueError(
            "--graft-chains cannot be combined with --sel-chains: --sel-chains "
            "appends the unused chains at the END of the output, so the output and "
            "the input no longer share a residue order, and upstream's graft_seq.py "
            "copies BY POSITION -- it would write the wrong residues. --sel-chains "
            "already returns the unused chains with their own sequence, so no graft "
            "is needed: pass --graft-chains none."
        )
    return chains


def _flag_tokens(args: HBDesignerArgs, guide_res: str, anchor_res: str) -> list[str]:
    """The run_hbdesigner argv (minus --pdb/--out_dir) for one design.

    Every value is passed explicitly, defaults included, so the submitted command
    is fully readable in the manifest and an upstream default change cannot move
    a run's meaning underneath it.
    """
    tokens: list[str] = [
        "--design_model", args.design_model,
        "--n_res", str(args.n_res),
        "--n_samples", str(args.n_samples),
        "--top_k", str(args.top_k),
        "--T_range", str(args.t_range[0]), str(args.t_range[1]),
        "--min_burial", str(args.min_burial),
        "--min_core_res", str(args.min_core_res),
        "--guide_radius", str(args.guide_radius),
        "--max_BUNs", str(args.max_buns),
        "--max_BUPHs", str(args.max_buphs),
        "--min_sat", str(args.min_sat),
        "--max_hb_energy", str(args.max_hb_energy),
        "--max_hb_score", str(args.max_hb_score),
    ]
    optional: list[tuple[str, str]] = [
        ("--guide_res", guide_res),
        ("--guide_seq", args.guide_seq.strip()),
        ("--anchor_res", anchor_res),
        ("--omit_AA", args.omit_aa.strip()),
        ("--sel_chains", _chain_list(args.sel_chains, "--sel-chains")),
        ("--omit_chains", _chain_list(args.omit_chains, "--omit-chains")),
        ("--symm_chains", _symm_chain_groups(args.symm_chains)),
    ]
    for flag, value in optional:
        if value:
            tokens += [flag, value]
    if args.symm_file.strip():
        tokens += ["--symm_file", str(volume_path(Path(args.symm_file.strip())))]
    if args.seed is not None:
        tokens += ["--seed", str(args.seed)]
    if args.cpu:
        tokens.append("--cpu")
    tokens += [str(tok) for tok in args.set]
    return tokens


def build_hbdesigner_manifest(ctx: ManifestCtx[HBDesignerArgs]) -> list[tuple[str, ...]]:
    args = ctx.args

    if args.cpu:
        args.gpus_per_task = 0

    # Fail loudly on a column that matches nothing, instead of repeating the known
    # proteinmpnn bug where a wrong default silently submits zero tasks.
    column = args.input_column
    if column not in ctx.df.columns:
        available = ", ".join(str(c) for c in ctx.df.columns if str(c).endswith("_path"))
        raise ValueError(
            f"hbdesigner: table {args.table!r} has no column {column!r}. Pass "
            f"-i/--input-column with the backbone column to design onto. Structure "
            f"columns available: {available or '(none)'}."
        )

    if args.n_workers is not None and args.n_workers < 1:
        raise ValueError(
            f"--n-workers must be at least 1 (got {args.n_workers}): upstream builds "
            f"its packing DataLoader with persistent_workers=True, which raises on "
            f"num_workers=0. Use --n-workers 1 for 'no parallelism'."
        )
    n_workers = args.n_workers or args.cpus_per_task or DEFAULT_CPUS

    if args.symm_file.strip() and not args.symm_chains.strip():
        raise ValueError(
            "--symm-file is only used for strict symmetry and needs --symm-chains "
            "to say which chains it ties."
        )
    if args.top_k < 1:
        raise ValueError(f"--top-k must be at least 1 (got {args.top_k})")

    omit_chains = _chain_list(args.omit_chains, "--omit-chains")
    graft_chains = _resolve_graft_chains(args, omit_chains)

    manifest_rows: list[tuple[str, ...]] = []
    for name in ctx.ready.index:
        name = cast(str, name)
        src = Path(str(ctx.ready.at[name, column]))
        if not src.exists():
            print(f"{name}: MISSING {src} (skipping)")
            continue
        # HBDesigner reads PDB only (Protein.from_pdb_file).
        input_pdb = ensure_pdb(src, args.run_dir)

        # Per-design {expr} resolution happens HERE, so a bad column name kills the
        # submit instead of 100 tasks. Both flags take HBDesigner's own
        # chain+resnum syntax verbatim; only the {...} islands are ours.
        guide_res = resolve_template(args.guide_res.strip(), ctx.lookup, name)
        anchor_res = resolve_template(args.anchor_res.strip(), ctx.lookup, name)

        tokens = _flag_tokens(args, guide_res, anchor_res) + ["--n_workers", str(n_workers)]
        bad = [tok for tok in tokens if any(ch.isspace() for ch in tok)]
        if bad:
            raise ValueError(
                f"{name}: flag token(s) {bad} contain whitespace; the task script "
                f"word-splits the manifest's argument field, so they cannot be "
                f"passed. Remove the spaces (e.g. --guide-seq 'S,N|Q,T', not 'S, N')."
            )

        manifest_rows.append(
            (
                name,
                str(volume_path(input_pdb)),
                # May be empty, so never last.
                graft_chains,
                # Always non-empty: the manifest line never ends on an empty field.
                " ".join(tokens),
            )
        )

    return manifest_rows
