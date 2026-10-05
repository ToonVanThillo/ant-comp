"""Local smoke test for the manifest builder and the collector (no model, no cluster).

Exercises the guardrails and the create-contract against the real example structure
shipped with Proton-PottsMPNN. Run with:

    PROSAPIA_TOOLS_DIR=tools python tools/protonpottsmpnn/_smoketest.py <pdl1_seed_binder.pdb>
"""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prosapia.core import ManifestCtx  # noqa: E402

from protonpottsmpnn.collect_protonpottsmpnn import collect_protonpottsmpnn  # noqa: E402
from protonpottsmpnn.run_protonpottsmpnn import (  # noqa: E402
    build_protonpottsmpnn_manifest,
)

PDB = Path(sys.argv[1]).resolve()
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(("  ok   " if condition else "  FAIL ") + label + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def expect_raise(label: str, fn, fragment: str) -> None:
    try:
        fn()
    except Exception as exc:
        ok = fragment.lower() in str(exc).lower()
        check(label, ok, f"raised {type(exc).__name__}: {str(exc)[:90]}")
        return
    check(label, False, "did not raise")


def make_args(run_dir: Path, **overrides):
    args = Namespace(
        run_dir=run_dir,
        input_column="pdb_path",
        force=False,
        gpus_per_task=1,
        binder_chain="A",
        num_designs=4,
        samples_per_design=1,
        lambda_min=0.0,
        lambda_max=1.0,
        center_types="HIS-P,ASP-P,GLU-P",
        explicit_centers="",
        placement_region="all",
        placement_by="scan_potts",
        neighbour_k=16,
        max_mutations=20,
        block_size=3,
        temperature=0.05,
        seed=0,
        seed_column="",
        checkpoint="",
        n_jobs=0,
        set=[],
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def make_ctx(run_dir: Path, out_dir: Path, df: pd.DataFrame, **overrides) -> ManifestCtx:
    lookup_table = {("design0", "motif_end"): 57}
    return ManifestCtx(
        df=df,
        args=make_args(run_dir, **overrides),
        out_dir=out_dir,
        lookup=lambda name, column: lookup_table.get((name, column)),
        write_meta=lambda **_: None,
    )


with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp)
    out_dir = run_dir / "table0" / "protonpottsmpnn"
    out_dir.mkdir(parents=True)
    df = pd.DataFrame({"pdb_path": [str(PDB)]}, index=["design0"])

    print("\n-- manifest builder --")
    ctx = make_ctx(run_dir, out_dir, df)
    rows = build_protonpottsmpnn_manifest(ctx)
    check("one manifest row per design", len(rows) == 1, f"{len(rows)} row(s)")
    check("row is (name, config_path)", len(rows[0]) == 2 and rows[0][0] == "design0")
    check("gpus_per_task forced to 0 for CPU work", ctx.args.gpus_per_task == 0)

    import json

    config = json.loads(Path(rows[0][1]).read_text())
    check(
        "lambda ladder spans min..max",
        config["lambdas"] == [0.0, 0.3333, 0.6667, 1.0],
        str(config["lambdas"]),
    )
    check("binder chain recorded", config["binder_chain"] == "A")
    check("v6 dep_map wired", config["dep_map"]["HIS-P"] == ["HIS-S"])
    check("ambiguous microstates forbidden", "HIS-A" in config["forbidden_tokens"])

    print("\n-- the {expr} mini-language on --explicit-centers --")
    ctx = make_ctx(run_dir, out_dir, df, explicit_centers="{motif_end}:HIS-P,78:ASP-P")
    config = json.loads(Path(build_protonpottsmpnn_manifest(ctx)[0][1]).read_text())
    check(
        "column expression resolved up the lineage",
        config["explicit_centers"] == [
            {"res_id": 57, "protonation_type": "HIS-P"},
            {"res_id": 78, "protonation_type": "ASP-P"},
        ],
        str(config["explicit_centers"]),
    )
    check(
        "--explicit-centers overrides --center-types",
        config["center_types"] == [],
    )

    print("\n-- guardrails (all must fail BEFORE submission) --")
    expect_raise(
        "unknown binder chain is rejected",
        lambda: build_protonpottsmpnn_manifest(make_ctx(run_dir, out_dir, df, binder_chain="Z")),
        "not a polymer chain",
    )
    expect_raise(
        "deprotonated state cannot be placed as a centre",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df, center_types="HIS-S")
        ),
        "not a protonated state",
    )
    expect_raise(
        "malformed centre is rejected",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df, explicit_centers="45-HIS-P")
        ),
        "malformed centre",
    )
    expect_raise(
        "unknown placement region is rejected",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df, placement_region="epitope")
        ),
        "is not a region",
    )
    expect_raise(
        "lambda outside [0,1] is rejected",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df, lambda_max=1.5)
        ),
        "must be in [0, 1]",
    )
    expect_raise(
        "malformed --set is rejected",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df, set=["block_max_rounds 20"])
        ),
        "malformed token",
    )
    expect_raise(
        "empty --seed-column is an error, not a silent fallback",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df, seed_column="proteinmpnn_sequence")
        ),
        "is empty for design",
    )

    print("\n-- the single-chain premise --")
    single = run_dir / "single_chain.pdb"
    single.write_text(
        "\n".join(
            line
            for line in PDB.read_text().splitlines()
            if not (line.startswith(("ATOM", "HETATM")) and line[21] != "A")
        )
        + "\n"
    )
    df_single = pd.DataFrame({"pdb_path": [str(single)]}, index=["design0"])
    expect_raise(
        "interface region on a 1-chain input is refused",
        lambda: build_protonpottsmpnn_manifest(
            make_ctx(run_dir, out_dir, df_single, placement_region="interface")
        ),
        "only chain",
    )
    rows = build_protonpottsmpnn_manifest(
        make_ctx(run_dir, out_dir, df_single, placement_region="core")
    )
    check("a 1-chain input is still allowed for core/surface/all", len(rows) == 1)

    print("\n-- collector --")
    design_dir = out_dir / "design0"
    design_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "design_id\tsequence\textended_tokens\tpotts_energy\tselective_energy\tglobal_dh\t"
        "combined_lambda\tsample\tcenters\tcenter_seqpos\tn_centers\tcenters_verified\t"
        "n_designable\tbinder_chain\tbinder_len\tresnum_offset\tn_mut\tseq_rec\t"
        "seed_source\tstatus\tpareto"
    )
    (design_dir / "designs.tsv").write_text(
        header
        + "\n"
        + "d0\tMKT\tMET LYS HIS-P\t-120.5\t-2.5\t-1.0\t0.0\t0\t57:HIS-P\t3\t1\t1.0\t"
        "16\tA\t3\t0\t5\t0.94\tnative\tOK\tTrue\n"
        + "d1\tMRT\tMET ARG HIS-P\t-118.0\t-3.5\t-1.2\t1.0\t0\t57:HIS-P\t3\t1\t0.0\t"
        "16\tA\t3\t0\t7\t0.91\tnative\tOK\tTrue\n"
    )

    class FakeCollectCtx:
        out_dir = out_dir

    class FakeDesign:
        name = "design0"

    collected = list(collect_protonpottsmpnn(FakeCollectCtx())(FakeDesign()))
    check("one child row per design", len(collected) == 2, f"{len(collected)} row(s)")
    check("child rows keyed <parent>_p<i>", [c.name for c in collected] == ["design0_p0", "design0_p1"])
    check("every child carries its parent", all(c.parent == "design0" for c in collected))
    check("numeric columns are numbers", collected[0].data["potts_energy"] == -120.5)
    check("pareto parsed as bool", collected[0].data["pareto"] is True)
    check(
        "unverified centre is visible as a number, not a blank",
        collected[1].data["centers_verified"] == 0.0,
    )
    check("status OK propagated", collected[0].status == "OK")

    print("\n-- collector on a worker failure --")
    (design_dir / "designs.tsv").write_text("status\nerror: RuntimeError: HBPLUS_PATH is not set\n")
    failed = list(collect_protonpottsmpnn(FakeCollectCtx())(FakeDesign()))
    check("a failure still mints one row", len(failed) == 1)
    check(
        "the error is the row's status, not a blank",
        failed[0].status.startswith("error: RuntimeError"),
        failed[0].status,
    )

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
