#!/usr/bin/env python3
"""
Self-test for epitope: synthetic complexes with a hand-countable answer.

The geometry is trivial on purpose -- a target chain of residues on a line and a
binder placed a known distance from a known stretch of it -- so every column has an
answer that can be written down before running anything. What the test is really
for is the ERROR CONTRACTS, which are the reason the tool exists:

    * a hotspot residue number the target chain does not have is an ERROR, with
      every column NA -- never a recall of 0.0 (the renumbering trap);
    * a hotspot naming a chain that is not a target chain is an error;
    * an absent design chain and an absent target chain are errors;
    * a binder that bound elsewhere is status OK with recall 0.0, hits 'none' and
      a real min_dist_hotspot -- and the collector keeps that 0.0 as 0.0 while
      turning every error into NA. NA is never 0.
    * hotspot_resnames reports the residue names actually found, in the order
      listed -- the trust column that makes a renumbered-but-valid structure (the
      nastier case, where the numbers exist and mean something else) visible.
    * hotspot_contact_frac is NA at a zero interface (the share is 0/0) while
      hotspot_n_contacts is a real 0 -- the two halves of that pair are not the
      same kind of thing, and the tool never conflates them.

The weight columns (added 2026-10-02) are hand-countable by construction: every
residue is a 4-atom cluster 0.5 A across, binder residues sit 4.0 A above the
target residues they cover, and consecutive target residues are 4.0 A apart. So a
binder residue reaches only the residue directly beneath it (16 atom pairs, all
inside the 5 A cutoff) and not its neighbours (5.66 A, outside). Every count in
those cases is a multiple of 16 that can be written down before running anything.

Plus the submit-side parsing: ranges, duplicates, malformed tokens, empty lists.

Run:  uv run python tools/epitope/test_epitope.py
"""

import importlib.util
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("epitope_worker")
run_epitope = _load("run_epitope")
collect_epitope = _load("collect_epitope")

from prosapia.core import CollectCtx, DesignCtx, ManifestCtx  # noqa: E402

FAILURES: list[str] = []

# Distinct residue names, so hotspot_resnames is checkable position by position.
CYCLE = ["ALA", "LYS", "GLU", "LEU", "SER", "ARG", "TRP"]
SPACING = 4.0  # A between consecutive target CAs, along x


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def raises(fn, fragment: str, message: str) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        ok = fragment.lower() in str(e).lower()
        check(ok, f"{message} (raised: {str(e)[:100]})")
        return
    check(False, f"{message} (raised nothing)")


# ---------------------------------------------------------------- structures


def write_complex(
    path: Path,
    target_start: int = 90,
    n_target: int = 21,
    binder_over: tuple[int, int] | None = (96, 101),
    gap: float = 4.0,
    binder_chain: str = "B",
    target_chain: str = "A",
) -> Path:
    """A target chain laid out along x (one 4-atom residue every SPACING A) and a
    binder chain of 4-atom residues parked `gap` A above the target residues
    [binder_over[0], binder_over[1]] inclusive. Every distance is therefore known:
    at a 5 A cutoff the binder contacts exactly that span and nothing else.
    """
    st = gemmi.Structure()
    st.name = path.stem
    model = gemmi.Model("1")

    target = gemmi.Chain(target_chain)
    for i in range(n_target):
        num = target_start + i
        res = gemmi.Residue()
        res.name = CYCLE[i % len(CYCLE)]
        res.seqid = gemmi.SeqId(num, " ")
        base = np.array([i * SPACING, 0.0, 0.0])
        # Four heavy atoms in a tight 0.5 A cluster: the residue is effectively a
        # point, so min_dist is exactly the layout distance.
        for j, atom_name in enumerate(("N", "CA", "C", "O")):
            atom = gemmi.Atom()
            atom.name = atom_name
            atom.element = gemmi.Element(atom_name[0])
            pos = base + np.array([0.0, 0.0, 0.1 * j])
            atom.pos = gemmi.Position(*(float(v) for v in pos))
            atom.occ = 1.0
            res.add_atom(atom)
        target.add_residue(res)
    model.add_chain(target)

    if binder_over is not None:
        lo, hi = binder_over
        binder = gemmi.Chain(binder_chain)
        for k, num in enumerate(range(lo, hi + 1), start=1):
            res = gemmi.Residue()
            res.name = "GLY"
            res.seqid = gemmi.SeqId(k, " ")
            x = (num - target_start) * SPACING
            for j, atom_name in enumerate(("N", "CA", "C", "O")):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(float(x), float(gap), float(0.1 * j))
                atom.occ = 1.0
                res.add_atom(atom)
            binder.add_residue(res)
        model.add_chain(binder)

    st.add_model(model)
    st.setup_entities()
    path.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(path))
    return path


def run_worker(
    tmp: Path,
    pdb: Path,
    name: str,
    hotspots: str,
    design_chains: str = "B",
    target_chains: str = "A",
    contact_cutoff: float = 5.0,
    sequence: str = "",
    seq_source: str = "structure",
    task_line: str | None = None,
) -> pd.Series:
    """One design through the worker; returns its result row.

    ``task_line`` overrides the whole sub-manifest line, which is how the
    3-field (pre-2026-09-30) layout is tested."""
    out_dir = tmp / f"out_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    task = tmp / f"task_{name}.tsv"
    task.write_text(
        task_line
        if task_line is not None
        else f"{name}\t{pdb}\t{hotspots}\t{sequence}\n"
    )
    sys.argv = [
        "epitope_worker.py",
        "--task-file",
        str(task),
        "--design-chains",
        design_chains,
        "--target-chains",
        target_chains,
        "--contact-cutoff",
        str(contact_cutoff),
        "--seq-source",
        seq_source,
        "--input-column",
        "boltz_path",
        "--out-dir",
        str(out_dir),
    ]
    worker.main()
    return pd.read_csv(out_dir / f"{name}.tsv", sep="\t", dtype=str).iloc[0]


def collect_one(tmp: Path, name: str, result_row: pd.Series) -> dict:
    """The same design through the collector, as the driver would call it."""
    out_dir = tmp / f"out_{name}"
    ctx = CollectCtx(
        df=pd.DataFrame(index=pd.Index([name])),
        args=Namespace(
            run_dir=tmp, table="table1", dir_label="", force=False
        ),
        table_name="table1",
        out_dir=out_dir,
        status_col="epitope_status",
        path_col="epitope_path",
        parent_table=None,
        parent_df=pd.DataFrame(),
        lookup=lambda n, column: None,
        creates_table=False,
        default_input_column="boltz_path",
    )
    one = collect_epitope.collect_epitope(ctx)
    emitted = list(one(DesignCtx(name, out_dir, ctx.lookup)))
    assert len(emitted) == 1, f"expected one Collected, got {len(emitted)}"
    c = emitted[0]
    return {"data": dict(c.data), "path": c.path, "status": c.status}


def build_manifest(tmp: Path, structures: dict[str, Path], **overrides):
    """The manifest builder, as the driver would call it. Returns
    (rows, sub-manifest lines)."""
    out_dir = tmp / "manifest_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {"boltz_path": [str(p) for p in structures.values()]},
        index=pd.Index(list(structures), name="name"),
    )
    for key, value in overrides.pop("columns", {}).items():
        df[key] = value
    args = Namespace(
        run_dir=tmp,
        table="table1",
        input_column=overrides.pop("input_column", "boltz_path"),
        force=False,
        gpus_per_task=1,
        design_chains=overrides.pop("design_chains", "B"),
        target_chains=overrides.pop("target_chains", "A"),
        hotspots=overrides.pop("hotspots", "A96,A99"),
        contact_cutoff=overrides.pop("contact_cutoff", 5.0),
        designs_per_task=overrides.pop("designs_per_task", 20),
        sequence_column=overrides.pop("sequence_column", None),
    )
    lookup_values = overrides.pop("lookup", {})
    ctx = ManifestCtx(
        df=df,
        args=args,
        out_dir=out_dir,
        lookup=lambda name, column: lookup_values.get(column),
        write_meta=lambda **_: None,
    )
    rows = run_epitope.build_epitope_manifest(ctx)
    sub = []
    for row in rows:
        sub.extend(
            line.split("\t")
            for line in Path(row[0]).read_text().splitlines()
            if line.strip()
        )
    return rows, sub, args


def num(row: pd.Series, col: str) -> float | None:
    value = row.get(col)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def text(row: pd.Series, col: str) -> str | None:
    value = row.get(col)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return str(value)


# ---------------------------------------------------------------- the cases


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        def case(title, fn):
            print(f"\n== {title}")
            fn()

        # --- submit-side parsing ------------------------------------------
        def t_parse():
            p = run_epitope.parse_hotspots
            check(
                p("A96,A99,A101,A155") == ["A96", "A99", "A101", "A155"],
                "plain list parses in order",
            )
            check(
                p("A96-99") == ["A96", "A97", "A98", "A99"],
                "a range expands inclusively",
            )
            check(
                p("A96, A99-A101 ,B5") == ["A96", "A99", "A100", "A101", "B5"],
                "ranges may repeat the chain, whitespace is ignored",
            )
            check(p("A96A") == ["A96A"], "an insertion code survives")
            raises(lambda: p("A99-96"), "ends before", "a backwards range raises")
            raises(lambda: p("A96-B99"), "two chains", "a cross-chain range raises")
            raises(lambda: p("96"), "cannot parse", "a chainless residue raises")
            raises(lambda: p("A96,A96"), "more than once", "a duplicate raises")
            raises(lambda: p(" , "), "empty", "an empty list raises")

        case("--hotspots parsing (the denominator of recall)", t_parse)

        def t_manifest():
            structures = {"d0": write_complex(tmp / "d0.pdb")}
            rows, sub, args = build_manifest(tmp, structures)
            check(args.gpus_per_task == 0, "the builder forces gpus_per_task = 0")
            check(
                len(rows) == 1 and len(rows[0]) == 6,
                "one task row per batch, with 6 fields (task_file, design, target, "
                "cutoff, seq_source, input_column) -- the two new ones APPENDED, "
                "so epitope.sh's cut -f1..4 still reads what it always did",
            )
            check(
                rows[0][1] == "B" and rows[0][2] == "A" and rows[0][3] == "5.0",
                "the task row still carries the chains and the cutoff, unchanged",
            )
            check(
                rows[0][4] == "structure" and rows[0][5] == "boltz_path",
                f"and echoes the identity source and the input column "
                f"(got {rows[0][4]!r}, {rows[0][5]!r})",
            )
            check(
                sub == [["d0", str(structures["d0"]), "A96,A99", ""]],
                f"the sub-manifest carries the EXPANDED hotspot list per design, "
                f"plus an EMPTY trailing sequence field when --sequence-column is "
                f"not used (got {sub})",
            )
            # Batching.
            many = {f"d{i}": write_complex(tmp / f"m{i}.pdb") for i in range(5)}
            rows, sub, _ = build_manifest(tmp, many, designs_per_task=2)
            check(
                len(rows) == 3 and len(sub) == 5,
                f"5 designs at 2 per task -> 3 tasks (got {len(rows)})",
            )
            # {expr}, resolved per design at build time.
            rows, sub, _ = build_manifest(
                tmp,
                structures,
                hotspots="A{ep_start}-{ep_end}",
                lookup={"ep_start": 96, "ep_end": 98},
            )
            check(
                sub[0][2] == "A96,A97,A98",
                f"{{expr}} is resolved and expanded at submit time (got {sub[0][2]})",
            )
            # The build-time refusals.
            raises(
                lambda: build_manifest(tmp, structures, hotspots="B96"),
                "not in --target-chains",
                "a hotspot outside the target chains kills the SUBMIT, before any "
                "task runs",
            )
            raises(
                lambda: build_manifest(tmp, structures, hotspots="A96,A96"),
                "more than once",
                "a duplicated hotspot kills the submit",
            )
            raises(
                lambda: build_manifest(tmp, structures, design_chains="A"),
                "both --design-chains and --target-chains",
                "a chain on both sides kills the submit",
            )
            raises(
                lambda: build_manifest(tmp, structures, input_column="nope_path"),
                "resolved no structure at all",
                "an input column that resolves NOWHERE -- not on the row, not up the "
                "lineage -- kills the submit (never a silent empty run)",
            )
            raises(
                lambda: build_manifest(tmp, structures, contact_cutoff=0.0),
                "must be > 0",
                "a non-positive cutoff kills the submit",
            )

        case("the manifest builder (failing at the cheap place)", t_manifest)

        # --- the happy path -----------------------------------------------
        pdb = write_complex(tmp / "hit.pdb")

        def t_full_recall():
            row = run_worker(tmp, pdb, "hit", "A96,A99,A101")
            check(row["status"] == "OK", "status OK")
            check(num(row, "hotspot_recall") == 1.0, "recall 1.0 when all three are hit")
            check(text(row, "hotspot_hits") == "A96,A99,A101", "hits lists all three")
            check(num(row, "n_hotspots") == 3, "n_hotspots is the list length")
            # Layout: binder sits over residues 96..101, 4.0 A above.
            check(
                num(row, "n_iface_target_res") == 6,
                f"6 target residues contacted (got {text(row, 'n_iface_target_res')})",
            )
            check(
                text(row, "iface_target_resnums")
                == "A96,A97,A98,A99,A100,A101",
                f"the epitope actually used is listed "
                f"(got {text(row, 'iface_target_resnums')})",
            )
            check(
                abs((num(row, "min_dist_hotspot") or 0) - 4.0) < 0.2,
                f"min_dist_hotspot is the real 4 A gap "
                f"(got {text(row, 'min_dist_hotspot')})",
            )
            check((num(row, "n_contacts") or 0) > 0, "n_contacts is non-zero")
            check(
                text(row, "design_chains") == "B"
                and text(row, "target_chains") == "A"
                and num(row, "contact_cutoff") == 5.0,
                "the echo columns repeat what was asked for",
            )
            # Residue 90 is CYCLE[0]=ALA, so 96 -> CYCLE[6]=TRP, 99 -> CYCLE[2]=GLU,
            # 101 -> CYCLE[4]=SER.
            check(
                text(row, "hotspot_resnames") == "A96=TRP,A99=GLU,A101=SER",
                f"hotspot_resnames names the residues actually found, in order "
                f"(got {text(row, 'hotspot_resnames')})",
            )
            per_res = pd.read_csv(
                tmp / "out_hit" / "hit_per_residue.tsv", sep="\t"
            )
            check(
                len(per_res) == 21,
                f"the per-residue file covers EVERY target residue, contacted or "
                f"not (got {len(per_res)})",
            )
            check(
                int(per_res["is_hotspot"].sum()) == 3,
                "the per-residue file flags the hotspots",
            )

        case("a binder on the chosen epitope", t_full_recall)

        # --- partial recall, and the graded miss ---------------------------
        def t_partial():
            row = run_worker(tmp, pdb, "partial", "A96,A99,A108,A110")
            check(num(row, "hotspot_recall") == 0.5, "recall 0.5 when 2 of 4 are hit")
            check(text(row, "hotspot_hits") == "A96,A99", "hits lists only the two")

        case("partial recall", t_partial)

        def t_bound_elsewhere():
            row = run_worker(tmp, pdb, "elsewhere", "A108,A110")
            check(row["status"] == "OK", "status is OK -- this is a real measurement")
            check(
                num(row, "hotspot_recall") == 0.0,
                "recall is 0.0, not NA: the binder folded and bound somewhere else",
            )
            check(text(row, "hotspot_hits") == "none", "hits is 'none', not blank")
            d = num(row, "min_dist_hotspot")
            check(
                d is not None and d > 5.0,
                f"min_dist_hotspot grades the miss (got {d})",
            )
            check(
                (num(row, "n_iface_target_res") or 0) > 0,
                "it still contacted the target, just not the epitope",
            )
            collected = collect_one(tmp, "elsewhere", row)
            check(
                collected["status"] == "OK"
                and collected["data"]["hotspot_recall"] == 0.0,
                "the collector keeps a real 0.0 as 0.0",
            )

        case("a binder that ignored the epitope (0.0 is not NA)", t_bound_elsewhere)

        def t_no_contact():
            far = write_complex(tmp / "far.pdb", gap=40.0)
            row = run_worker(tmp, far, "far", "A96,A99")
            check(row["status"] == "OK", "a binder touching nothing is still OK")
            check(num(row, "n_iface_target_res") == 0, "n_iface_target_res is 0")
            check(
                text(row, "iface_target_resnums") == "none",
                "iface_target_resnums is 'none', never blank (blank would collect "
                "as NA and read as 'not measured')",
            )
            check(num(row, "n_contacts") == 0, "n_contacts is 0")
            check(num(row, "hotspot_recall") == 0.0, "recall is 0.0")

        case("a binder in contact with nothing", t_no_contact)

        # --- THE numbering contract ---------------------------------------
        def t_absent_hotspot():
            row = run_worker(tmp, pdb, "badnum", "A96,A500")
            status = str(row["status"])
            check(status.startswith("error:"), f"status is an error (got {status})")
            check(
                "A500" in status and "numbering" in status.lower(),
                "the error names the missing residue and blames the numbering",
            )
            check(
                "90-110" in status,
                f"the error reports the span actually present (got {status[:160]})",
            )
            for col in (
                "hotspot_recall",
                "hotspot_hits",
                "n_iface_target_res",
                "min_dist_hotspot",
            ):
                check(text(row, col) is None, f"{col} is NA, never 0")
            collected = collect_one(tmp, "badnum", row)
            check(
                collected["status"].startswith("error:")
                and collected["data"]["hotspot_recall"] is None
                and collected["path"] == "",
                "the collector emits the error status with every column NA",
            )

        case(
            "a hotspot the target does not have -> ERROR, not recall 0.0",
            t_absent_hotspot,
        )

        def t_renumbered_is_visible():
            # The nastier case: the numbers EXIST but mean different residues,
            # because the chain was renumbered from 1. No error is possible -- only
            # hotspot_resnames can show it.
            renum = write_complex(tmp / "renum.pdb", target_start=1, binder_over=(7, 12))
            row = run_worker(tmp, renum, "renum", "A7,A10,A12")
            check(row["status"] == "OK", "a renumbered target still scores")
            names = text(row, "hotspot_resnames")
            check(
                names == "A7=TRP,A10=GLU,A12=SER",
                f"hotspot_resnames exposes what was really scored (got {names})",
            )
            def resnames(cell: str) -> list[str]:
                return [tok.split("=")[1] for tok in cell.split(",")]

            original = run_worker(tmp, pdb, "orig", "A96,A99,A101")
            check(
                resnames(text(original, "hotspot_resnames")) == resnames(names),
                "same residues, different numbers -- the resnames are the only way "
                "to tell the two structures are describing the same epitope",
            )

        case("a renumbered target: hotspot_resnames is the only witness",
             t_renumbered_is_visible)

        def t_wrong_chain():
            row = run_worker(tmp, pdb, "wrongchain", "B96")
            status = str(row["status"])
            check(
                status.startswith("error:") and "not target" in status.lower(),
                f"a hotspot on a non-target chain is an error (got {status[:120]})",
            )

        case("a hotspot naming a chain that is not the target", t_wrong_chain)

        # --- absent chains -------------------------------------------------
        def t_absent_chains():
            row = run_worker(tmp, pdb, "nodesign", "A96", design_chains="Z")
            status = str(row["status"])
            check(
                status.startswith("error:") and "design chain" in status,
                f"an absent design chain is an error (got {status[:120]})",
            )
            check(
                all(
                    text(row, c) is None
                    for c in ("hotspot_recall", "n_iface_target_res", "n_contacts")
                ),
                "every column is NA for an absent design chain",
            )

            row = run_worker(tmp, pdb, "notarget", "Z96", target_chains="Z")
            status = str(row["status"])
            check(
                status.startswith("error:") and "target chain" in status,
                f"an absent target chain is an error (got {status[:120]})",
            )

            missing = tmp / "does_not_exist.pdb"
            row = run_worker(tmp, missing, "nofile", "A96")
            check(
                str(row["status"]).startswith("error:"),
                "a missing structure file is an error, not a crash",
            )

        case("absent chains and missing files", t_absent_chains)

        # --- the cutoff is a real knob, and it is recorded -----------------
        def t_cutoff():
            tight = run_worker(tmp, pdb, "tight", "A96,A99", contact_cutoff=3.5)
            check(
                num(tight, "hotspot_recall") == 0.0
                and num(tight, "contact_cutoff") == 3.5,
                "a 3.5 A cutoff finds no contact at a 4 A gap, and says so in "
                "contact_cutoff",
            )
            wide = run_worker(tmp, pdb, "wide", "A96,A99", contact_cutoff=8.0)
            check(
                num(wide, "hotspot_recall") == 1.0
                and (num(wide, "n_iface_target_res") or 0) > 6,
                "a wider cutoff finds more -- which is why the cutoff is a column",
            )

        case("the contact cutoff changes the answer (so it is recorded)", t_cutoff)

        # --- a missing output collects as 'missing' ------------------------
        def t_missing_output():
            empty = tmp / "out_never_ran"
            empty.mkdir(parents=True, exist_ok=True)
            ctx_row = collect_one(tmp, "never_ran", pd.Series(dtype=str))
            check(
                ctx_row["status"] == "missing"
                and ctx_row["data"]["hotspot_recall"] is None,
                "a design with no TSV collects as 'missing' with NA columns",
            )

        case("a design the run never produced", t_missing_output)

        # =============== added 2026-09-30: the design side ================

        def t_design_side():
            """The binder's own interface residues, from the same matrix."""
            row = run_worker(tmp, pdb, "dside", "A96,A99")
            check(row["status"] == "OK", f"status OK ({row['status']})")
            # The binder is 6 residues (numbered 1..6) parked over target 96-101.
            check(
                int(row["n_iface_design_res"]) == 6,
                f"all 6 binder residues contact the target "
                f"(got {row['n_iface_design_res']})",
            )
            check(
                text(row, "iface_design_resnums") == "B1,B2,B3,B4,B5,B6",
                f"iface_design_resnums is chain-prefixed and ASCENDING, formatted "
                f"like iface_target_resnums (got {row['iface_design_resnums']})",
            )
            check(
                text(row, "iface_design_resnames") == (
                    "B1=GLY,B2=GLY,B3=GLY,B4=GLY,B5=GLY,B6=GLY"
                ),
                f"iface_design_resnames is 'LABEL=RESNAME', exactly the "
                f"hotspot_resnames format, in the same order as the resnums "
                f"(got {row['iface_design_resnames']})",
            )
            check(
                text(row, "seq_source") == "structure"
                and int(row["seq_len"]) == 6
                and int(row["n_design_res_struct"]) == 6,
                f"the trust columns say identities came from the structure and "
                f"agree on the length (seq_source={row['seq_source']}, "
                f"seq_len={row['seq_len']}, n_design_res_struct="
                f"{row['n_design_res_struct']})",
            )
            got = collect_one(tmp, "dside", row)
            check(
                got["data"]["n_iface_design_res"] == 6
                and got["data"]["iface_design_resnames"].startswith("B1=GLY"),
                "and they round-trip through the collector as real columns",
            )

        case("the design side of the contact matrix", t_design_side)

        def t_design_side_none():
            """Measured-and-empty stays 'none'/0, the way the target side does."""
            far = write_complex(tmp / "far.pdb", binder_over=(96, 101), gap=40.0)
            row = run_worker(tmp, far, "dside_none", "A96,A99")
            check(row["status"] == "OK", f"status OK ({row['status']})")
            check(
                int(row["n_iface_design_res"]) == 0
                and text(row, "iface_design_resnums") == "none"
                and text(row, "iface_design_resnames") == "none",
                f"a binder touching nothing gives 0 / 'none' / 'none' -- the SAME "
                f"convention as n_iface_target_res 0 and iface_target_resnums "
                f"'none', i.e. 'measured, and the answer is nothing'. A blank cell "
                f"would be read back as NA, which means 'not measured' "
                f"(got {row['n_iface_design_res']}, "
                f"{row['iface_design_resnums']}, {row['iface_design_resnames']})",
            )
            got = collect_one(tmp, "dside_none", row)
            check(
                got["data"]["n_iface_design_res"] == 0
                and got["data"]["iface_design_resnums"] == "none",
                "and the collector keeps the 0 as a 0, not as NA",
            )
            # The error case is what produces NA, and it blanks EVERYTHING.
            bad = run_worker(tmp, pdb, "dside_err", "A96", design_chains="Z")
            got_bad = collect_one(tmp, "dside_err", bad)
            check(
                got_bad["status"].startswith("error:")
                and got_bad["data"]["n_iface_design_res"] is None
                and got_bad["data"]["iface_design_resnames"] is None,
                "while an ERROR gives NA for the new columns too -- never 0",
            )

        case("design side: 'none' is measured, NA is not", t_design_side_none)

        # ========== added 2026-09-30: --sequence-column contracts =========

        def t_sequence_column():
            """Identities from a sequence, not from the coordinates."""
            row = run_worker(
                tmp, pdb, "seqcol", "A96,A99",
                sequence="MHKYWD", seq_source="atomium_sequence",
            )
            check(row["status"] == "OK", f"status OK ({row['status']})")
            check(
                text(row, "iface_design_resnames")
                == "B1=MET,B2=HIS,B3=LYS,B4=TYR,B5=TRP,B6=ASP",
                f"the identities come from the SEQUENCE (the backbone is poly-GLY), "
                f"three-letter formatted like hotspot_resnames "
                f"(got {row['iface_design_resnames']})",
            )
            check(
                text(row, "seq_source") == "atomium_sequence"
                and int(row["seq_len"]) == 6,
                f"and seq_source names the column (got {row['seq_source']})",
            )
            check(
                float(row["hotspot_recall"]) == 1.0
                and text(row, "hotspot_resnames") is not None,
                "while every pre-existing column is untouched -- the GEOMETRY still "
                "comes from the structure",
            )

        case("--sequence-column: identities from the designed sequence",
             t_sequence_column)

        def t_sequence_contracts():
            """The two error contracts. Both must name what went wrong."""
            short = run_worker(
                tmp, pdb, "seq_short", "A96,A99",
                sequence="MHK", seq_source="atomium_sequence",
            )
            status = str(short["status"])
            check(
                status.startswith("error:"),
                f"a length mismatch is an ERROR, not a truncated mapping ({status})",
            )
            check(
                "seq_len=3" in status and "n_design_res_struct=6" in status,
                f"and it names BOTH numbers ({status})",
            )
            got = collect_one(tmp, "seq_short", short)
            check(
                all(v is None for v in got["data"].values()),
                "with every column NA -- old and new alike",
            )

            multi = run_worker(
                tmp, pdb, "seq_multi", "A96,A99",
                sequence="MHKYWD/GGGG", seq_source="mkcomplex_sequence",
            )
            status = str(multi["status"])
            check(
                status.startswith("error:") and "MULTI-CHAIN" in status,
                f"a '/' in the sequence is an ERROR naming the problem ({status})",
            )
            check(
                "mkcomplex_sequence" in status,
                f"and names the column that was wrong ({status})",
            )

            empty = run_worker(
                tmp, pdb, "seq_empty", "A96,A99",
                sequence="", seq_source="atomium_sequence",
            )
            check(
                str(empty["status"]).startswith("error:")
                and "atomium_sequence" in str(empty["status"]),
                f"an unresolvable sequence column is an error naming it "
                f"({empty['status']})",
            )

        case("--sequence-column: the two error contracts", t_sequence_contracts)

        # ====== added 2026-09-30: lineage + loud unresolvable rows ========

        def t_lineage():
            """The structure column resolved from an ANCESTOR, not this frame."""
            structures = {"d0": write_complex(tmp / "lin.pdb")}
            # The frame has NO rfdiffusion3_path column at all -- exactly what a
            # child table produced by a sequence designer looks like.
            rows, sub, _ = build_manifest(
                tmp,
                structures,
                input_column="rfdiffusion3_path",
                lookup={
                    "rfdiffusion3_path": str(structures["d0"]),
                    "atomium_sequence": "MHKYWD",
                },
                sequence_column="atomium_sequence",
            )
            check(
                sub[0][1] == str(structures["d0"]),
                f"a structure column ABSENT from the frame is resolved up the "
                f"lineage (got {sub[0][1]!r})",
            )
            check(
                sub[0][3] == "MHKYWD",
                f"and so is the sequence column (got {sub[0][3]!r})",
            )
            check(
                rows[0][4] == "atomium_sequence"
                and rows[0][5] == "rfdiffusion3_path",
                "the task row echoes both, so the worker and the table agree",
            )
            # Precedence: the row's OWN value wins over the ancestor's.
            other = write_complex(tmp / "own.pdb")
            rows, sub, _ = build_manifest(
                tmp,
                structures,
                input_column="boltz_path",
                lookup={"boltz_path": str(other)},
            )
            check(
                sub[0][1] == str(structures["d0"]),
                f"when the column IS on the row, the row wins over the lineage "
                f"(got {sub[0][1]!r})",
            )

        case("lineage: the structure column comes from the parent", t_lineage)

        def t_unresolvable_is_loud():
            """A row that resolves nowhere is SUBMITTED and errors, never dropped."""
            structures = {
                "has_it": write_complex(tmp / "hasit.pdb"),
                "lacks_it": write_complex(tmp / "lacksit.pdb"),
            }
            out_dir = tmp / "manifest_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {"boltz_path": [str(structures["has_it"]), ""]},
                index=pd.Index(list(structures), name="name"),
            )
            args = Namespace(
                run_dir=tmp, table="table1", input_column="boltz_path",
                force=False, gpus_per_task=1, design_chains="B",
                target_chains="A", hotspots="A96,A99", contact_cutoff=5.0,
                designs_per_task=20, sequence_column=None,
            )
            ctx = ManifestCtx(
                df=df, args=args, out_dir=out_dir,
                lookup=lambda name, column: None,
                write_meta=lambda **_: None,
            )
            rows = run_epitope.build_epitope_manifest(ctx)
            sub = [
                line.split("\t")
                for line in Path(rows[0][0]).read_text().splitlines()
                if line.strip()
            ]
            check(
                len(sub) == 2,
                f"BOTH designs are in the manifest -- the unresolvable one is NOT "
                f"silently dropped, which would make the run quietly smaller than "
                f"the table (got {len(sub)})",
            )
            unresolved = [r for r in sub if r[0] == "lacks_it"][0]
            check(
                unresolved[1] == "",
                "the unresolvable design rides with an EMPTY structure field",
            )
            # ...and the worker turns that into an error IN THE TABLE.
            row = run_worker(
                tmp, pdb, "unres", "A96,A99",
                task_line="unres\t\tA96,A99\t\n",
            )
            status = str(row["status"])
            check(
                status.startswith("error:") and "could not resolve" in status,
                f"which the worker records as an error status ({status})",
            )
            check(
                "boltz_path" in status,
                f"naming the column it looked for ({status})",
            )
            got = collect_one(tmp, "unres", row)
            check(
                got["status"].startswith("error:")
                and all(v is None for v in got["data"].values()),
                "and it collects as an error row with NA columns -- a fact in the "
                "table, not an absence",
            )

        case("an unresolvable structure is loud, not a dropped row",
             t_unresolvable_is_loud)

        # ====== added 2026-10-02: hotspot WEIGHT, not just presence ======

        # Three more geometries, all at gap 4.0 so that every contacted residue
        # scores exactly 16 pairs and no neighbour scores any (4.0 A up, 4.0 A
        # along x -> 5.66 A, outside the 5 A cutoff). Nothing here sits ON the
        # cutoff, so no count depends on a floating-point tie.
        far_pdb = write_complex(tmp / "w_far.pdb", binder_over=(108, 110))
        wide_pdb = write_complex(tmp / "w_wide.pdb", binder_over=(90, 110))
        apart_pdb = write_complex(tmp / "w_apart.pdb", gap=30.0)

        def t_hotspot_weight():
            """Hand-countable by construction. Every residue is a 4-atom cluster
            0.5 A across, the binder residues sit 4.0 A above target residues
            96..101, and consecutive target residues are 4.0 A apart along x. So a
            binder residue reaches ONLY the target residue directly beneath it
            (4.0 A, all 16 atom pairs inside the 5 A cutoff) and not its neighbours
            (sqrt(4^2+4^2) = 5.66 A, outside). Therefore:

                each contacted target residue     exactly 16 pairs
                6 contacted residues (96..101)    n_contacts = 96
            """
            row = run_worker(tmp, pdb, "w_three", "A96,A99,A101")
            check(
                num(row, "n_contacts") == 96,
                f"the interface total is the hand-counted 6 x 16 = 96 pairs "
                f"(got {text(row, 'n_contacts')})",
            )
            # (a) the sum over the LISTED residues: 3 x 16.
            check(
                num(row, "hotspot_n_contacts") == 48,
                f"hotspot_n_contacts is the sum of the listed residues' counts, "
                f"3 x 16 = 48 (got {text(row, 'hotspot_n_contacts')})",
            )
            # (b) and the share is exactly that over the interface total.
            check(
                num(row, "hotspot_contact_frac") == 0.5,
                f"hotspot_contact_frac is 48/96 = 0.5 "
                f"(got {text(row, 'hotspot_contact_frac')})",
            )
            # (c) per-hotspot counts, in the order LISTED, with an explicit 0 for a
            # hotspot the binder never touches. A90 is the first target residue,
            # 24 A from the nearest binder atom.
            row = run_worker(tmp, pdb, "w_order", "A101,A90,A96")
            check(
                text(row, "hotspot_contacts") == "A101=16,A90=0,A96=16",
                f"hotspot_contacts follows the order --hotspots was given, and an "
                f"untouched hotspot is an explicit 0, not an omission "
                f"(got {text(row, 'hotspot_contacts')})",
            )
            check(
                text(row, "hotspot_contacts").split(",")[1].split("=")[0]
                == text(row, "hotspot_resnames").split(",")[1].split("=")[0],
                "hotspot_contacts and hotspot_resnames are index-aligned, so the "
                "two can be read against each other residue by residue",
            )
            check(
                num(row, "hotspot_n_contacts") == 32
                and num(row, "hotspot_contact_frac") == round(32 / 96, 4),
                f"and the totals follow the per-hotspot counts "
                f"({text(row, 'hotspot_n_contacts')}, "
                f"{text(row, 'hotspot_contact_frac')})",
            )
            # The whole interface on the listed residues -> a share of exactly 1.0.
            row = run_worker(tmp, pdb, "w_all", "A96,A97,A98,A99,A100,A101")
            check(
                num(row, "hotspot_n_contacts") == 96
                and num(row, "hotspot_contact_frac") == 1.0,
                f"listing every contacted residue gives a share of exactly 1.0 "
                f"(got {text(row, 'hotspot_contact_frac')})",
            )
            # Bound elsewhere: a REAL zero share, with status OK. Not NA.
            row = run_worker(tmp, far_pdb, "w_elsewhere", "A96,A99")
            got = collect_one(tmp, "w_elsewhere", row)
            check(
                row["status"] == "OK"
                and num(row, "n_contacts") > 0
                and got["data"]["hotspot_n_contacts"] == 0
                and got["data"]["hotspot_contact_frac"] == 0.0,
                f"a binder that bound elsewhere has a MEASURED 0 share, not NA "
                f"(n_contacts={text(row, 'n_contacts')}, "
                f"frac={got['data']['hotspot_contact_frac']!r})",
            )

        case("hotspot weight: the share of the interface on the listed residues",
             t_hotspot_weight)

        def t_zero_interface_is_na_not_zero():
            """(d) NA IS NEVER 0, and the two halves of this pair are not the same
            kind of thing. With no interface at all the SHARE is 0/0 -- undefined,
            so NA -- while the COUNT is a real, measured 0."""
            row = run_worker(tmp, apart_pdb, "w_apart", "A96,A99")
            check(row["status"] == "OK", f"the design scored OK ({row['status']})")
            check(
                num(row, "n_contacts") == 0
                and text(row, "iface_target_resnums") == "none",
                f"the binder is 30 A away: zero contacts, 'none' as the epitope "
                f"(got {text(row, 'n_contacts')})",
            )
            got = collect_one(tmp, "w_apart", row)["data"]
            check(
                got["hotspot_n_contacts"] == 0,
                f"hotspot_n_contacts is a real 0 -- it was measured "
                f"(got {got['hotspot_n_contacts']!r})",
            )
            check(
                got["hotspot_contact_frac"] is None,
                f"but hotspot_contact_frac is NA: a share of an interface that does "
                f"not exist is undefined, and a 0.0 here would read as 'the binder "
                f"built its interface elsewhere' (got "
                f"{got['hotspot_contact_frac']!r})",
            )
            check(
                got["hotspot_contacts"] == "A96=0,A99=0",
                f"the per-hotspot counts are still real zeros "
                f"(got {got['hotspot_contacts']!r})",
            )

        case("zero interface: the count is 0, the share is NA",
             t_zero_interface_is_na_not_zero)

        def t_error_row_is_na():
            """(e) An 'error:' row gives NA for all three, exactly like every
            pre-existing column."""
            row = run_worker(tmp, pdb, "w_err", "A500")
            check(
                str(row["status"]).startswith("error:"),
                f"a hotspot the target does not have is an error "
                f"({str(row['status'])[:60]})",
            )
            got = collect_one(tmp, "w_err", row)
            for col in (
                "hotspot_n_contacts",
                "hotspot_contact_frac",
                "hotspot_contacts",
            ):
                check(
                    got["data"][col] is None,
                    f"{col} is NA on an error row (got {got['data'][col]!r})",
                )
            check(
                all(v is None for v in got["data"].values()),
                "and so is every other column -- the new ones did not become a "
                "number where the old ones are NA",
            )

        case("an error row is NA in the new columns too", t_error_row_is_na)

        def t_invariants():
            """(f) The two invariants that must hold on every successful row,
            checked across geometries rather than on one hand-picked case."""
            cases = [
                ("inv_a", pdb, "A96,A99,A101"),
                ("inv_b", pdb, "A90,A91"),
                ("inv_c", pdb, "A96,A97,A98,A99,A100,A101"),
                ("inv_d", far_pdb, "A96,A99"),
                ("inv_e", wide_pdb, "A95,A99,A105"),
                ("inv_f", apart_pdb, "A96,A99"),
            ]
            bad = []
            for name, src, hotspots in cases:
                row = run_worker(tmp, src, name, hotspots)
                if row["status"] != "OK":
                    bad.append(f"{name}: status {row['status']}")
                    continue
                hs = num(row, "hotspot_n_contacts")
                total = num(row, "n_contacts")
                frac = num(row, "hotspot_contact_frac")
                if hs is None or total is None or not (0 <= hs <= total):
                    bad.append(f"{name}: {hs} not within 0..{total}")
                if total == 0:
                    if frac is not None:
                        bad.append(f"{name}: frac {frac} should be NA at total 0")
                elif frac is None or not (0.0 <= frac <= 1.0):
                    bad.append(f"{name}: frac {frac} outside 0..1")
                # The per-hotspot counts must sum to hotspot_n_contacts, and there
                # must be exactly one entry per listed hotspot.
                entries = text(row, "hotspot_contacts").split(",")
                if len(entries) != int(num(row, "n_hotspots")):
                    bad.append(f"{name}: {len(entries)} entries for "
                               f"{num(row, 'n_hotspots')} hotspots")
                if sum(int(e.split("=")[1]) for e in entries) != hs:
                    bad.append(f"{name}: entries do not sum to {hs}")
            check(
                not bad,
                f"across {len(cases)} geometries: 0 <= hotspot_n_contacts <= "
                f"n_contacts, 0 <= frac <= 1 (NA at total 0), and the per-hotspot "
                f"counts sum to hotspot_n_contacts" + (f" -- {bad}" if bad else ""),
            )
            # The share must also be insensitive to WHICH designs are compared: it
            # is a share of this design's own interface, so a big interface and a
            # small one can carry the same value. Spell that out as a measurement.
            small = run_worker(tmp, pdb, "inv_small", "A99")
            large = run_worker(tmp, wide_pdb, "inv_large", "A99")
            check(
                num(large, "n_contacts") > num(small, "n_contacts"),
                f"the wide binder has the bigger interface "
                f"({text(large, 'n_contacts')} vs {text(small, 'n_contacts')} pairs)",
            )
            check(
                num(large, "hotspot_contact_frac")
                < num(small, "hotspot_contact_frac"),
                f"yet the SAME single hotspot carries a SMALLER share of it -- the "
                f"column is a share, not an area ({text(large, 'hotspot_contact_frac')}"
                f" vs {text(small, 'hotspot_contact_frac')})",
            )

        case("the invariants, and 'a share is not an area'", t_invariants)

        # ============ added 2026-09-30: backward compatibility ============

        def t_backward_compatible():
            """With --sequence-column omitted, every pre-existing column is what it
            was. The reference values are the ones the pre-edit tool produced for
            this exact complex."""
            row = run_worker(tmp, pdb, "compat", "A96,A99")
            check(
                float(row["hotspot_recall"]) == 1.0
                and text(row, "hotspot_hits") == "A96,A99"
                and int(row["n_hotspots"]) == 2,
                "hotspot_recall / hotspot_hits / n_hotspots unchanged",
            )
            check(
                int(row["n_iface_target_res"]) == 6
                and text(row, "iface_target_resnums")
                == "A96,A97,A98,A99,A100,A101",
                f"n_iface_target_res and iface_target_resnums unchanged, and still "
                f"in STRUCTURE order (got {row['iface_target_resnums']})",
            )
            check(
                text(row, "hotspot_resnames") is not None
                and "=" in text(row, "hotspot_resnames"),
                "hotspot_resnames unchanged in format",
            )
            check(
                text(row, "design_chains") == "B"
                and text(row, "target_chains") == "A"
                and float(row["contact_cutoff"]) == 5.0,
                "the echo columns unchanged",
            )
            # A 3-field sub-manifest line (the pre-2026-09-30 layout) still runs.
            old_layout = run_worker(
                tmp, pdb, "compat_old", "A96,A99",
                task_line=f"compat_old\t{pdb}\tA96,A99\n",
            )
            check(
                old_layout["status"] == "OK"
                and float(old_layout["hotspot_recall"]) == 1.0,
                f"a 3-field task row (no sequence field) still scores exactly as "
                f"before ({old_layout['status']})",
            )
            # The per-residue file's schema is untouched.
            per_res = pd.read_csv(Path(str(row["path"])), sep="\t")
            check(
                list(per_res.columns)
                == ["chain", "resnum", "resname", "n_contacts", "min_dist",
                    "is_hotspot"],
                f"the per-residue file schema is untouched "
                f"({list(per_res.columns)})",
            )
            # (g) added 2026-10-02: the three new columns are APPENDED, so every
            # pre-existing column keeps its position as well as its value. (The
            # cell-by-cell byte comparison against the pre-edit tool is done
            # outside this file; this is the structural half of it.)
            pre_2026_10_02 = [
                "name", "status", "path",
                "hotspot_recall", "hotspot_hits", "n_hotspots",
                "hotspot_resnames", "n_iface_target_res", "iface_target_resnums",
                "min_dist_hotspot", "n_contacts", "design_chains",
                "target_chains", "contact_cutoff",
                "n_iface_design_res", "iface_design_resnums",
                "iface_design_resnames",
                "seq_len", "n_design_res_struct", "seq_source",
            ]
            result_cols = list(
                pd.read_csv(tmp / "out_compat" / "compat.tsv", sep="\t").columns
            )
            check(
                result_cols[: len(pre_2026_10_02)] == pre_2026_10_02,
                f"the per-design TSV's pre-2026-10-02 header is an unchanged "
                f"PREFIX of the new one (got {result_cols[:len(pre_2026_10_02)]})",
            )
            check(
                result_cols[len(pre_2026_10_02):]
                == ["hotspot_n_contacts", "hotspot_contact_frac",
                    "hotspot_contacts"],
                f"and the only change is the three appended columns "
                f"(got {result_cols[len(pre_2026_10_02):]})",
            )
            # A per-design TSV written by the PRE-2026-10-02 tool -- i.e. one
            # lacking the three columns entirely -- must still collect, with them
            # simply empty. This is the already-collected-table case.
            old_dir = tmp / "out_compat_pre1002"
            old_dir.mkdir(parents=True, exist_ok=True)
            old = pd.read_csv(tmp / "out_compat" / "compat.tsv", sep="\t")
            old = old.drop(
                columns=["hotspot_n_contacts", "hotspot_contact_frac",
                         "hotspot_contacts"]
            )
            old["name"] = "compat_pre1002"
            old.to_csv(old_dir / "compat_pre1002.tsv", sep="\t", index=False)
            got = collect_one(tmp, "compat_pre1002", old.iloc[0])
            check(
                got["status"] == "OK"
                and got["data"]["hotspot_recall"] == 1.0
                and got["data"]["n_contacts"] == 96,
                f"a TSV from before this edit still collects its old columns "
                f"({got['status']})",
            )
            check(
                got["data"]["hotspot_n_contacts"] is None
                and got["data"]["hotspot_contact_frac"] is None
                and got["data"]["hotspot_contacts"] is None,
                "with the three new columns simply NA -- an older run's rows are "
                "not retro-fitted with invented numbers",
            )

        case("backward compatibility: nothing existing moved", t_backward_compatible)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all epitope checks passed")


if __name__ == "__main__":
    main()
