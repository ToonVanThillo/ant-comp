#!/usr/bin/env python3
"""
Dock ONE scaffold into a one-component symmetric architecture with RPXdock, dump
the top poses as structures, and write a per-design TSV with one row per dock.

This is the per-array-task step of the rpxdock tool. One task is one scaffold,
which is also one RPXdock "job" (``ijob 0``) -- so RPXdock's "top across all docks"
and "top for each dock" coincide and there is a single ``--nout-top``.

WHY THE DOCK RUNS IN-PROCESS instead of shelling out to ``python -m rpxdock``
===========================================================================
The natural division -- run the CLI, then reopen its ``_Result.txz`` and dump
structures from it -- does NOT work in a PyRosetta-free image.
``rp.search.result_from_tarball`` rebuilds each stored body by handing an
``io.BytesIO`` to ``rp.Body``, and that branch of ``Body.set_pose_info``
(body.py:86-93) is the ONE branch with no ``use_rosetta`` guard: it calls
``ros.pose_from_file`` unconditionally, which does not exist when PyRosetta is
absent. So a tarball written here cannot be reopened here.

Running the dock in-process keeps the live ``Result`` -- whose ``bodies`` are the
real ``Body`` objects built from the input file, never round-tripped -- and lets us
dump each pose by index under a filename WE choose:

    result.dump_pdb(imodel, fname=<our path>, sym=<architecture>, **kw)

``dump_pdb`` (the singular) is the only entry point that accepts ``fname``;
``dump_pdbs`` (the plural) explicitly refuses it and composes a name from body
labels, job indices and zero-padded model numbers instead. Driving the singular
form per model is what makes the model <-> file <-> score mapping exact rather
than parsed back out of a filename.

``sym=`` is passed explicitly for a second, sharper reason: ``dump_pdb`` otherwise
falls back to ``self.attrs['sym']``, and ``result_to_tarball`` rewrites every attr
to ``repr(v)`` -- so after any tarball round trip that attribute reads ``"'C2'"``,
quotes included, and ``rp.geom.symframes`` would not recognise it. The tarball is
therefore written AFTER the structures are dumped, and the architecture is passed
in from the config either way.

Only the 4-line protocol dispatch is ours (and it is deliberately NARROWER than
``rpxdock/app/dock.py``'s: one-component protocols only). The protocols themselves
are upstream's own ``dock_cyclic`` / ``dock_onecomp``, imported, not copied.

TRUST METRICS
=============
A docking score is a motif-table lookup over residue pairs, and two things can make
it meaningless without raising:

* **secondary structure.** Without PyRosetta, SS comes from willutil's pure-Python
  DSSP. The default ``ilv_h`` tables score HELIX pairs only, so a scaffold read as
  all-loop scores ~0 everywhere and looks merely "bad". ``frac_helix`` /
  ``frac_sheet`` / ``frac_loop`` (measured on the Body RPXdock actually scored) are
  reported so that failure is visible as a column. RPXdock's own tripwire for the
  extreme case is ``body.py:151``: ``assert np.sum(self.ss == 'L') < len(self.ss)``,
  message ``'body is all loops and not sub-body!!'`` -- a design failing on that
  has a coordinate/SS problem in its input, not a docking problem.
* **the frame.** RPXdock does not centre inputs and its samplers assume an
  origin-centred body. ``input_com_dist`` is the distance of the input file's CA
  centre of mass from the origin, measured BEFORE any ``--recenter_input``, and
  ``recentered`` says whether recentring was applied.

``n_chains_in`` is reported for the same reason: ``Body`` collapses every chain of
the input into one and renumbers it from 1, so a multi-chain input is silently
treated as a single long chain. For a cyclic dock of a monomer that is a red flag;
for a cage/dihedral the input is supposed to be a single-chain asymmetric unit.
Since the multi-chain hang (see ``rpxdock_structure.py``) the manifest builder
REFUSES more than one chain for a cyclic dock and more than two for a cage or
dihedral, so this column should now read 1 -- or 2 only on a ``Dx_y``/cage dock.
It stays collected anyway: it is the audit trail for what was docked, and this
worker can also be driven standalone, bypassing the guard.

Errors are recorded as data (status ``error: ...``) so a partial array still
collects.

Normally invoked per array task -- run ``sapia run rpxdock`` rather than calling
this directly.

Usage (standalone):
    python tools/rpxdock/rpxdock_worker.py --name design_0 \\
        --input scaffold.pdb --config configs/design_0.json --out-dir out/
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Sibling module, shipped into the image with the rest of the tool dir
# (`add_local_dir(tool_dir)`) and found because a script's own directory is
# sys.path[0]. Stdlib only -- it is the SAME parser the manifest builder runs at
# submit time, so the chains and CA atoms counted here are the ones the guard
# already passed.
from rpxdock_structure import scan_structure

# Columns of the per-design TSV, in order. Bare names: collect_rpxdock.py hands
# them to the driver, which leaf-prefixes them to rpxdock_<name>. `name`,
# `parent`, `status` and `path` are framework fields and are NOT collected as
# data columns.
TSV_COLUMNS = [
    "name",
    "parent",
    "status",
    "path",
    # the dock itself
    "model",
    "rank",
    "score",
    "rpx",
    "ncontact",
    "reslb",
    "resub",
    "disp",
    # what produced it
    "architecture",
    "nfold",
    "hscore",
    "scaffold_path",
    "result_path",
    "n_docks",
    # trust metrics (identical for every row of one scaffold)
    "n_res",
    "n_chains_in",
    "frac_helix",
    "frac_sheet",
    "frac_loop",
    "input_com_dist",
    "recentered",
    "hscore_seconds",
    "seconds",
]

# Result data variables collected when the protocol produced them. `disp` is the
# one-component cage/dihedral displacement along the symmetry axis; reslb/resub are
# the surviving residue range after --max_trim (and are present even at max_trim 0,
# where they are the whole chain).
OPTIONAL_VARS = ("reslb", "resub", "disp")


def input_frame_metrics(path: Path) -> tuple[float, int]:
    """``(distance of the CA centre of mass from the origin, number of chains)``.

    Measured on the FILE, before RPXdock sees it, so it describes the input the
    caller supplied rather than whatever ``--recenter_input`` turned it into.

    The parsing lives in ``rpxdock_structure.scan_structure`` -- deliberately one
    hand-rolled reader shared with the submit-time guard rather than two, and
    stdlib-only because the image is a slim Debian carrying RPXdock and nothing
    else. Both numbers are collected columns (``input_com_dist``,
    ``n_chains_in``), so the rule behind them must not drift between the two
    phases.
    """
    scan = scan_structure(path)
    if not scan.ca_coords:
        raise ValueError(f"no CA atoms found in {path}")
    coords = np.asarray(scan.ca_coords, dtype=float)
    com = coords.mean(axis=0)
    return float(np.linalg.norm(com)), scan.n_chains


def first_body(result):
    """The single Body this dock used.

    ``Result.bodies`` is normalised by ``process_body_labels`` to a job-by-component
    nesting, so one-component results are ``[[body]]`` whether the protocol passed
    ``[[monomer]]`` (cyclic) or ``[body]`` (onecomp).
    """
    bodies = result.bodies
    while isinstance(bodies, (list, tuple)) and bodies:
        if not isinstance(bodies[0], (list, tuple)):
            return bodies[0]
        bodies = bodies[0]
    raise ValueError("result carries no body; cannot dump structures or report SS")


def ss_fractions(body) -> tuple[int, float, float, float]:
    """``(nres, frac_helix, frac_sheet, frac_loop)`` of the Body actually scored."""
    ss = np.asarray(body.ss)
    n = len(ss)
    if n == 0:
        raise ValueError("body has no residues")
    return (
        n,
        float(np.sum(ss == "H") / n),
        float(np.sum(ss == "E") / n),
        float(np.sum(ss == "L") / n),
    )


def run_protocol(protocol: str, hscore, kw):
    """Dispatch to the one-component RPXdock protocol named by the config.

    Narrower than ``rpxdock/app/dock.py:main`` on purpose -- every other branch of
    that dispatch needs ``--inputs2``/``--inputs3``, which this tool does not wire.
    The run-side manifest builder rejects those architectures before submitting;
    this is the second line of the same defence.
    """
    from rpxdock.app import dock as rpx_app

    if protocol == "cyclic":
        return rpx_app.dock_cyclic(hscore, stack=False, **kw)
    if protocol == "cyclic_stack":
        return rpx_app.dock_cyclic(hscore, stack=True, **kw)
    if protocol == "onecomp":
        return rpx_app.dock_onecomp(hscore, **kw)
    raise ValueError(
        f"unknown one-component protocol {protocol!r} (expected cyclic, "
        f"cyclic_stack or onecomp)"
    )


def dock_one(
    name: str,
    input_pdb: Path,
    config: dict,
    design_dir: Path,
) -> list[dict]:
    """Run one scaffold's dock and return the per-dock rows (best-scoring first)."""
    import rpxdock as rp

    t_start = time.time()
    architecture = config["architecture"]

    # Measured on the input file, before RPXdock touches it.
    com_dist, n_chains_in = input_frame_metrics(input_pdb)

    argv = list(config["argv"]) + [
        "--inputs1",
        str(input_pdb),
        # Note: rpxdock appends f'{architecture}_' to output_prefix itself (and
        # does so BEFORE uppercasing architecture), so the tarball name is not
        # predictable from this string alone. Nothing here depends on it: the
        # tarball is written by us, under a name we choose.
        "--output_prefix",
        str(design_dir / "rpx_"),
    ]
    if config.get("allowed_residues_file"):
        argv += ["--allowed_residues1", config["allowed_residues_file"]]

    kw = rp.options.get_cli_args(argv)

    t_hscore = time.time()
    # Expect 'WARNING: using slower, portable tarball format...' on stdout here for
    # any alias without .pickle sidecars (ilv_h, afilmv_ehl). It is normal output,
    # not a failure.
    hscore = rp.RpxHier(kw.hscore_files, **kw)
    hscore_seconds = time.time() - t_hscore

    result = run_protocol(config["protocol"], hscore, kw)
    if result is None or len(result.data["scores"]) == 0:
        raise ValueError(
            "RPXdock returned no dock for this scaffold -- every sampled placement "
            "clashed or was filtered out"
        )

    body = first_body(result)
    n_res, frac_h, frac_e, frac_l = ss_fractions(body)

    scores = np.asarray(result.data["scores"].data, dtype=float)
    n_docks = len(scores)
    order = np.argsort(-scores)[: int(config["nout_top"])]

    rows: list[dict] = []
    for rank, imodel in enumerate(order, start=1):
        imodel = int(imodel)
        child = f"{name}_d{rank}"
        out_pdb = design_dir / f"{child}.pdb"
        # dump_pdb (singular) is the only form that takes `fname`; `sym` is passed
        # explicitly because the attrs fallback is unreliable (see module docstring).
        result.dump_pdb(imodel, fname=str(out_pdb), sym=architecture, **kw)
        if not out_pdb.is_file():
            raise ValueError(f"dump_pdb wrote no file for model {imodel}")
        row = {
            "name": child,
            "parent": name,
            "status": "OK",
            "path": str(out_pdb),
            "model": imodel,
            "rank": rank,
            "score": float(scores[imodel]),
            "rpx": float(result.data["rpx"].data[imodel]),
            "ncontact": float(result.data["ncontact"].data[imodel]),
            "architecture": architecture,
            "nfold": int(config["nfold"]),
            "hscore": config["hscore"],
            "scaffold_path": str(input_pdb),
            "n_docks": n_docks,
            "n_res": n_res,
            "n_chains_in": n_chains_in,
            "frac_helix": round(frac_h, 4),
            "frac_sheet": round(frac_e, 4),
            "frac_loop": round(frac_l, 4),
            "input_com_dist": round(com_dist, 3),
            "recentered": bool(config["recentered"]),
            "hscore_seconds": round(hscore_seconds, 2),
        }
        for var in OPTIONAL_VARS:
            if var in result.data:
                row[var] = float(result.data[var].data[imodel])
        rows.append(row)

    # Written LAST: result_to_tarball rewrites every attr to repr(v), which would
    # break a later dump_pdb's sym lookup.
    tarball = design_dir / f"{name}_Result.txz"
    rp.search.result_to_tarball(result, str(tarball), overwrite=True)

    seconds = round(time.time() - t_start, 2)
    for row in rows:
        row["result_path"] = str(tarball)
        row["seconds"] = seconds
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in TSV_COLUMNS})


class Args(argparse.Namespace):
    name: str
    input: Path
    config: Path
    out_dir: Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dock one scaffold into a one-component symmetric architecture."
    )
    ap.add_argument("--name", required=True, help="Parent design name.")
    ap.add_argument("--input", type=Path, required=True, help="Scaffold PDB.")
    ap.add_argument(
        "--config", type=Path, required=True, help="Per-design JSON written by the run."
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Tool output dir.")
    args = ap.parse_args(namespace=Args())

    result_tsv = args.out_dir / f"{args.name}.tsv"
    design_dir = args.out_dir / args.name
    design_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = json.loads(args.config.read_text())
        rows = dock_one(args.name, args.input, config, design_dir)
    except Exception as e:  # noqa: BLE001 - errors are recorded as data
        traceback.print_exc()
        write_tsv(
            result_tsv,
            [{"name": "", "parent": args.name, "status": f"error: {e}", "path": ""}],
        )
        # Non-zero so the task is visibly failed in the scheduler too: a scaffold
        # that produced no dock mints NO child row, so the table alone would not
        # show it.
        sys.exit(1)

    write_tsv(result_tsv, rows)
    print(f"{args.name}: OK ({len(rows)} dock(s) kept of {rows[0]['n_docks']})")


if __name__ == "__main__":
    main()
