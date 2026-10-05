"""Run one backbone's Proton-PottsMPNN pH-switch redesign and write a per-design TSV.

Invoked by ``protonpottsmpnn.sh`` with one argument: the JSON config
``run_protonpottsmpnn.py`` wrote for this design. Real Python rather than shell because
the engine is a library API (``PottsMPNNPHEngine.run_ph_redesign``), the lambda ladder is
built as a list of dataclasses, and the trust metrics below need the featurised context.

Writes, under ``<out_dir>/``:

    designs.tsv            one row per design + a ``status`` column  (what collect reads)
    designs.fasta          1-letter canonical binder sequences (foldable)
    designs_states.fasta   the parallel 3-letter + protonation-state sequences

Errors are recorded AS DATA: a failure writes ``designs.tsv`` with a single
``status=error: ...`` row and exits non-zero, so a partial run still collects and the
table says why rather than showing an empty cell that reads like a pass.

The trust metrics, and why each one is here
-------------------------------------------
A pH-switch design is a claim about SPECIFIC residues, so the expensive failure is a
plausible energy computed against a residue mapping that is wrong. Three columns exist
only to make that visible:

  * ``centers_verified`` -- the fraction of pinned centres whose token in the OUTPUT
    sequence really is the protonated microstate that was pinned. Anything below 1.0
    means the design does not carry the centre it was optimised for, and its
    ``selective_energy`` is meaningless. Filter on it.
  * ``resnum_offset`` -- ``res_id`` of the binder's first residue minus 1. Generators
    routinely renumber a chain from 1, so this says whether ``--explicit-centers 45``
    meant the target's residue 45 or the 45th residue of the binder. 0 = numbered from
    1 (i.e. renumbered).
  * ``n_mut`` / ``seq_rec`` -- mutations and identity against the sequence the redesign
    actually STARTED from. A design that mutated 0 positions is the seed echoed back,
    not a design.
"""

import json
import sys
import traceback
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

# The engine forks a CPU pool over the lambda ladder, and forking after CUDA init is
# unsafe -- so the device is pinned to CPU before torch is imported by anything below.
# A stray non-boolean DEBUG in the environment breaks environs/foundry at import.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.pop("DEBUG", None)


def _die(out_dir: Path, message: str) -> None:
    """Record the failure as a one-row designs.tsv, then exit non-zero."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "designs.tsv"
    tsv.write_text("status\n" + f"error: {message}".replace("\t", " ").replace("\n", " ") + "\n")
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _check_hbplus() -> str:
    """HBPLUS is a HARD dependency of design, and its absence fails confusingly.

    ``prepare_potts_input`` runs ``CalculateHbondsPlus`` on every call (the v6
    protonation labeller consumes its bonds), and when ``HBPLUS_PATH`` is unset
    ``bond_annotation.calculate_hbonds`` falls back to a hardcoded path on the original
    author's laptop -- so the real error is a ``FileNotFoundError`` on someone's
    OneDrive directory, hundreds of frames deep. Check it here, by hand, first.
    """
    exe = os.environ.get("HBPLUS_PATH", "")
    if not exe:
        raise RuntimeError(
            "HBPLUS_PATH is not set. Proton-PottsMPNN needs the HBPLUS binary at DESIGN "
            "time -- the v6 protonation labeller reads its H-bond geometry -- even "
            "though the repo README says only the labeller needs it. Build HBPLUS into "
            "the image and export HBPLUS_PATH (see modal_image.py)."
        )
    if not (Path(exe).is_file() and os.access(exe, os.X_OK)):
        raise RuntimeError(
            f"HBPLUS_PATH={exe!r} is not an executable file. The protonation labeller "
            f"shells out to it on every design call."
        )
    return exe


def _criteria(config: dict[str, Any], lam: float, PHDesignCriteria):
    """One PHDesignCriteria for one lambda.

    ``method='block_descent'`` + ``backend='potts'`` is the optimiser the manuscript's
    campaigns used; this tool does not expose the MCMC / autoregressive arms, which have
    different objectives and would make the columns mean different things run to run.
    """
    crit = PHDesignCriteria(
        method="block_descent",
        backend="potts",
        combined_lambda=float(lam),
        temperature=float(config["temperature"]),
        samples_per_site=int(config["samples_per_design"]),
        block_size=int(config["block_size"]),
        seed_source="inverse" if config["seed_sequence"] else "native",
        center_types=list(config["center_types"]),
        explicit_centers=list(config["explicit_centers"]),
        dep_map=dict(config["dep_map"]),
        forbidden_tokens=list(config["forbidden_tokens"]),
        placement_by=config["placement_by"],
        placement_region=list(config["placement_region"]),
        neighbour_k=int(config["neighbour_k"]),
        max_mutations=int(config["max_mutations"]),
        record_trajectory=False,
    )
    extra = config.get("extra") or {}
    if extra:
        known = {f.name for f in dataclass_fields(PHDesignCriteria)}
        unknown = sorted(set(extra) - known)
        if unknown:
            raise RuntimeError(
                f"--set names no such PHDesignCriteria field: {', '.join(unknown)}. "
                f"Known fields: {', '.join(sorted(known))}"
            )
        crit = replace(crit, **extra)
    return crit


def _pareto(points: list[tuple[float, float]]) -> list[bool]:
    """Minimise BOTH coordinates; True = not dominated by any other point."""
    out = []
    for i, p in enumerate(points):
        dominated = any(
            j != i and q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
            for j, q in enumerate(points)
        )
        out.append(not dominated)
    return out


def main(config_path: str) -> None:
    config = json.loads(Path(config_path).read_text())
    out_dir = Path(config["out_dir"])
    try:
        _run(config, out_dir)
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        _die(out_dir, f"{type(exc).__name__}: {exc}")


def _run(config: dict[str, Any], out_dir: Path) -> None:
    _check_hbplus()

    from biotite.structure.io.pdb import PDBFile
    from mpnn.inference_engines.potts_mpnn_ph import PHDesignCriteria, PottsMPNNPHEngine

    checkpoint = config["checkpoint"]
    if not Path(checkpoint).is_file():
        raise RuntimeError(
            f"checkpoint not found: {checkpoint}. The v6 checkpoint ships in the repo; "
            f"pass --checkpoint if it lives elsewhere."
        )

    binder_chain = config["binder_chain"]
    engine = PottsMPNNPHEngine(
        checkpoint_path=checkpoint,
        extended_vocab="v6",
        out_directory=None,
        write_fasta=False,
        write_structures=False,
    )
    atom_array = PDBFile.read(config["pdb"]).get_structure(model=1)

    # Featurise once, up front, so the trust metrics below are computed against the SAME
    # context the designs come from. run_ph_redesign builds its own identical context.
    ctx = engine._build_context(atom_array, binder_chain, base_seed=int(config["seed"]))
    binder_idx = ctx.chainA_all_idx.tolist()
    seq_pos_of_ctx_pos = {p: i for i, p in enumerate(binder_idx)}
    binder_res_ids = [int(ctx.token_aa.res_id[int(p)]) for p in binder_idx]
    native_sequence = ctx.decode_canonical(ctx.S_native, binder_idx)
    n_free = int(len(ctx.chA_free_idx))
    resnum_offset = binder_res_ids[0] - 1 if binder_res_ids else 0

    # region_masks is FAIL-SOFT in the engine: the interface and core/surface passes are
    # each wrapped in `except Exception: pass`, so a failed SASA or contact computation
    # leaves an all-False mask and placement quietly has nowhere to go. A requested
    # region that is empty is an error here, not a design with centres somewhere else.
    for region in config["placement_region"]:
        if region == "all":
            continue
        mask = ctx.region_masks.get(region)
        n_positions = 0 if mask is None else int(mask.sum())
        if n_positions == 0:
            raise RuntimeError(
                f"--placement-region {region!r} selects 0 binder positions on this "
                f"structure. The engine classifies regions fail-soft (an all-False mask "
                f"on error), so this would otherwise place centres as if the region "
                f"were never requested. For 'interface', check the target chains are "
                f"present and in contact; for 'core'/'surface', check the structure has "
                f"side chains (a poly-glycine backbone has no RASA split)."
            )

    # The sequence the redesign starts FROM -- what n_mut/seq_rec are measured against.
    seed_sequence = config["seed_sequence"]
    if seed_sequence:
        if len(seed_sequence) != len(binder_idx):
            raise RuntimeError(
                f"--seed-column gave a sequence of length {len(seed_sequence)} but "
                f"chain {binder_chain} has {len(binder_idx)} residues. The seed must be "
                f"the binder chain ALONE -- if it came from a complex-sequence column "
                f"(e.g. mkcomplex_sequence) it carries the target chains too."
            )
        reference = seed_sequence
        initial_sequences = [seed_sequence]
    else:
        reference = native_sequence
        initial_sequences = None

    n_jobs = int(config["n_jobs"]) or (os.cpu_count() or 1)
    criteria = [_criteria(config, lam, PHDesignCriteria) for lam in config["lambdas"]]
    print(
        f"{config['name']}: binder chain {binder_chain}, {len(binder_idx)} residues "
        f"({n_free} designable), resnum {binder_res_ids[0]}..{binder_res_ids[-1]}; "
        f"{len(criteria)} lambda(s) on {n_jobs} worker(s)"
    )

    designs = engine.run_ph_redesign(
        atom_array=atom_array,
        binder_chain=binder_chain,
        criteria_list=criteria,
        seed=int(config["seed"]),
        initial_sequences=initial_sequences,
        n_jobs=min(n_jobs, len(criteria)) if len(criteria) > 1 else 1,
    )
    if not designs:
        raise RuntimeError(
            "the engine returned 0 designs. With placement criteria this usually means "
            "no candidate position survived --placement-region/--center-types."
        )

    rows: list[dict[str, Any]] = []
    for d in designs:
        res_ids = list(d.center_res_ids or [])
        types = list(d.center_protonation_types or [])

        # Trust metric: does the OUTPUT sequence actually carry each pinned microstate at
        # the pinned residue? A centre whose res_id is not a binder position at all, or
        # whose token came back something else, is counted as not verified.
        verified = 0
        seq_positions: list[str] = []
        for res_id, want in zip(res_ids, types):
            ctx_pos = ctx.res_id_to_pos.get(int(res_id))
            if ctx_pos is None:
                seq_positions.append("NA")
                continue
            i = seq_pos_of_ctx_pos[ctx_pos]
            seq_positions.append(str(i + 1))
            if i < len(d.extended_tokens) and str(d.extended_tokens[i]) == want:
                verified += 1
        centers_verified = (verified / len(res_ids)) if res_ids else 0.0

        sequence = d.canonical_sequence
        n_mut = sum(1 for a, b in zip(sequence, reference) if a != b)
        seq_rec = (
            sum(1 for a, b in zip(sequence, reference) if a == b) / len(reference)
            if reference
            else 0.0
        )

        rows.append(
            {
                "design_id": d.design_id(),
                "sequence": sequence,
                "extended_tokens": " ".join(d.extended_tokens),
                "potts_energy": d.final_potts_energy,
                "selective_energy": d.selective_energy,
                "global_dh": d.global_protonation_dH,
                "combined_lambda": d.combined_lambda,
                "sample": d.sample,
                "centers": ";".join(f"{r}:{t}" for r, t in zip(res_ids, types)),
                "center_seqpos": ";".join(seq_positions),
                "n_centers": d.n_centers if d.n_centers is not None else 0,
                "centers_verified": round(centers_verified, 4),
                "n_designable": d.n_neighbours if d.n_neighbours is not None else n_free,
                "binder_chain": binder_chain,
                "binder_len": len(sequence),
                "resnum_offset": resnum_offset,
                "n_mut": n_mut,
                "seq_rec": round(seq_rec, 4),
                "seed_source": "inverse" if seed_sequence else "native",
                "status": "OK",
            }
        )

    # Pareto within THIS backbone only -- the energies are z-scored per backbone and are
    # not comparable across parents, so a table-wide front would be meaningless.
    front = _pareto(
        [
            (float(r["potts_energy"]), float(r["selective_energy"] or 0.0))
            for r in rows
        ]
    )
    for row, on_front in zip(rows, front):
        row["pareto"] = bool(on_front)

    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with open(out_dir / "designs.tsv", "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for row in rows:
            fh.write("\t".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    with open(out_dir / "designs.fasta", "w") as fh:
        for row in rows:
            fh.write(f">{row['design_id']} lambda={row['combined_lambda']}\n{row['sequence']}\n")
    with open(out_dir / "designs_states.fasta", "w") as fh:
        for row in rows:
            fh.write(
                f">{row['design_id']} lambda={row['combined_lambda']}\n"
                f"{row['extended_tokens']}\n"
            )

    n_front = sum(front)
    n_bad = sum(1 for r in rows if r["centers_verified"] < 1.0)
    print(
        f"{config['name']}: {len(rows)} design(s), {n_front} Pareto-optimal"
        + (f", {n_bad} with unverified centres" if n_bad else "")
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: protonpottsmpnn_worker.py <config.json>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
