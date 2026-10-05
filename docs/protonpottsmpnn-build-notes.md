# `protonpottsmpnn` — build notes and decision record

**Who this is for:** an agent or person who has been asked "how does protonpottsmpnn work"
or "why is it built this way", and needs more than the how-to.

- **How to run it** → `docs/protonpottsmpnn-modal-guide.md`
- **Flags, columns, traps** → `.claude/skills/protonpottsmpnn/SKILL.md`
- **Why it exists and how it was validated** → this file

**Provenance.** This is a record written by Claude (Opus) on 2026-10-02 in the session that
built the tool, at Alvaro's request, so later sessions do not re-derive any of it. It is a
**reconstruction of the work and the reasoning, not a verbatim transcript** — every claim
below is one that was executed and checked, and where a number appears it was measured.
Commands that were run are reproduced as they were run.

---

## 1. What was asked, and what was built

The request: wrap <https://github.com/christian-creator/ProtonPottsMPNN> (published
2026-09-30) as a sapia tool producing columns like `atomium` / `proteinmpnn`.

**The science in two sentences.** Proton-PottsMPNN is PottsMPNN with a doubled alphabet:
every titratable residue is two tokens, protonated vs not (`HIS-P`/`HIS-S`,
`ASP-P`/`ASP-D`, `GLU-P`/`GLU-D`), so one energy function can score *"would this residue
rather be charged or neutral here?"*. That lets the design engine pin protonated centres
and rebuild their neighbourhood, trading stability against the protonated-vs-deprotonated
energy gap — which is how you get a binder that grips at pH 5 and releases at pH 7.4.

**The result:** `tools/protonpottsmpnn/`, `action: create`, CPU-only,
`default_input_column = rfdiffusion3_path`.

---

## 2. The three checks that had to pass before building

These are required by the tool-authoring process. Recorded because repeating them is
wasted work.

### Does an existing tool already produce this? — No.

Checked by reading **collector column lists**, not skill prose:

| Tool | Columns it actually writes | Overlap |
|---|---|---|
| `proteinmpnn` | `sequence`, `score`, `seq_recovery` | sequence only |
| `atomium` | `sequence`, `sample`, `temperature`, `seq_rec` | sequence only |
| `cms` | `target`, `binder`, `sc`, `sc_area`, `path` | none — measures folded complexes |
| `pyrosetta` | `if_dG`, `if_dSASA`, `packstat`, `buried_unsat`, … | none — wrong phase |
| `ringfit` | `bridge_ratio`, `hotspot_recall`, `n_clash`, … | none |

Nothing anywhere carries a protonation alphabet or a selectivity term.

### Is it a new scope, or a fork of `atomium`? — New tool.

Same *slot* (backbone → sequences, `create`), but a different model, a different
objective, and a **stricter premise**: `atomium` redesigns any backbone; this one needs a
binder+target complex because centres are placed against the interface. Premise differs →
new tool, per the authoring rule.

### What filter would a caller write? — Stated before building, tested after.

```python
def apply_filter(df):
    return df[(df["protonpottsmpnn_status"] == "OK")
            & (df["protonpottsmpnn_centers_verified"] == 1.0)
            & (df["protonpottsmpnn_selective_energy"] < 0)
            & (df["protonpottsmpnn_pareto"])
            & (df["protonpottsmpnn_n_mut"] > 0)]
```

Lives on the Volume at `/runs/tests/protonpottsmpnn/ph_switch.py`. Verified against a real
table: 8 rows in, 7 out.

> **Correction worth carrying forward.** Early in the session this was written as an inline
> `-f "expr"`. That is wrong — prosapia's `-f/--filter` takes a **path to a Python module**
> defining `apply_filter(df) -> df` (`base_run.py:109`). Caught only when the real run
> happened. Do not write inline filter expressions in this workspace.

---

## 3. Design decisions, and why

| Decision | Why |
|---|---|
| `action: create` | It mints new entities (sequences), one child row per design. |
| One task = one backbone | The engine featurises the structure **once** and then runs the whole λ ladder against the shared Potts tables. Splitting λ across tasks would re-featurise per design. |
| `gpus_per_task = 0`, forced | `run_ph_redesign` forks a CPU pool, and forking after CUDA init is unsafe — so it only parallelises on CPU. Forcing it means callers never need `-g 0`. |
| λ ladder as `--num-designs` | This is the repo's own Pareto sweep (`inference/design_ph.py`). One run returns the whole trade-off curve instead of one arbitrary point. |
| Only `block_descent` + `backend=potts` exposed | The MCMC and autoregressive arms optimise different objectives; exposing them would make the same column mean different things run to run. `--set` is the escape hatch. |
| A `_worker.py` rather than shell | The engine is a library API, and the trust metrics need the featurised context while it is still in memory. |
| Config passed as JSON, not TSV fields | ~20 parameters including nested maps; tab-field surgery in bash would be fragile. |
| Pareto computed per backbone | The two energies are z-scored *within* a featurisation. A table-wide front would be meaningless. |

### The trust metrics, and the reasoning behind them

A pH switch is a claim about **specific residues**, so the expensive failure mode is a
plausible-looking energy computed against a wrong residue mapping. Three columns exist
only to make that visible:

- **`centers_verified`** — fraction of pinned centres whose token in the *output* sequence
  really is the microstate that was pinned. Below 1.0 the design does not carry the centre
  it was optimised for and its `selective_energy` is meaningless.
- **`resnum_offset`** — `res_id` of the binder's first residue minus 1. Generators
  routinely renumber chains from 1; this says whether `--explicit-centers 45` meant the
  target's residue 45 or the 45th residue of the binder. **It fired on the very first real
  structure**: the PD-L1 example is numbered 1..114, offset 0.
- **`n_mut` / `seq_rec`** — against the sequence the redesign actually *started from*.
  `n_mut == 0` means the seed was echoed back, not designed.

### Fail loudly, in three places

1. **Premise guards at submit time** — unknown `--binder-chain`, a deprotonated state as a
   centre, malformed `--explicit-centers`, bad region, λ outside [0,1], an empty
   `--seed-column`. All raise *before* a container starts.
2. **`--placement-region` emptiness** — the engine classifies regions inside
   `try: … except Exception: pass`, so a failed SASA or contact computation leaves an
   all-False mask and placement quietly goes nowhere. The worker checks each requested
   region for a non-empty mask and raises instead.
3. **Errors as data** — a worker failure writes `designs.tsv` with `status = error: …`, so
   the table says *why* rather than showing a blank that reads as a pass.

---

## 4. The two traps that only running it could find

### HBPLUS is required at DESIGN time — the upstream README says otherwise

The repo's own table says *"design a binder → needs HBPLUS: —"*. **That is wrong.**

`prepare_potts_input` builds its inference pipeline through
`get_protonation_state_transforms`, which runs `CalculateHbondsPlus` **unconditionally**
(`pipelines/potts_mpnn.py:269`); the v6 vocabulary consumes those bonds as its feature
pool.

Proven, not inferred — a stub binary was put on `HBPLUS_PATH` and the worker run:

```
INVOKED: -h 3.2 -d 4.0 /tmp/.../20261002111349_9037.pdb /tmp/.../20261002111349_9037.pdb
```

Invoked during design featurisation, with the v6 cutoffs.

**Why it matters:** with `HBPLUS_PATH` unset, `bond_annotation.calculate_hbonds` falls back
to a **hardcoded path on the original author's laptop**
(`/Users/chrjac/.../hbplus`). The failure is then a `FileNotFoundError` on a stranger's
OneDrive directory, hundreds of frames deep. The worker therefore checks `HBPLUS_PATH`
itself, first thing.

**Why it is vendored and git-ignored:** not on PyPI (404), not on conda-forge (empty
search), and the EBI tarball URL serves an HTML page — it is behind a licence form. It
ships under a *confidentiality agreement* (`confid.txt`) requiring it be kept "in a
reasonably secure place to prevent unauthorised access". **This repo is public**, so the
tarball is git-ignored and each user supplies their own; `modal_image.py` fails the build
immediately with instructions when it is absent. This is the same pattern as PyRosetta and
AlphaFold params.

### xgboost needs an OpenMP runtime

With HBPLUS working, the task died at `AnnotateProtonationStates` with *"XGBoost Library
could not be loaded"* — i.e. **after** HBPLUS succeeded, so it reads like a
protonation-labelling bug rather than a missing apt package. The v6 labeller is a pickled
FLAML/xgboost model and xgboost links `libgomp` at import. `libgomp1` is now named
explicitly in `modal_image.py`.

---

## 5. Validation — what was actually run

### Local, macOS/arm64

22-assertion smoke test (`tools/protonpottsmpnn/_smoketest.py`) covering the manifest
builder, the `{expr}` mini-language, all seven guardrails, the single-chain premise, the
collector, and the failure path. Then a real design run on the shipped PD-L1 example.

### Modal — three runs

| Run | What it tested | Result |
|---|---|---|
| `20261002_103217_ph_modal_test` / `table1_ph` | first end-to-end, `--placement-region all`, 2 λ | exit 0, 2 rows, 26 columns, lineage stamped |
| …same run_dir / `table1_iface` | the real premise: `--placement-region interface`, 8 λ | exit 0, 8 rows, monotonic trade-off |
| `20261002_112639_ph_quickstart` | the published quickstart, verbatim, from scratch | exit 0, 8 rows, reproduced `table1_iface` |

The interface run, which is the one worth remembering:

| λ | potts_energy | selective_energy | global_dh | n_mut | pareto |
|---|---|---|---|---|---|
| 0.000 | −51650.7 | −3.75 | 24.2 | 15 | ✔ |
| 0.143 | −51645.4 | −5.62 | 24.1 | 14 | ✘ |
| 0.286 | −51647.3 | −7.31 | 23.9 | 15 | ✔ |
| 0.429 | −51616.0 | −13.90 | 21.9 | 16 | ✔ |
| 0.571 | −51587.2 | −19.98 | 20.2 | 16 | ✔ |
| 0.714 | −51514.9 | −24.25 | 18.5 | 18 | ✔ |
| 0.857 | −51457.5 | −27.05 | 15.5 | 21 | ✔ |
| 1.000 | −51406.0 | −27.78 | 12.4 | 22 | ✔ |

**Why this is convincing, and not just "it ran":**

- **Monotonic in both columns** across all 8 points — selectivity improves 7-fold while
  stability degrades smoothly. That is the objective doing exactly what it claims.
- **`global_dh` fell 24.2 → 12.4 without being optimised at all.** An independent signal:
  the whole binder drifts toward preferring protonation, not just the three centres.
- **Centres verified by hand.** Tokens are parallel to the sequence (114 == 114), position
  8 carries `ASP-P`, 17 `GLU-P`, 28 `HIS-P`.
- **Mutations are neighbourhood-local**: positions 7–19, 27–31, 38–48, 106–107 — clustered
  on the centres, as `--neighbour-k 16` requires.
- **Placement responds to the region flag**: centres at 8/17/28 under `interface`, versus
  33/100 on the same structure under `all`.
- **The Pareto column does real work**: λ=0.143 is dominated by λ=0.286 on *both* axes.

### Two predictions made in advance, one falsified

- *"3–6 of 8 on the Pareto front"* → **wrong, 7 of 8.** A smooth monotonic ladder dominates
  almost nothing. Consequence: `pareto` is a weak gate here, not a strong one.
- *"interface placement will choose different residues than region=all"* → correct.

### `n_mut` can exceed `--max-mutations` — chased, not accepted

`--max-mutations 20` produced `n_mut` of 21 and 22. Not a bug: the cap bounds the
designable **neighbourhood**; the pinned centres are mutations too and sit outside it
(`8 T→D`, `17 A→E`, `28 G→H`). Real ceiling is `--max-mutations + n_centers`. Anyone
filtering `n_mut <= 20` to stay near a validated parent would be silently misled.

### Reproducibility — measured, and it is not bitwise

| Comparison | Agreement |
|---|---|
| Modal → Modal, same input | 5 decimal places |
| macOS → Modal, same seed | **3rd significant figure** (−13.99 vs −14.62; 8 vs 7 mutations) |

Centres chosen, direction of the trade-off and `centers_verified` were identical. The
cause is the protonation labeller: different BLAS/thread counts and a newer xgboost reading
the FLAML pickles flip a few borderline labels, which shifts the energies.
**Compare designs within one run, not across backends.**

### A healthy task writes ~2 KB of `.err`

xgboost "model generated by an older version", a biotite `chararray` DeprecationWarning,
and `fork() may lead to deadlocks` from the CPU pool. Exit `0` with a noisy `.err` is the
normal signature. Documented so nobody chases it.

---

## 6. Deliberate scope limits

The upstream repo has four parts. Only one is wrapped.

| Repo part | Status | Why |
|---|---|---|
| `inference/` design engine | **wrapped** | the thing that makes designs |
| `labeller/` | not wrapped | annotates a structure rather than designing — a different question, and would be an `update` tool |
| `scoring/` + `fold_rf3.py` | not wrapped | RF3's ~3 GB weights are not shipped, and folding is `boltz`'s job here |
| `benchmarks/`, `training/` | not wrapped | not pipeline steps |

**And the standing caveat:** this tool *designs*. It does not fold and it does not measure
binding. A good `selective_energy` is the model's opinion. The honest test is the usual
chain — `mkcomplex` → `boltz` → `usalign` / `cms` / `pyrosetta`. **That has not been run.**

---

## 7. Known gaps

- **Not ported to vib.** Needs a `SAPIA_ACTIVATE_PROTONPOTTSMPNN` activation script and an
  HBPLUS build on the cluster.
- **No calibration for `selective_energy`.** Only the sign test (`< 0`) is established.
  "How negative is good" is unknown; the manuscript ranks within a campaign.
- **Nothing downstream has been run.** No fold, no interface score, so there is no evidence
  these sequences fold, let alone switch.
- **Shared-workspace licensing.** HBPLUS is baked into a Modal image in the shared
  `vubmodal` workspace, so lab members can pull it. That reads as "the department's own
  research" under the agreement, but it is a judgement for the licence holder, not an
  agent. A per-user Secret or private volume is the stricter alternative.
- `tools/bindcraft2/{bindcraft2.sh,modal_image.py}` were found modified during this session
  by something other than this work, and were deliberately left unstaged.

---

## 8. Reference material on the Volume

```
sapia-runs  AND  sapia-runs-alvaro:
  /runs/tests/protonpottsmpnn/
    pdl1_seed_binder.pdb      PD-L1 binder (chain A, 114 aa) + target (chain B)
    seed_table0.py            builds a root table0 from one structure
    ph_switch.py              the filter module above
sapia-runs-alvaro only:
  /runs/outputs/20261002_112639_ph_quickstart/     known-good reference run
  /runs/outputs/20261002_103217_ph_modal_test/     table1_ph (region=all) + table1_iface
```

`/runs/tests/protonpottsmpnn/` exists on both the shared `sapia-runs` Volume and
`sapia-runs-alvaro`, so the guide's paths work for either. The run_dirs above are on
`sapia-runs-alvaro` only. On any other Volume, copy the tests directory across rather than
assuming the path exists.
