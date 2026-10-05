# Running `protonpottsmpnn` on Modal

**What it does:** designs **pH-switchable** binder sequences. It pins protonated residues
(`HIS-P` / `ASP-P` / `GLU-P`) into a binder and redesigns around them, so the binder can
grip at one pH and let go at another.

**What it is:** a `create` tool, same slot as `proteinmpnn` / `atomium` — a backbone goes
in, one table row per designed sequence comes out. CPU only, no GPU.

Every command below was run end to end on 2026-10-02 before this was written. Nothing here
is from memory.

---

## Before your first run: you must supply HBPLUS yourself

**This is the only setup step, and the tool will not build without it.**

Proton-PottsMPNN's protonation labeller shells out to HBPLUS on *every* design call.
HBPLUS is a small C program that is not on PyPI or conda, and its download sits behind an
academic licence form — so it is deliberately **not** in the repo, and each person gets
their own copy.

```
1. Download hbplus.tar.gz from
   https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/   (academic licence)
2. Save it to:   tools/protonpottsmpnn/vendor/hbplus.tar.gz
```

That path is git-ignored on purpose — **do not commit it.** The image build compiles it
for you. If it is missing you get this, in one second, before anything starts:

```
protonpottsmpnn needs the HBPLUS source tarball at .../vendor/hbplus.tar.gz.
...
Get it from https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/ and save the
tarball to that path.
```

If you publish results using this tool you must cite
McDonald IK & Thornton JM (1994), *J Mol Biol* 238:777-793.

---

## Quickstart — copy, paste, done

A ready-made test case lives at `/runs/tests/protonpottsmpnn/` on **both** the shared
`sapia-runs` Volume and `sapia-runs-alvaro`, so the paths below work whichever one your
`.env` points at. It is a PD-L1 binder + target complex — the example shipped with the
upstream repo.

```bash
# 1. make a run_dir  (ALWAYS inside the workstation, never on your laptop)
sapia modal-shell --cmd 'cd /runs && sapia new_run --label ph_quickstart'
#    -> outputs/20261002_112639_ph_quickstart        (yours will differ)

RUN=outputs/<the_run_dir_it_printed>
T=/runs/tests/protonpottsmpnn

# 2. build table0 from the example structure
sapia modal-shell --cmd "cd /runs && python $T/seed_table0.py $RUN $T/pdl1_seed_binder.pdb pdl1_seed"

# 3. design — 8 points along the stability <-> pH-selectivity trade-off
sapia modal-shell --cmd "cd /runs && sapia run protonpottsmpnn $RUN \
    -t table0 -i pdb_path -e modal --table-label ph \
    --binder-chain A --num-designs 8 --placement-region interface --n-jobs 8"

# 4. wait for the task, then collect
sapia modal-shell --cmd "cat /runs/$RUN/table1_ph/protonpottsmpnn/protonpottsmpnn_logs/*.exit"
#    -> 0  means done.  Empty means still running; poll every 30-60 s.

sapia modal-shell --cmd "cd /runs && sapia collect protonpottsmpnn $RUN -t table1_ph"
#    -> Collected 8 row(s) into table1_ph.
```

**Timing:** about 5 minutes for 8 designs. The λ ladder is fanned across `--n-jobs` CPU
workers, so 8 designs cost roughly the same wall time as 1 — budget per *backbone*, not
per design. The very first run in a workspace also builds the image (~15 min, once).

**Cost:** CPU only, no GPU. Pennies.

### On your own target instead of the example

Replace step 2 with your own structure on the Volume. Two requirements:

- It is a **binder + target complex**, and `--binder-chain` names the binder. Every other
  chain is held fixed as context. That is what makes it a *binder* redesign.
- With `--placement-region interface` it **must** have a target chain. On a single-chain
  input the tool refuses to run rather than quietly placing centres anywhere.

---

## What you get back

```yaml
<run_dir>/
  table0.tsv                    # your input structure(s)
  table1_ph.tsv                 # <- THE RESULT: one row per designed sequence
  table1_ph/protonpottsmpnn/
    pdl1_seed/
      designs.tsv               # the same numbers, per backbone
      designs.fasta             # 1-letter sequences - what you fold
      designs_states.fasta      # the 3-letter + protonation-state sequences
    protonpottsmpnn_logs/       # .out .err .exit per task
```

Row names are `<parent>_p0`, `_p1`, … **`_p0` is the most STABLE design, not the best
one** — rows are sorted by stability. Rank by a column, never by the suffix.

---

## How to read `table1_ph`

Here is the real output of the quickstart above:

| name | λ | potts_energy | selective_energy | centers | verified | pareto | n_mut |
|---|---|---|---|---|---|---|---|
| `pdl1_seed_p0` | 0.00 | −51650.7 | −3.75 | `8:ASP-P;17:GLU-P;28:HIS-P` | 1.0 | ✔ | 15 |
| `pdl1_seed_p2` | 0.14 | −51645.4 | −5.62 | same | 1.0 | **✘** | 14 |
| `pdl1_seed_p1` | 0.29 | −51647.3 | −7.31 | same | 1.0 | ✔ | 15 |
| `pdl1_seed_p3` | 0.43 | −51616.0 | −13.90 | same | 1.0 | ✔ | 16 |
| `pdl1_seed_p4` | 0.57 | −51587.2 | −19.98 | same | 1.0 | ✔ | 16 |
| `pdl1_seed_p5` | 0.71 | −51514.9 | −24.25 | same | 1.0 | ✔ | 18 |
| `pdl1_seed_p6` | 0.86 | −51457.5 | −27.05 | same | 1.0 | ✔ | 21 |
| `pdl1_seed_p7` | 1.00 | −51406.0 | −27.78 | same | 1.0 | ✔ | 22 |

### The one idea: λ

Every design minimises a blend of two things, and λ is the dial between them:

```
O = (1 - λ)·z(stability)  +  λ·z(pH selectivity)

λ = 0   pure stability      -> a good binder that barely switches
λ = 1   pure selectivity    -> a strong switch that may not fold
```

`--num-designs 8` walks λ from 0 to 1, so **one run gives you the whole trade-off curve**,
not one answer. Above, selectivity improves 7-fold (−3.75 → −27.78) while stability
degrades by ~245 units. Both columns move monotonically — that is what a healthy run
looks like.

### The columns that matter

**Both energies are LOWER = BETTER, and both are arbitrary units.**

| Column | Read it as |
|---|---|
| `potts_energy` | stability. Lower = stabler. |
| `selective_energy` | the pH switch. Lower = the protonated state is more strongly preferred at the centres. `< 0` is the minimum bar. |
| `global_dh` | whether the *whole binder* leans low-pH, not just the centres. |
| `combined_lambda` | which λ this design came from. |
| `pareto` | `True` = no other design of the same backbone beats it on **both** energies. |
| `centers` | which residues were pinned, e.g. `8:ASP-P;17:GLU-P;28:HIS-P`. |
| `sequence` | the binder sequence — what you fold next. |
| `extended_tokens` | the same sequence with protonation states spelled out. |

### Trust columns — check these FIRST

| Column | Why |
|---|---|
| **`centers_verified`** | fraction of pinned centres actually present in the output sequence. **Below 1.0 the design does not carry the centre it was optimised for, and its `selective_energy` is meaningless.** Always filter on `== 1.0`. |
| `resnum_offset` | `0` means the chain was renumbered from 1, so `centers` are *sequence positions*, not your target's numbering. In the example it is `0` — those `8/17/28` are positions 8/17/28 of the binder. |
| `n_mut`, `seq_rec` | how far it moved from the starting sequence. `n_mut == 0` means nothing was designed. |

### Three gotchas

1. **Energies only compare WITHIN one backbone.** They are z-scored per structure.
   Sorting the whole table by `potts_energy` across different parents ranks nothing.
2. **`n_mut` can exceed `--max-mutations`.** The cap bounds the redesigned *neighbourhood*;
   the pinned centres are extra. Real ceiling is `--max-mutations + n_centers`
   (measured: 22 mutations with `--max-mutations 20` and 3 centres).
3. **A healthy task still writes ~2 KB of `.err`** — an xgboost "older version" warning, a
   biotite deprecation, and a `fork()` notice. Exit `0` with a noisy `.err` is normal.

### Filtering to the ones worth folding

`-f/--filter` takes a **Python module**, not an inline expression. One is ready for you at
`/runs/tests/protonpottsmpnn/ph_switch.py`:

```python
def apply_filter(df):
    return df[
        (df["protonpottsmpnn_status"] == "OK")
        & (df["protonpottsmpnn_centers_verified"] == 1.0)
        & (df["protonpottsmpnn_selective_energy"] < 0)
        & (df["protonpottsmpnn_pareto"])
        & (df["protonpottsmpnn_n_mut"] > 0)
    ]
```

```bash
sapia run mkcomplex $RUN -t table1_ph -f /runs/tests/protonpottsmpnn/ph_switch.py
```

On the quickstart table this keeps 7 of 8 (drops `pdl1_seed_p2`, the one design beaten on
both axes).

---

## The flags you will actually touch

| Flag | Default | What it changes |
|---|---|---|
| `--binder-chain` | `A` | The chain to redesign. Everything else is fixed target. |
| `--num-designs` | 8 | Points on the λ ladder. Cheap — they run in parallel. |
| `--placement-region` | `all` | `interface` is usually what you want for a binder. Also `core`, `surface`. |
| `--center-types` | `HIS-P,ASP-P,GLU-P` | What to place. Repeat for several: `'HIS-P,HIS-P'`. |
| `--explicit-centers` | — | Pin by hand: `'45:HIS-P,78:ASP-P'`. Uses the input's own residue numbers. |
| `--max-mutations` | 20 | Mutation budget around the centres. |
| `--n-jobs` | = `--cpus-per-task` | CPU workers for the λ ladder. |
| `--seed-column` | — | Redesign *from* an existing sequence, e.g. `proteinmpnn_sequence`. |

You do **not** need `-g 0`; the tool forces CPU-only itself.

Full flag list and the deeper traps: `.claude/skills/protonpottsmpnn/SKILL.md`.

---

## What it does NOT tell you

**It designs. It does not fold, and it does not measure binding.** A good
`selective_energy` is the model's opinion, not evidence of a pH switch. The next steps are
the normal ones:

```
protonpottsmpnn  ->  mkcomplex  ->  boltz  ->  usalign          (did it fold back?)
                                           ->  cms / pyrosetta  (is the interface real?)
```

Nothing in this table says these sequences fold at all.

---

## If something breaks

| Symptom | Cause |
|---|---|
| Image build fails instantly, mentions HBPLUS | You skipped the setup step at the top. |
| Task dies at `AnnotateProtonationStates`, "XGBoost Library could not be loaded" | Missing OpenMP. Should not happen on the shipped image — report it. |
| `--placement-region interface` refused | Your input has one chain. It needs binder + target. |
| Run submits 0 designs | Wrong `-i`. Pass the column that actually holds the structure path. |
| `.exit` never appears | Still running, or the app died — `modal app logs <app_id>` using the id in `protonpottsmpnn_modal.json`. |

Known-good reference run: `outputs/20261002_112639_ph_quickstart` on
`sapia-runs-alvaro`. The test inputs in `/runs/tests/protonpottsmpnn/` are on both
`sapia-runs` and `sapia-runs-alvaro`; if you use a third Volume, copy that directory
across.

Deeper background, every design decision and how the tool was validated:
`docs/protonpottsmpnn-build-notes.md`.
