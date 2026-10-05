---
name: atomium
description: How to run the custom atomium tool on Modal — AtomiUM, a private ProteinMPNN-like sequence designer with noise-conditioned weights. Covers --model-noise and how to pick it, the multi-temperature sampling count, the --bias-aa calibration trap (it is global not positional, and ~16x stronger than exp(bias) at the default sampling temperature, so it must be fitted empirically and can make an "exactly one of X" criterion worse), what seq_rec really measures, the private-repo image built with Modal Secrets, and why its FASTA carries no score column (so designs cannot be ranked the way proteinmpnn's score allows). Load before composing an atomium run, before setting any amino-acid bias, or when interpreting its columns.
---

# atomium

**Custom tool** (lives in `tools/atomium/`, not bundled with prosapia).
**`action: create`** — mints a child table, one row per designed sequence.

AtomiUM is a private PyG re-implementation of ProteinMPNN-style sequence design, with
**noise-conditioned weights**: one checkpoint per training noise level. It ships
ProteinMPNN's own helper scripts and takes the same jsonl inputs, so this tool is
deliberately the `proteinmpnn` tool with the same two mini-languages, the same
signature-grouping/bin-packing, and the same lineage contract. **Everything you know
about composing a proteinmpnn run transfers** — read that skill for the mini-languages.

Repo: `AndreiSokolovskii/develop_atomium`, branch `pure_wo_jit`, **private**.

## Verified invocation

```bash
sapia run atomium <run_dir> -t table0 --num-seq-per-target 2
sapia collect atomium <run_dir> -t <the table the run reserved>
```

`default_input_column` is **`rfdiffusion3_path`** — unlike the bundled proteinmpnn,
whose default is the older `rfdiffusion_path` and silently submits nothing after an
rfd3 run. Coming from a different parent, pass `-i` explicitly.

## Flags that matter

| Flag | Default | Note |
| --- | --- | --- |
| `--model-noise` | `n05` | **The one flag that is not in proteinmpnn.** Which weight file to design with: `n00`…`n07` (training sigma 0.0 Å…0.7 Å), plus the `n05_v2` retrain. Validated at submit time, so a typo fails fast instead of per container. |
| `--num-seq-per-target` | 2 | Sequences per backbone **per temperature**. |
| `--sampling-temp` | `0.1` | One or more space-separated temperatures: `'0.1 0.2'`. See the count rule below. |
| `--seed` | 37 | **`0` means pick a random seed**, it does not mean seed zero. |
| `--designs-per-task` | 10 | Designs bin-packed per task; identical-param designs share one batched call. |
| `--chains-to-design`, `--fixed-positions`, `--tied-positions`, `--symmetry`, `--bias-aa`, `--set` | | Identical to proteinmpnn, including the guardrail that positions require a chain list. |

Default Modal resources: **L4**, 8 CPU, 16 GiB, 1 h timeout. Weights ship inside the
image — no cache Volume, no download step.

### Picking `--model-noise`

The suffix is the coordinate noise the checkpoint was **trained to denoise**, so it is a
statement about how much you trust the input geometry:

- **Low (`n00`–`n02`)** — trusts the backbone as given. For crystal structures or
  already-relaxed models.
- **Mid (`n03`–`n05`)** — the useful default range for **de novo backbones**, which carry
  generator-specific geometry quirks. `n03` is AtomiUM's own default; this tool defaults
  to `n05`.
- **High (`n06`–`n07`)** — tolerates rough or coarsely sampled backbones.

Noise level is a *cheap* axis to scan: run the same parent table twice with different
`--table-label`s and compare downstream self-consistency, not the sequences themselves.

```bash
sapia run atomium <run_dir> -t table0 --table-label n03 --model-noise n03
sapia run atomium <run_dir> -t table0 --table-label n05 --model-noise n05
```

## The sampling count is a product

`BATCH_COPIES = num_seq_per_target * len(temperatures)` — each temperature gets a **full**
`--num-seq-per-target` set. So `--num-seq-per-target 4 --sampling-temp '0.1 0.2 0.3'` is
**12 sequences per backbone**, not 4. Easy way to accidentally triple a run.

There is **no `--batch-size`**: `atomium.py` accepts the flag but never reads it. Do not
pass it through `--set` expecting an effect.

## `--bias-aa` is far stronger than `exp(bias)` — calibrate it, never reason about it

`--bias-aa` takes space-separated `AA:bias` pairs and writes a single run-wide `bias_AA.jsonl`
(`make_bias_AA.py` is purely generative, no structure input). Two things follow, and both have
bitten:

**It is GLOBAL, not positional.** One logit offset applied to every position of every design in
the run. There is no per-position bias path in the manifest builder. So it cannot place a residue
*somewhere specific* — it raises that residue everywhere, buried core included. `--fixed-positions`
does not help either: it *freezes* whatever the input structure already has, it does not choose.
**To get a residue at a particular place, oversample and select on a measured column.**

**Its magnitude does not follow `exp(bias)` at low sampling temperature.** Measured on EGFR dIII
binders, 2310 sequences per arm, `--model-noise n05`, default `--sampling-temp 0.1`:

| `--bias-aa` | global His frequency | mean His per ~80-residue binder |
| --- | --- | --- |
| none | 0.615% | 0.49 |
| `H:1.39` | **14.238%** | ~11.4 |

`exp(1.39) = 4.0`, so a plain logit offset would predict ~2.5%. The observed shift is **23×, not
4×** — sequences with up to **52** histidines. The mechanism is almost certainly the temperature:
at T = 0.1 the distribution is nearly deterministic, so a modest shift flips the biased residue to
the argmax at every position where it was anywhere near competitive.

**Practical rule: fit the multiplier from two runs before trusting a third.** Those two points
imply `exp(2.26 × bias)` here, so the bias landing the mean at 1.0 is ≈ `H:0.31`, not 1.39. A
bias tuned at one `--sampling-temp` does not transfer to another.

**Nor does it transfer across amino acids, and the SIGN matters more than the magnitude.**
Measured 2026-10-01 on 1107 EGFR dIII binder sequences, `--model-noise n05`,
`--sampling-temp 0.1`, `--bias-aa C:-1.0`, against a 10,944-sequence unbiased baseline:

| | unbiased | `C:-1.0` | ratio |
| --- | --- | --- | --- |
| Cys frequency | 1.41% | **0.774%** | **1.8×** |
| mean Cys per ~75-residue binder | 1.11 | **0.584** | **1.9×** |

`exp(2.26 × bias)` predicted **10×**. Observed **1.9×** — and even plain `exp(bias)` would predict
2.7×, so **suppression came in weaker than a naive logit offset**, while the His enrichment above
came in 5.7× *stronger* than naive. Fitting this arm alone gives `exp(0.64 × bias)`.

**The asymmetry is the rule: at low temperature, enrichment is amplified and suppression is
attenuated.** Near-deterministic sampling means a positive bias flips the residue to argmax
everywhere it was merely competitive, while a negative bias only bites where its margin was
already thin. Suppression saturates against the model's conviction. **So budget a negative bias
of roughly `exp(0.6 × bias)` and a positive one of roughly `exp(2.3 × bias)` as starting points,
and expect to need a filter as well when suppressing.**

**A bias cannot break a COUPLED choice, only reduce how often it is made.** In the same measured
arm, cysteines still came in pairs: **93.7% of sequences carried an EVEN number** (818 with zero,
only 36 with one, 193 with two, 28 with three, 25 with four) — essentially unchanged from the
unbiased 94%. Two cysteines stayed five times commoner than one. The bias talked some sequences
out of building a disulfide at all; it never broke a pair apart. Where a residue's placement is
structurally coupled rather than sampled per position, **a per-position logit offset is the wrong
instrument — gate on the measured count instead, and treat the bias as a way to improve the yield
of that gate, not to replace it.**

**And a strong bias can make a "exactly one of X" criterion WORSE, not better.** The count of
sequences carrying exactly one copy peaks when the mean is 1.0. Pushing the mean past that
overshoots: the same selection went from 210 designs / 25 backbones (unbiased, mean 0.49) to
28 / 8 at `H:1.39`. Loose criteria (`>= 1`) improve; strict ones collapse.

**Always verify the bias actually applied**: `cat <out_dir>/bias_AA.jsonl` and confirm
`--bias_AA_jsonl` appears in the staged task argv. An unbiased arm has no such file. Without that
check a no-op bias produces a run that looks entirely normal and silently duplicates its control.

## The image is built from a private repo with Modal Secrets

`modal_image.py` clones the repo at **build** time using the `github-token` and
`github-username` Modal Secrets, then deletes `.git` so the credentialed remote is not
baked into the image.

**This has to happen at build time.** prosapia's Modal executor hardcodes the task
secret to the run's `.env` (`executors/modal.py`) and exposes no per-tool `secrets()`
hook, so a named Secret is simply not reachable from a running task. Cloning in the
task (as an ad-hoc Modal script would) also re-clones a private repo on every container.

Consequences:

- **You need access to those two Secrets** in the Modal workspace, or the image build
  fails. They are workspace-level, not personal.
- The checkout is **pinned to a commit** in `modal_image.py` (`ATOMIUM_COMMIT`). The
  branch is under active development, so bump that constant deliberately — a rebuild
  will not drift on its own.
- First run pays a **long image build inside the submit call** (torch 2.11 + cu128 and
  ~470 MB of weights). Allow 15+ minutes before assuming a first submit is stuck; later
  runs are seconds.

`torch_cluster` is a **required** dependency and easy to drop: nothing imports it
directly, but `model_lib.py` calls `torch_geometric.nn.radius_graph`, which is a
torch-cluster binding. Without it every task dies at model construction.

## What it collects

Child rows named `<parent>_a1`, `<parent>_a2`, … each linked to its parent backbone.
(The `_a` suffix distinguishes them from proteinmpnn's `_f` rows.)

Columns (leaf-prefixed `atomium_`): `sequence`, `sample`, `temperature`, `seq_rec`,
`iteration`, plus `_status` and `_path` (the FASTA).

`atomium_sequence` is what the structure predictors consume. Note it is **not** Boltz's
default input column (that is `proteinmpnn_sequence`), so a Boltz run after atomium
needs `-i atomium_sequence` — or `-i mkcomplex_sequence` on a binder table, where
mkcomplex still belongs in between.

### There is no score column — this changes how you rank

AtomiUM's FASTA reports only sequence recovery. **There is no `score` or `global_score`**,
so the model-likelihood ranking that `proteinmpnn_score` supports **does not exist here**.

- **`seq_rec` is not a quality metric, but it is NOT meaningless either** (corrected
  2026-09-30 — this entry previously claimed rfd3 backbones are "effectively poly-glycine"
  and the column close to meaningless, and **that was wrong**). rfd3 emits real side chains
  and real residue identities, so `seq_rec` is a genuine comparison. Measured on 32 designs
  from 8 rfd3 backbones: values **0.21–0.54**, nowhere near zero, and varying **systematically
  by backbone** (one parent 0.209–0.254 across its draws, another 0.433–0.537). It therefore
  appears to track **backbone designability** — how strongly the geometry dictates a sequence —
  rather than sequence quality. Do not rank designs by it, and do not dismiss it; whether it
  predicts anything downstream is still open.
- **`sample` is not unique per row.** AtomiUM numbers samples `n % num_seq_per_target` and
  walks temperatures with `n // num_seq_per_target`, so with several temperatures the same
  `sample` id recurs once per temperature. Use the row name, or the
  (`sample`, `temperature`) pair, to identify a draw.

Rank atomium designs by what comes **after** them — predictor confidence and, above all,
self-consistency (`usalign` back to the parent backbone). That is the honest test for any
sequence designer, and here it is the only one available.
