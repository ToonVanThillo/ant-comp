---
name: boltz
description: >-
  How to run the boltz structure-prediction tool on Modal — flags, its weight cache Volume,
  timing, and the confidence columns it writes. Also covers binder complexes: why iptm is diluted
  by chain count, where the binder-specific pair_chains_iptm/chains_ptm/PAE actually live, the
  per-chain MSA policy flag --msa-empty-chains (target searched, de-novo binder empty) and the
  user-side shadow tool that provides it, the forced-template schema and its silent failure modes
  (_entity_poly_seq, label_asym_id, sequence-search fallback), and the shard-count traps. Load
  before composing a boltz run, and before interpreting any confidence number from a complex.
---

# boltz

Predicts structures for designed sequences. **`action: update`** — annotates the table it
reads, in place. It does **not** create a table, and `-t` is required.

There is no dedicated doc page; `sapia run boltz --help` is authoritative.

IMPORTANT: prosapia's bundled boltz is a wrapper of the original. Check the github repo for all information: https://github.com/jwohlwend/boltz/tree/main/docs

## Verified invocation

```bash
sapia run boltz <run_dir> -t table1
```

10 sequences predicted in ~16 min on one task, including the first-time weight download.
Later runs reuse the cache and start much faster.

`default_input_column` is `proteinmpnn_sequence`, so straight after a ProteinMPNN step no
`-i` is needed. From any other source, pass `-i <sequence column>`.

## Flags that matter

| Flag | Default | Note |
| --- | --- | --- |
| `--shard-size` | 10 | YAML inputs per shard directory / task. |
| `--devices` | 1 | GPUs per task; also sets `--gpus-per-task` to match. |
| `--use-msa-server` | off | Adds MSA information **to every chain**. Slower, and calls an external server — don't enable it without being asked. |
| `--msa-empty-chains` | unset | Chain mini-language (`B`, `A,C`, `B:D`). Chains pinned to `msa: empty` **even when `--use-msa-server` is on** — name the de-novo binder chain. Requires `--use-msa-server`; an unknown chain is an error. See the per-chain MSA section. |
| `--chains` | all | Chain mini-language, e.g. `A:D`. Letters beyond the sequence's chain count are dropped. |
| `--positions` | full | Position mini-language: `/` maps groups onto chains, `start:end` inclusive 1-indexed, open ends allowed (`10:`, `:50`), `{expr}` islands resolve up the lineage. |
| `--template-yaml` | none | File spliced verbatim as a `templates:` block into every input YAML. |

Default Modal resources: **A10**, 24 CPU, 64 GiB, 8 h timeout.

## The weight cache

Weights and the CCD download on first use into the Volume named by `SAPIA_MODAL_VOLUME_BOLTZ_CACHE`, mounted at `/boltz_cache`. It holds `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/` and `mols.tar`. Budget extra time on the very first run of a fresh cache; never delete that Volume to "clean up".

## What it collects

Columns added to the **same** table (leaf-prefixed `boltz_`): `confidence_score`, `ptm`, `iptm`, `ligand_iptm`, `protein_iptm`, `complex_plddt`, `complex_iplddt`, `complex_pde`, `complex_ipde`, plus `_status` and `_path` (the predicted structure).

Reading them: `confidence_score` and `complex_plddt` around 0.9+ is a confident prediction for a de-novo monomer; `ptm` tracks global fold confidence. High confidence means the predictor believes the fold — it is **not** proof the sequence folds to the backbone it was designed for. For that, compare the prediction back to its parent backbone with `usalign`.

##  Important notes

- Forced templates steer; they don't constrain fully (e.g.: `force: true, threshold: 2.0`). Use the flag anyway but always verify independently and exclude rows where the target didn't land.

## Binder complexes: what the collected confidence does and does not mean

**`iptm` is diluted by chain count.** With N protein chains it averages N(N−1)/2 interchain
pairs. In a binder complex all but a handful of those are target–target — and if you supplied a
template, they are pairs *you forced*. A 9-chain complex has 36 pairs of which **35 are
target–target and 1 is binder–target**, so the collected `iptm` is overwhelmingly a statement
about template reproduction.

Measured on a 27-design binder campaign against a trimmed 4-protomer target:

| | |
| --- | --- |
| `confidence_score` vs **target** RMSD | **r = −0.896** — a perfect separator (landed 0.706–0.736, distorted 0.542–0.634, empty band) |
| `confidence_score` vs **binder-interface** ipTM, among rows that landed | **rho = −0.250** |

So: **use `confidence_score` as a gate for "did the target assemble", then stop using it.** It
carries no binder signal once past that gate. The highest-`confidence_score` design in that set
also had the highest target pLDDT and near-bottom binder-interface confidence.

### The binder-specific numbers, and where to get them

Not collected by prosapia. They are in the per-prediction confidence JSON at
`<out_dir>/boltz_results_shard_*/predictions/<name>/confidence_<name>_model_0.json`:

- `chains_ptm` — `{"0": …, "1": …}`, **chain index is positional from the input YAML**, so the
  binder is the last index if `mkcomplex` prepended the target.
- `pair_chains_iptm` — full N×N. **Not symmetric** (observed max asymmetry 0.079); symmetrise
  by averaging both directions. The diagonal just repeats `chains_ptm`.

Also on disk per prediction: `plddt_<name>_model_0.npz` (`plddt`, 0–1, per token) and
`pae_<name>_model_0.npz` (`pae`, n_res × n_res). The cif's `B_iso_or_equiv` carries the same
pLDDT on a 0–100 scale (agrees to 7e-5). There is **no PAE or per-residue data in the JSON**.

Caveat: `pair_chains_iptm` is whole-chain vs whole-chain, so a binder touching a small patch of
a large chain is still diluted. An **interface-restricted PAE** (contacting residue pairs only,
from the npz) is the sharper metric and is not provided by anything here.

### The cheapest honest interface signal

**Compare the binder's mean pLDDT alone vs. in complex.** A real interface raises it. Measured:
all 27 designs *lost* 7.5–23.3 points (mean 13.7) on docking — the signature of a
confidently-folded domain the predictor is confidently unsure where to place. This was more
informative than any ipTM flavour.

Rough reference points: a believed interface is ~0.5–0.6 binder-target ipTM and < 10 Å interface
PAE. The best in that campaign were 0.357 and 19.5 Å.

## MSA policy differs per chain in a binder complex

`--use-msa-server` is off by default and the early runs here used `msa: empty` everywhere. For a
**de-novo binder that is correct** — no MSA exists, a search is meaningless, and it would put an
unpublished design sequence on an external server. For a **natural target it is not**: denying
it an MSA weakens target assembly *and* interface confidence, and the result then reads as bad
designs. Boltz supports the split (an entity with `msa: empty` is skipped, an entity without the
key is searched), and **`--msa-empty-chains` is how you ask for it**:

```bash
# 2-chain complex: A the natural target (searched), B the de-novo binder (no MSA)
sapia run boltz <run_dir> -t table1 -i mkcomplex_sequence \
    --use-msa-server --msa-empty-chains B
```

This is the **only configuration in which the "did the target land" gate means anything** — a
target folded without an MSA may fail that gate for reasons that have nothing to do with the
binder. Say which policy you used when reporting; it is recorded in the run's
`<out_dir>/.meta.json` (`use_msa_server`, `msa_empty_chains`) and visible per design in
`<out_dir>/boltz_inputs/<name>.yml`.

Its contract, all enforced at **submit** time (nothing is queued if it raises):

| Situation | Behaviour |
| --- | --- |
| Flag unset | **Byte-identical to bundled boltz**, with or without `--use-msa-server`. Regression-tested. |
| `--msa-empty-chains` without `--use-msa-server` | **Error.** It cannot be honoured (nothing is searched anyway), and accepting it would hide a forgotten `--use-msa-server`. |
| A named chain the complex does not have | **Error**, naming the chains that *are* present. Never a silent no-op — that would mean the binder quietly got an MSA. |
| A named chain whose sequence is shared with an unnamed chain | **Error.** Identical sequences collapse into one boltz entity, which has one MSA policy. Name all of them or none. |
| A named chain that `--chains` dropped | **Error** (it is no longer in the complex). |
| Every chain named | Same YAML as plain `msa: empty` everywhere. |

Chain letters are **positional in the `/`-chainbreak sequence** (first segment is `A`), *after*
`--chains` has narrowed it — not the chain names in any PDB you started from. After `mkcomplex`
prepended a target, the binder is the last letter.

What it does **not** do: supply a *pre-computed* MSA (boltz's `msa: <path.a3m>` form) — the
choice is only searched vs empty; and it cannot tell you whether a searched chain actually found
homologues. A natural target with no hits behaves like an empty one and no column says so.

### Where this flag lives

It is **not** in prosapia's bundled boltz. `tools/boltz/` in this repo is a user-side **shadow**
(`get_builtin("boltz").with_overrides(...)`, replacing only `add_run_args_fn` and
`build_manifest_fn`); the task script, Modal image and collector are still the bundled ones, and
the tool keeps the name `boltz`, so every column and every downstream step is unchanged.
Consequence: like every other custom tool here, **it is only present when `sapia` is run with
this repo's `tools/` on `$PROSAPIA_TOOLS_DIR`** — i.e. `sapia modal-shell` launched from the
project root, or the synced vib workspace. Launch the workstation from elsewhere and boltz
silently reverts to the bundled tool, where `--msa-empty-chains` is simply an unrecognised flag
(argparse errors — it does not run with the flag ignored). Self-test:
`uv run python tools/boltz/test_boltz.py`.

## Templates

**Do not hand-write the CIF.** `utils/boltz_template.py` builds it and the matching YAML,
and verifies every trap below by re-reading the written file. It is not a sapia tool — run it with
the workstation's python:

```bash
sapia modal-shell --cmd 'python /runs/utils/boltz_template.py \
    --fetch 7OJG --out /runs/inputs/7ojg_tmpl_kabc.cif \
    --chains K,A,B,C --rename-to A,B,C,D --residues K:19-59+107-155 \
    --yaml /runs/inputs/7ojg_tmpl_kabc.yaml'
```

`--chains` picks source chains in order, `--rename-to` renames positionally, `--residues` trims
per **source** chain (`K:19-59+107-155`), `--predict-chains` sets the YAML's `chain_id` when the
prediction's chain IDs differ from the template's. `--check <file> --expect-chains A,B,C` verifies
a CIF from anywhere and exits non-zero. Outputs go under `/runs/` — each `modal-shell` call is a
fresh container. `--fetch` downloads from RCSB inside the container, which is how inputs get
staged here (`modal volume put` is unreliable on this network).

**The script lives at `/runs/utils/boltz_template.py` in the workstation.** It is NOT under
`$PROSAPIA_TOOLS_DIR/_utils/` — that path appeared here until 2026-09-30 and fails with `rc=2`.

**`--check` does not verify everything it should.** A clean pass prints three `OK` lines
(`label_asym_id == auth_asym_id`, `_entity_poly_seq` count vs modelled, `template chains
present`) and **says nothing about `label_seq_id`**, even though a `.` there silently gives a
residue no token. Until the checker covers it, verify separately before spending — with gemmi,
count `_atom_site.label_seq_id`: it must have **zero `.` values** and exactly as many distinct
values as the chain has residues. Measured on a good file: 0 dots, 193 distinct, 193
`_entity_poly_seq.mon_id` rows.

The rest of this section is why that script exists — read it before overriding any of its choices.

`--template-yaml` is spliced **verbatim** at column 0 after the `sequences:` block, so the file
must start at column 0. Schema read from `boltz/data/parse/schema.py`:

```yaml
templates:
  - cif: /runs/inputs/target.cif
    chain_id: [A, B, C, D]      # prediction chains
    template_id: [A, B, C, D]   # template chains
    force: true
    threshold: 2.0              # mandatory when force is true
```

- **Give `chain_id` AND `template_id`, equal length.** Then Boltz zips them positionally — an
  explicit 1:1 map. Omit either and it falls back to `get_template_records_from_search`, which
  scores sequence alignments to pick chains. **With sequence-identical chains (any homo-oligomer)
  that can silently assign the wrong template chain and destroy the geometry.**
- **Boltz names template chains by `label_asym_id`/subchain, not auth chain ID.** A CIF written
  by gemmi defaults to `Axp`, `Bxp`, … and `template_id: [A, …]` then raises *"Template chain A
  is not one of the protein chains"*. Set `label_asym_id == auth_asym_id`.
- **`_entity_poly_seq` must equal the modelled residue count for every entity.** `parse_polymer`
  uses it as the token list, and entries absent from the model still **consume token indices** —
  so a too-long declaration silently shifts every residue index downstream. A common way to get
  this wrong: copied gemmi residues keep their source `subchain` label, `setup_entities()` then
  merges two chains into one entity and takes `full_sequence` from whichever came first. **Count
  the `_entity_poly_seq.mon_id` rows per entity and check them before spending.**
- **`_atom_site.label_seq_id` must be written and run 1..N per chain** — it is the index Boltz gives each template residue, and a `.` means that residue has no token. gemmi's `assign_label_seq_id()` only numbers residues whose `entity_type` is `Polymer`, and that flag is inherited from the source file: a plain PDB or a bare `_atom_site` mmCIF (most target files here) carries no entity annotation, so hand-built clones arrive as `Unknown` and every residue is skipped without a word. `boltz_template.py` sets it explicitly (fixed 2026-09-30) and raises if any residue is left unnumbered. The symptom, if it ever comes back: `--check` says *"declares N residues but N−k are modelled"* on a file that is otherwise correct, where k is the number of **adjacent identical residue pairs** in the sequence — the checker dedups residues on `(label_seq_id, comp_id)` and `(".", "PRO") == (".", "PRO")`. That is a real bug in the file, never a reason to loosen the check. Self-test: `uv run python utils/test_boltz_template.py`.
- Do not template the binder chain.
- Forced templates steer, they do not constrain. **Always superpose the predicted target chains
  back onto the template and exclude rows where the target did not land.** In one batch only
  9 of 27 landed, and the distorted ones produced the largest interfaces and the best hotspot
  recall in the set.

## Two operational traps

- **`Submitting N designs` counts SHARDS**, not designs (`--shard-size`, default 10). Two designs
  print `Submitting 1 designs`. Count `<out_dir>/boltz_inputs/*.yml` instead.
- **`boltz_shards/` is never cleared** (`mkdir(exist_ok=True)`). A rerun in a run_dir that already
  held a pilot will re-predict rows you meant to skip. Check the staged inputs first.
