---
name: prosapia
description: The prosapia workbench model and the sapia CLI contract — run_dirs, tables and lineage, create vs update, the base run/collect flags, labels, the ready set, and the traps that make a run silently do nothing. Load before composing any sapia command; tool-specific flags live in the per-tool skills.
---

# prosapia

`prosapia` (CLI `sapia`) is a **workbench, not a pipeline**. There is no fixed order of steps: there is one tabular data format and tools that consume and produce it. Your job is to compose one step correctly, run it, and report what landed in the table.

To define how tools interact with the data interface, prosapia establishes the following principles:

1. A design is a row, a generation of designs is a table.

2. When a protein diverges in sequence or structure, it is no longer the same protein but a child of a parent — therefore it needs a new table.

Authoritative references: **The installed package does not ship `docs/`** — only `src/`. Query https://github.com/jlmoraleshellin/prosapia/tree/dev (dev branch specifically) plus `sapia run <tool> --help` for source of truth docs.

| Question | Page |
| --- | --- |
| Base `sapia run` flags, executors, the task contract | `docs/running-a-tool.md` |
| `sapia collect` | `docs/collecting-a-tool.md` |
| `create` vs `update`, roots, lineage, `lookup` | `docs/lineage-and-tables.md` |
| `-l/--dir-label` vs `--table-label` | `docs/using-labels.md` |
| The workstation, tool images, `.exit` files | `docs/running-on-modal.md` |
| `-f/--filter` | `docs/writing-a-filter-function.md` |
| Per-tool guides | `docs/tools/{rfdiffusion,rfdiffusion3,proteinmpnn}.md` |

`sapia run <tool> --help` (run it **in the workstation**) is always authoritative for a
tool's own flags. Don't guess flag names.

## The data model

```
run_dir/
├── _registry.tsv      catalog of every table: gen, table_label, parent_table, tool, created_at
├── table0.tsv         one row per design, keyed by `name`
├── table1.tsv         a child table (the next generation)
├── .manifests/        transient per-run manifests
└── table0/            per-table, per-tool outputs on disk
    └── rfdiffusion3/  = run_dir/<table>/<leaf>/
```

- **A design is a row; a generation of designs is a table.** Each tool contributes columns to a row, never files to another tool.
- **Columns are leaf-prefixed**: the leaf is `<tool>`, or `<tool>_<dir_label>` if you passed `-l`. Every tool writes `<leaf>_status` and usually `<leaf>_path`. **`<leaf>_status == "OK"` is the only proof a design succeeded.**
- **Row lineage** lives in `parent_name`, `parent_table`, `gen`. A root table's
  `parent_table` is the sentinel `root`.
- Only `sapia new_run` mints a run_dir. No tool ever does.

## The verbs

`sapia` has exactly these: `new_run`, `run`, `collect`, `modal-shell`, `init`, `fork-tool`.

```bash
sapia new_run --label <label>        # prints the run_dir path and nothing else
sapia run     <tool> <run_dir> [-t <table>] [flags]
sapia collect <tool> <run_dir> -t <table> [-l <dir_label>]
```

**There is no verb for reading a table.** Read TSVs with shell or pandas inside the workstation (recipes below).

## Two phases, and where output lands

`run` fans tasks out — each writes only its own files. `collect` folds those files into the
table. This is what keeps table writes safe under parallelism and reruns idempotent.

**You never name the output table.** The driver derives it at submit time from the tool's
`action`:

- **`create`** (rfdiffusion3, rfdiffusion, proteinmpnn) reserves a **child table**, `table<gen+1>[_<label>]`, registers it before any task runs, and links each new row to its parent. With no `-t`, it starts a fresh root, `table0`.
- **`update`** (boltz, alphafold3, colabfold, openfold3, usalign, make_symmdef,align_symm_axis) writes back into the table it read. `-t` is **required**; without it the run errors.

So for a `create` tool the `Output:` line that `sapia run` prints is how you learn the child table's name — you need it for `collect -t`. Never guess it; `_registry.tsv` also has it.

```
Submitting 5 designs
Output:  outputs/2026…_run/table1/proteinmpnn          <- <run_dir>/<table>/<leaf>
Logs:    outputs/2026…_run/table1/proteinmpnn/proteinmpnn_logs
```

## Base run flags

Shared by every tool (each tool adds its own on top).

| Flag | Default | Meaning |
| --- | --- | --- |
| `run_dir` (positional) | — | From `sapia new_run`. Pass it exactly as printed, relative to `/runs`. |
| `-t`, `--table` | — | Source table, name without `.tsv`. Required for `update`; omit on a `create` to start a root. |
| `-i`, `--input-column` | the tool's `default_input_column` | Which column feeds the tool. **Check it every time** — see the traps. |
| `-l`, `--dir-label` | `""` | Same-tool variants in one table. Leaf becomes `<tool>_<label>`. Must be repeated on `collect`. |
| `--table-label` | `""` | `create` only. Names the child table `table<gen>_<label>`; labels accumulate down generations. |
| `-f`, `--filter` | none | **Path to a `.py` file** defining `apply_filter(df) -> df`, applied to the source frame before the manifest. Not an inline expression — see the traps. |
| `--force` | off | Re-submit designs this tool already finished. |
| `-e`, `--executor` | `$SAPIA_EXECUTOR` (`modal` here) | Leave alone. |
| `-C`, `--max-concurrent` | `40` | Containers running at once. |
| `-g`, `--gpus-per-task` | `1` | **`-g 0` for CPU-only tools** (proteinmpnn, usalign). |
| `-c` / `--mem` / `-T` | tool's `RESOURCES` | CPUs / memory / wall time per task. |
| `--modal-gpu` | tool's `RESOURCES["gpu"]` | GPU type, e.g. `A100`. With `-g N>1` becomes `TYPE:N`. |

### The ready set

You never subset by hand. The driver submits rows with a **present `--input-column`**,
minus those already `OK` for this leaf, unless `--force`. A `-f` filter is applied first;
the framework's own filtering applies on top of whatever it returns. Because the module
sees the whole frame before the manifest is built, it is also the supported way to run a
tool on **one named design** or a small subset — `return df[df["name"] == "design_3"]` —
which is how you buy a cheap probe before committing a batch.

The already-`OK` skip only fires for **`update`** tools, whose status column lives in the
table being read. For a `create` tool that column lives in the child table, so **every**
ready row is submitted on a rerun.

## Collect

```bash
sapia collect <tool> <run_dir> -t <table> [-l <dir_label>] [--force]
```

`-t` is always required and names the table the **run wrote to**: the derived child for a
`create`, the same table for an `update`. There is no `-i` — collect reuses the column the
run recorded in its `.meta.json` sidecar, so the phases can't disagree. `-l` must match the
run's. It prints `Collected N row(s) into <table>.`; rows already `OK` are skipped unless
`--force`.

- `create` stamps lineage on each new row and **fails** if a child has no real parent row.
- `update` only **warns** if it produces a row that isn't in the table, and marks a ready
  design with no output on disk as `missing`.

## Labels

| | `-l` / `--dir-label` | `--table-label` |
| --- | --- | --- |
| Scopes | output dir + columns *within* a table | the child *table* name |
| Phases | run **and** collect (must match) | run only |
| Tools | any | `create` only |
| Use for | same-tool variants (seeds, params) on the same designs | forks of the lineage |

Collect has no `--table-label` — the label is part of the table name, so pass that name to
`-t` (e.g. `-t table1_lowT`).

## Reading tables and the registry

All from the workstation (cwd `/runs`), never locally. It has pandas.

```bash
# what tables exist, and how they relate
sapia modal-shell --cmd 'cat <run_dir>/_registry.tsv'

# columns of a table
sapia modal-shell --cmd 'head -1 <run_dir>/table1.tsv | tr "\t" "\n"'

# selected columns, all rows
sapia modal-shell --cmd 'python -c "
import pandas as pd
df = pd.read_csv(\"<run_dir>/table1.tsv\", sep=\"\t\")
print(df[[\"name\",\"boltz_status\",\"boltz_confidence_score\"]].to_string())"'

# how many designs actually succeeded
sapia modal-shell --cmd 'python -c "
import pandas as pd
df = pd.read_csv(\"<run_dir>/table1.tsv\", sep=\"\t\")
print(df[\"boltz_status\"].value_counts())"'
```

`--cmd` is base64-wrapped before it reaches the container, so quotes inside it are safe.

## Traps

Each of these produces a *successful-looking* command that did the wrong thing.

- **`No designs to submit.` exits 0.** A wrong `-i`, an empty filter or an all-`OK` table
  prints that one line and returns success. Always read the submit output: if you don't
  see `Submitting N designs`, nothing was queued — say so instead of moving on to collect.
- **Default input columns rarely match.** `proteinmpnn`'s is `rfdiffusion_path`
  (RFdiffusion, *not* rfdiffusion3) — from an rfd3 table you must pass
  `-i rfdiffusion3_path`. The predictors default to `proteinmpnn_sequence`. Check the
  parent table's header before composing the command.
- **A `create` rerun from the same parent reuses the same child table name.** `table1` is
  derived from the parent's `gen`, not from what already exists, and re-registering an
  identical lineage is a no-op. So a second rfd3 run on `table0` writes into the same
  `table1` — and a *different* `create` tool on `table0` also derives `table1` and shares
  it. Pass `--table-label` whenever you branch, or you will merge two experiments.
- **A `--dir-label` mismatch fails at collect**, with `Output dir not found` — collect
  looks under the labelled leaf.
- **`-f` is a module *path*, not an expression.** It is declared `type=Path` and loaded
  with `importlib.util.spec_from_file_location`; the module must define
  `apply_filter(df) -> df`, or you get
  `AttributeError: … does not define an 'apply_filter' function`. Passing a pandas-style
  string fails at submit with `ImportError: Could not load module from <your expression>`
  — measured, and it costs a submit.
- **A `-f` filter module must live on the Volume.** The workstation image carries only
  prosapia's source, your tools dirs and `.env`; a local path won't exist there. Write the
  filter into the run_dir through the workstation (heredoc via `--cmd`) and pass that path.
  `modal volume put` is unreliable on this network.
- **Not every tool runs on Modal.** A tool needs a `modal_image.py`. Present for
  rfdiffusion3, proteinmpnn, boltz, alphafold3, colabfold, usalign. **Absent** for
  rfdiffusion (v1), openfold3, make_symmdef, align_symm_axis — those cannot run here; say
  so rather than trying.
- **Run `sapia` from `/runs`, never `cd` into a run_dir.** Stored paths are relative to the
  cwd, and tasks start at `/runs`; a table collected from inside a run_dir points at files
  no later task can resolve.
- **`sapia` never runs locally.** There is no `/runs` on this machine.

## Tool roster

| Tool | action | default `-i` | Modal | Notes |
| --- | --- | --- | --- | --- |
| `rfdiffusion3` | create | `pdb_path` | yes | Root run takes no `-t`. Skill: `rfdiffusion3`. |
| `proteinmpnn` | create | `rfdiffusion_path` | yes | CPU → `-g 0`. Skill: `proteinmpnn`. |
| `boltz` | update | `proteinmpnn_sequence` | yes | Skill: `boltz`. |
| `alphafold3` | update | `proteinmpnn_sequence` | yes | Needs its own image/params env. |
| `colabfold` | update | `proteinmpnn_sequence` | yes | |
| `usalign` | update | *n/a* | yes | No input column: selects on `--col-a`, plus `--col-b` (resolved up the lineage) or `--ref`. Forces its own `-g 0`. Skill: `usalign`. |
| `rfdiffusion` | create | `pdb_path` | **no** | |
| `openfold3` | update | `proteinmpnn_sequence` | **no** | |
| `make_symmdef` | update | `pdb_path` | **no** | |
| `align_symm_axis` | update | `make_symmdef_path` | **no** | |

Load the per-tool skill (`rfdiffusion3`, `proteinmpnn`, `boltz`, `usalign`) before
composing that tool's flags; this page only covers what every tool shares.

One caveat this page's `.exit` logic inherits: a task script may catch its own errors and
still exit `0` (usalign does). Exit codes prove the tasks *ran*; the `<leaf>_status` column
after collect is what proves they *worked*.
