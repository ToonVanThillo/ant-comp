---
name: epitope
description: How to run the custom epitope tool on Modal — did this binder land on the epitope I chose, and which target residues does it actually touch? Turns hotspot recall and the real contact footprint into filterable columns for a single-chain (or any) target. Covers --design-chains/--target-chains/--hotspots, the contact cutoff, the hotspot-numbering ERROR contract (a missing hotspot is never a recall of 0), the hotspot_resnames trust column, the BINDER-side interface columns (n_iface_design_res / iface_design_resnums / iface_design_resnames), the hotspot WEIGHT columns (hotspot_n_contacts / hotspot_contact_frac / hotspot_contacts) that say how CENTRAL a hotspot is to the interface rather than merely whether it was touched, and why their share is NA at a zero interface, --sequence-column for taking design-chain identities from a designed sequence instead of the backbone's generator artifacts, the lineage resolution that makes -i rfdiffusion3_path work on a child table, NA-is-never-zero, the dir-label rule, and every column it collects. Load before composing an epitope run, or before believing any statement about where a binder bound.
---

# epitope

**Custom tool** (lives in `tools/epitope/`, not bundled with prosapia). For each design it
counts heavy-atom contacts between the binder chain(s) and every residue of the target
chain(s), then reports **how much of the requested epitope the binder covers**, **which
target residues it actually touches**, and **which of the binder's own residues do the
touching**. Pure Python — gemmi reads the structure, numpy does the distance matrix — so
the Modal image is `debian_slim + gemmi + numpy`. No scipy, no biopython, no GPU.

**Touched, and how much.** `hotspot_recall` is binary per residue; `hotspot_contact_frac`
is the share of the whole interface the listed hotspots carry. Use the second when nearly
every design in a batch grazes the residue and the real question is whether the interface is
*built around* it.

**Both axes of one matrix.** The contact matrix has a target side and a design side and
both are reported from the same single pass: `n_iface_target_res` / `iface_target_resnums`
/ `hotspot_*` on one side, `n_iface_design_res` / `iface_design_resnums` /
`iface_design_resnames` on the other. There is no second tool and no second cutoff.

**`action: update`**: where a binder landed is a property of a design that already exists,
so it annotates the table it reads, in place. `-t` is required. **CPU-only** — the builder
forces `gpus_per_task = 0`, so you never need `-g 0`.

## Why it exists

A binder can be confident, well folded, well packed and **bound to the wrong face of the
target**. Nothing in `boltz`'s confidence, `pyrosetta`'s `if_dG` or `cms`'s `sc` can tell
you: a good interface somewhere else scores exactly like a good interface where you asked
for one. The epitope question needs its own columns.

The tools that came closest and why they do not cover it (their collector column lists
were read, not their prose):

- **`cms`** — collects `target, binder, sc, sc_area, sc_median_dist, n_atoms_*,
  n_radius0_*, device, seconds` plus `cms_path`, a per-residue file (`side, chain, resnum,
  resname, cms`). The epitope map is in **a file**, not in a column: there is no
  per-design recall to filter on, and reading that file by hand is what this workspace
  exists to avoid.
- **`pyrosetta`** — `if_dG, if_dSASA, if_hbonds, if_delta_unsat, packstat, …`: how good the
  interface is, never where it is.
- **`ringfit`** — does collect `hotspot_recall` / `hotspot_hits`, but its premise is a
  binder straddling **two protomers of an oligomeric assembly with a lipid belt**
  (`bridge_ratio`, `n_clash` against the rest of the ring, `lipid_clash`, an alignment onto
  a `--ref-structure`). On a single-chain target with no assembly and no membrane those
  numbers are undefined and the failure mode it detects cannot occur. It also requires a
  reference structure this question does not have. Its hotspots are matched against a
  reference-to-design residue mapping, and an unmatched hotspot is a **warning** there, not
  an error. Different premise → a different tool, not a fork.
- **`ssprofile`** — `n_iface_res` counts residues on the **binder** side and says nothing
  about which target residues they touch.

## Premise and scope limits

> **Heavy-atom CONTACT between two sets of chains in one structure file, scored against a
> list of target residues you chose.** Nothing else.

- **It measures contact, not buried surface area and not energy.** `n_contacts` is a count
  of atom pairs, not an area. For interface **area** use `cms` (`cms_target`, `sc`,
  `sc_area`) or `pyrosetta` (`if_dSASA`); for interface **energy** use `pyrosetta`
  (`if_dG`). A recall of 1.0 says the binder is *in contact with* your hotspots, not that
  it buries them well.
- **It says nothing about whether the structure it scored is trustworthy.** It scores the
  coordinates it is handed. On a predicted complex it must be read **after** the gate that
  says the target landed (`usalign` of the predicted target chains against the template —
  see the `binder-campaign` skill). **Never before.** A prediction that put the target in
  the wrong place still yields a tidy `epitope_hotspot_recall`, of a fiction.
- **No assembly, oligomer or membrane assumptions.** That is `ringfit`.
- **It does not know what a good epitope is.** It answers "did you hit the residues you
  named", and the hotspot list is yours.
- **One structure file, two chain sets.** It never aligns two structures and never
  renumbers anything, so there is no alignment RMSD to report — the numbers in the columns
  are whatever the input file carries, which is exactly why `hotspot_resnames` exists.

## The numbering contract — the reason this tool is worth its columns

**A hotspot residue number the target chain does not have is an `error` status, with every
column NA. It is never scored as a recall of 0.0.**

```
error: hotspot(s) A96,A99,A101 do not exist in the target chain(s) of d2.pdb -- the
numbering does not match. Target spans A: 1-21 (21 residues). NOT scored as a miss: a
renumbered chain would give a meaningless hotspot_recall
```

Generators and predictors renumber chains from 1 as a matter of course (rfd3 does). A
mis-numbered hotspot quietly scoring 0.0 is **indistinguishable from a genuinely bad
binder**, so a campaign would discard good designs while believing its data. The error
message reports the residue-number span actually present in each target chain, because
that is the number you need to fix the flag.

That covers the case where the numbers are **absent**. The nastier case is numbers that
**exist but mean something else** — a renumbered target where `A96` is a real residue, just
not yours. No error is possible there, so the tool reports:

| `epitope_hotspot_resnames` | Read as |
| --- | --- |
| `A96=TRP,A99=GLU,A101=SER` | the residue names found at each listed position, **in the order you listed them** |

**Check it against your reference structure the first time you run on any new structure
column.** If those are not the residues you chose, every other column is answering a
different question. This is the one column to look at before trusting a recall.

### Measured in this workspace: the EGFR target files are already renumbered

`hEGFR_311-503.cif` and `mEGFR_311-503.cif` are named for EGFR residues **311–503** but
their chain A is numbered **1–193** (verified 2026-09-30: first residue `A1 = LYS`, last
`A193`). So **file residue N = native EGFR residue N + 310**, and a hotspot list written in
native EGFR numbering (`A406`, `A440`, …) either errors or — if it happens to fall inside
1–193 — silently scores a completely different residue. Only `hotspot_resnames` would show
it.

Residue names at the file-numbered positions, for checking a list against the reference:

| file | `hEGFR_311-503.cif` | native (file + 310) |
| --- | --- | --- |
| `A96` | THR | 406 |
| `A99` | HIS | 409 |
| `A101` | GLN | 411 |
| `A155` | LYS | 465 |

**Whichever numbering the hotspots were chosen in, confirm `epitope_hotspot_resnames` on
the first run.** Note too that a design's own copy of the target may be renumbered again by
the generator or predictor, so the check is per structure column, not once per campaign.

## NA is never 0

| Situation | `_status` | `_hotspot_recall` | `_hotspot_hits` | `_iface_target_resnums` |
| --- | --- | --- | --- | --- |
| Binder on the epitope | `OK` | `1.0` | `A96,A99,A101` | the residues touched |
| **Binder bound elsewhere** | `OK` | **`0.0`** | **`none`** | the *other* site it used |
| Binder touching nothing | `OK` | `0.0` | `none` | **`none`** (and `n_iface_target_res` `0`) |
| Hotspot absent from the target | `error: …` | **NA** | NA | NA |
| Hotspot on a non-target chain | `error: …` | NA | NA | NA |
| Design or target chain absent | `error: …` | NA | NA | NA |
| Worker crashed / no output | `error: worker crashed` / `missing` | NA | NA | NA |
| **Structure path unresolvable** | `error: could not resolve …` | NA | NA | NA |
| **`--sequence-column` length ≠ structure** | `error: sequence/structure length mismatch …` | NA | NA | NA |
| **`--sequence-column` holds a `/`** | `error: … MULTI-CHAIN …` | NA | NA | NA |

For the three 2026-10-02 weight columns the rule has one extra wrinkle worth spelling out:
a binder touching **nothing** gives `hotspot_n_contacts` `0` and `hotspot_contacts`
`A96=0,A99=0` — real, measured zeros — but `hotspot_contact_frac` **NA**, because a share of
an interface that does not exist is undefined. Every `error:` row gives NA for all three.

The **design-side** columns follow the identical rule: a binder touching nothing gives
`n_iface_design_res` `0` with `iface_design_resnums` and `iface_design_resnames` both the
literal string **`none`** — *measured, and the answer is nothing*. Every `error:` row gives
NA for them, exactly as it does for the target-side columns. `0`/`none` and NA are never
interchanged in either direction.

A real zero and a failed measurement are **different answers** and the tool never conflates
them. Note the deliberate string **`none`**: an empty cell would be collected as NA and read
as "not measured", so "measured, and the list is empty" is spelled out.

Consequence for filtering: **always pair the metric with the status**. `epitope_hotspot_recall
< 0.5` alone silently includes nothing for the error rows (they are NaN, so comparisons are
False) — but on the inverse filter, forgetting the status is how you would mistake a
numbering bug for a design failure.

## The sequence / structure identity split *(added 2026-09-30)*

The contact **geometry** always comes from the structure. The design chain's residue
**identities** come from the structure too — *unless* you name `--sequence-column`, in
which case they come from that column's sequence, mapped positionally onto the design
chains' amino-acid residues in structure order.

**Why this exists, concretely.** In this workspace the sequence designer (`atomium`,
`proteinmpnn`) writes the binder sequence into a **child** table, while the backbone that
defines the interface geometry lives in the **parent**. A raw diffusion backbone's own
residue names are **generator artifacts** — very often poly-glycine or poly-alanine. So:

> **`iface_design_resnames` read off a raw rfd3 backbone reports the GENERATOR's residues,
> not a designed sequence.** It will cheerfully return `B12=GLY,B15=GLY,B19=GLY` with
> status `OK`. That is not a binder made of glycine; it is a backbone that has not been
> designed yet. **Check `epitope_seq_source` before quoting any residue name.** If it says
> `structure` and you are looking at an rfd3 backbone, the names are meaningless.

| Where the column lives | What to pass |
| --- | --- |
| Predicted complex — the coordinates *are* the designed sequence | `-i boltz_path`, no `--sequence-column` |
| rfd3 backbone (parent) + designed sequence (child) | `-t table1 -i rfdiffusion3_path --sequence-column atomium_sequence` |
| bindcraft2 complex — coordinates are the design | `-i bindcraft2_path`, no `--sequence-column` |

### The two error contracts on `--sequence-column`

Both are `error:` statuses for that design, with **every** column NA — never a partial
mapping, because a positional mapping off by one misreports the interface composition of
every row with no other symptom.

| Bad input | What happens |
| --- | --- |
| Sequence shorter or longer than the design chain | `error: sequence/structure length mismatch: seq_len=3 from column 'atomium_sequence' but n_design_res_struct=6 amino-acid residues in design chain(s) of hit.pdb. Refusing to map …` — **names both numbers** |
| Sequence contains `/` (a multi-chain string, e.g. `mkcomplex_sequence`) | `error: --sequence-column 'mkcomplex_sequence' holds a MULTI-CHAIN sequence for this design (…): it contains '/'. epitope maps the design chain alone …` — **names the column** |
| Column empty / absent on the row and every ancestor | `error: --sequence-column 'atomium_sequence' is empty for this design …` |

Non-amino-acid residues in a design chain (an ion, a ligand) are **not** part of the
mapping and keep their structural name; `n_design_res_struct` counts amino acids only. A
one-letter code outside the standard 20 is kept **as the letter** (`B12=Z`) rather than
guessed into a three-letter code.

## Lineage: the structure column usually is **not** in the table you run on

Verified in the prosapia source: `core/base_run.py` hands an `update` tool a plain
`read_frame(table)` — **not** `join_lineage()` — so a child table produced by a sequence
designer does **not** carry the parent's `rfdiffusion3_path`. And `filter_ready(df,
input_column)` is documented as defensive: *"an empty frame or a missing `input_column`
yields `df` unchanged"*, so a missing column does not stop submission, it just leaves every
row "ready" with nothing to read.

`epitope` therefore resolves the structure path itself, **row first, then up the lineage**
(`parent_name` / `parent_table`), the same way `usalign` resolves `--col-b`. `-i
rfdiffusion3_path -t table1` works. `--sequence-column` is resolved the same way, so the
common case — geometry from the parent, identities from this row — is one command.

**And a row that resolves nowhere is loud, not absent.** It is submitted anyway with an
empty path and lands in the table as `error: could not resolve a structure path from
column 'rfdiffusion3_path' for this design — neither on its own row nor on any ancestor`.
It is **never dropped from the manifest**: a run that silently submits fewer designs than
the table holds is this workspace's most expensive quiet failure (`No designs to submit.`
exits 0). If *no* ready design resolves, the whole submit raises instead.

> **One behaviour change this implies.** Because the builder no longer uses `ctx.ready`'s
> input-column filter, a row whose structure cell is blank is now submitted and collects as
> `error: could not resolve …` where it previously collected as `missing`. Both are non-`OK`,
> so any filter testing `epitope_status == "OK"` is unaffected — but if you run `epitope`
> on a table where the predictor has not finished, you will now see `error:` rows for the
> designs that have no structure yet rather than blanks. Re-running after the predictor
> finishes fixes them (they are not `OK`, so the resume skip does not protect them).

## Invocation

```bash
# On a predicted complex (target A, binder B), after the target-landed gate:
sapia run epitope <run_dir> -t table1 -i boltz_path \
    --design-chains B --target-chains A \
    --hotspots A96,A99,A101,A155 -l pred
sapia collect epitope <run_dir> -t table1 -l pred

# On the rfd3 backbone it was designed as, onto the same table, under its own label:
sapia run epitope <run_dir> -t table0 -i rfdiffusion3_path \
    --design-chains B --target-chains A --hotspots A96,A99,A101,A155 -l backbone
sapia collect epitope <run_dir> -t table0 -l backbone

# On a bindcraft2 complex (the binder chain letter is in bindcraft2_binder_chain):
sapia run epitope <run_dir> -t table1 -i bindcraft2_path \
    --design-chains B --target-chains A --hotspots A96-99,A155

# On the SEQUENCE-DESIGN generation: geometry from the PARENT's backbone
# (resolved by lineage), residue identities from THIS table's designed sequence.
# This is the invocation that makes iface_design_resnames mean anything.
sapia run epitope <run_dir> -t table1 -i rfdiffusion3_path \
    --sequence-column atomium_sequence \
    --design-chains B --target-chains A \
    --hotspots A96,A99,A101,A155 -l bb
sapia collect epitope <run_dir> -t table1 -l bb
```

| Flag | Default | Notes |
| --- | --- | --- |
| `-i/--input-column` | **effectively required** | Sentinel default `"not applicable"` (the cms/chainsel/ssprofile convention): the builder **raises** unless you name one. The file must hold **both** sides — the binder *and* the target. **The column does not have to be in the table you run on**: it is resolved row-first and then up the lineage, so `-i rfdiffusion3_path -t table1` reads the parent's backbone (see below). Reads `.pdb`, `.cif`, `.cif.gz`; CIF is staged to PDB up front via `ensure_pdb`, cached under `<run_dir>/.cif_to_pdb/`. |
| `--design-chains` | **required** | The binder chain(s), chain mini-language (`,` separates, `:` is an inclusive letter range): `B`, `A,B`, `A:D`. Contacts are counted from these chains' heavy atoms. |
| `--target-chains` | **required** | The target chain(s), same mini-language. Defines both the contact partner and the residue universe the epitope is reported over. Must be disjoint from `--design-chains` (checked at submit). |
| `--hotspots` | **required** | The epitope you chose, chain-prefixed and comma-joined — the same syntax `bindcraft2` takes: `A96,A99,A101,A155`, ranges `A96-99` (inclusive, expanded at submit), insertion codes `A96A`, and `{expr}` islands resolved per design (`A{ep_start}-{ep_end}`). Every chain named must be in `--target-chains`. Duplicates, backwards ranges, cross-chain ranges and chainless residues all **kill the submit**. |
| `--contact-cutoff` | `5.0` | Heavy-atom distance (Å) for a contact. A contact test, not a surface calculation. Recorded as a column, because a recall is only comparable with another recall at the same cutoff. |
| `--designs-per-task` | `20` | Designs per task (batched like `ssprofile`/`cms`). The work is a distance matrix, well under a second per design. |
| `--sequence-column` | `None` → identities from the structure | **Added 2026-09-30.** Table column holding the **design** chain's sequence as a plain one-chain, one-letter string (`atomium_sequence`, `proteinmpnn_sequence`). Resolved up the lineage like `-i`. Changes only *which residue names* `iface_design_resnames` reports; the contact geometry, and every pre-existing column, are untouched. Omit it and the tool behaves exactly as it did before. |
| `-l/--dir-label` | `""` | **In practice required** once you score more than one structure column or more than one epitope. See the traps. |

## Columns collected (`epitope_` prefix, or `epitope_<label>_` with `-l`)

**Did it land on the epitope I chose**

| Column | Meaning |
| --- | --- |
| `_hotspot_recall` | Contacted hotspots / listed hotspots, `0.0`–`1.0`. **The filter column.** `0.0` is a real value (bound elsewhere); NA means nothing was measured. |
| `_hotspot_hits` | The hotspots contacted, comma-joined, or `none`. |
| `_n_hotspots` | How many were listed — the denominator. Check it matches the flag you wrote. |
| `_min_dist_hotspot` | Closest heavy-atom approach to **any** listed hotspot, Å. The graded signal when recall is 0: ~6 Å is a near miss you might rescue, 30 Å is the other side of the target. |

**The trust column — read it first**

| Column | Meaning |
| --- | --- |
| `_hotspot_resnames` | The residue **name** found at each listed position, in the order listed: `A96=TRP,A99=GLU,A101=SER`. If these are not the residues you chose, the structure was renumbered and every other column here answers a different question. |

**Which epitope it actually used**

| Column | Meaning |
| --- | --- |
| `_n_iface_target_res` | Target residues with ≥ 1 contact. `0` is a real answer; NA is not. |
| `_iface_target_resnums` | Which ones (`A96,A97,A98,…`), or `none`. **This is the epitope the binder really used** — compare two designs' lists to see a confident binder that found a different site. |
| `_n_contacts` | Total heavy-atom contact pairs across the interface. A crude size proxy only — for area use `cms`/`pyrosetta`. |

**How CENTRAL the listed hotspots are to that interface** *(added 2026-10-02)* — the same
matrix, the same cutoff, the same single pass, restricted to the residues you listed.
`hotspot_recall` is **binary per residue**: it says *touched*. These say *how much of the
binding is there*.

| Column | Meaning |
| --- | --- |
| `_hotspot_n_contacts` | Contact atom pairs between the design chain(s) and the **listed hotspots only**. Same cutoff and same pair counting as `_n_contacts`, over fewer residues, so `_hotspot_n_contacts <= _n_contacts` always holds. `0` is a real answer. |
| `_hotspot_contact_frac` | `_hotspot_n_contacts / _n_contacts`, `0.0`–`1.0`: **the share of this design's interface carried by the listed residues.** The filter column when the question is *grazed it* vs *built around it*. **NA when `_n_contacts` is 0** — see below. |
| `_hotspot_contacts` | Per-hotspot pair counts **in the order you listed the hotspots**: `A24=0,A36=3,A99=21`. Formatted like `_hotspot_resnames` and index-aligned with it, so the two read against each other residue by residue. An untouched hotspot is an explicit `0`, never an omission. |

**Read `_hotspot_contact_frac` as a SHARE, not an area.** It is a fraction of *this design's
own* interface, so an interface of 300 contact pairs and one of 80 can both sit at `0.15` —
the big one has 45 pairs on the hotspots, the small one 12, and the column does not
distinguish them. Pair it with `_n_contacts` whenever absolute footprint matters, and note
the corollary: the share can rise either because the hotspot got more buried **or** because
the rest of the interface shrank.

**NA is never 0, and the pair is not symmetric.** With a zero interface
(`_n_contacts == 0`) the share is `0/0` — undefined — so `_hotspot_contact_frac` is **NA**
while `_hotspot_n_contacts` is a real, measured `0` and `_hotspot_contacts` is
`A96=0,A99=0`. A `0.0` there would read as *the binder built its interface somewhere else*,
which is a different fact. Every `error:` row gives **NA for all three**, exactly like every
pre-existing column.

**Which of the binder's own residues do the binding** *(added 2026-09-30)*

| Column | Meaning |
| --- | --- |
| `_n_iface_design_res` | Design-chain residues with ≥ 1 contact. `0` is a real answer; NA is not. |
| `_iface_design_resnums` | Which ones (`B12,B15,B19,…`), or `none`. Chain-prefixed like `iface_target_resnums`, but in **ascending residue order** (the target list stays in structure order — unchanged). |
| `_iface_design_resnames` | **The binder-side trust column.** The same residues with the identity at each: `B12=HIS,B15=TYR,B19=GLU` — formatted exactly like `hotspot_resnames`, index-aligned with `_iface_design_resnums`. Eyeball it. A frame shift between a designed sequence and the backbone it was mapped onto is obvious here and invisible in every other column. |

**Where the design identities came from** *(added 2026-09-30 — read before believing the names above)*

| Column | Meaning |
| --- | --- |
| `_seq_source` | `structure` (from the coordinates — the default, and the original behaviour) or the `--sequence-column` the identities were taken from. |
| `_seq_len` | Length of the sequence actually used. |
| `_n_design_res_struct` | Amino-acid residues in the design chain(s) **of the structure**. A disagreement with `_seq_len` is an `error:`, never a silent remap. |

**The echo of what was asked**

| Column | Meaning |
| --- | --- |
| `_design_chains` / `_target_chains` | The chains actually scored. |
| `_contact_cutoff` | The Å cutoff used. |
| `_path` | **Per-target-residue TSV**: `chain, resnum, resname, n_contacts, min_dist, is_hotspot` — **every** target residue, contacted or not. The per-design record in which a numbering mismatch or a second binding site is visible at a glance. |
| `_status` | `OK`, `error: …`, or `missing`. The only reliable success signal. |

On disk: `<run_dir>/<table>/epitope[_<label>]/<name>.tsv` and `<name>_per_residue.tsv`.

**The 2026-09-30 and 2026-10-02 additions are purely additive.** No pre-existing column was renamed,
removed, retyped or reformatted; no flag default changed; the per-residue TSV schema
(`chain, resnum, resname, n_contacts, min_dist, is_hotspot` — **target** residues) is
untouched; and with `--sequence-column` omitted every pre-existing number is what it was.
Tables collected before that date (`epitope_bb_*`, `epitope_bbnterm_*`, `epitope_pred_*`)
simply have the newer columns empty — a per-design TSV lacking them collects them as NA,
never as 0. To backfill, re-run and re-collect with `--force` under the same `-l` label.
Verified by dumping the pre-edit worker's output for 13 synthetic scenarios (hits, misses,
a zero interface, every error contract, the sequence-mapping paths, the legacy 3-field task
row) and comparing the post-edit output cell by cell: every pre-existing cell is the
identical raw string, every per-residue TSV is byte-identical, and the only header change is
the three appended names.

## The filter — and a live trap about `-f`

The predicate this tool was built for:

```
epitope_status == "OK" and epitope_hotspot_recall >= 0.5
```

and its inverse, the designs that ignored the epitope entirely:

```
epitope_status == "OK" and epitope_hotspot_recall == 0.0
```

and, when recall is binary and discriminates nothing — measured on this campaign's 400
backbones, target histidine `A99` is contacted by **389 of 400**, so a "touches A99" gate
removes nothing while "zero contact" leaves 11 — the **weight** predicate instead. Tolerate
a graze, reject an interface built around the residue:

```
epitope_his_status == "OK" and epitope_his_hotspot_contact_frac < 0.05
```

and, on the other leaf, rank by how much of the interface sits on the chosen epitope:

```
epitope_bb_status == "OK" and epitope_bb_hotspot_contact_frac >= 0.25
```

Take the threshold from the **measured distribution** on your own table, not from this page:
the share depends on the cutoff, on how many residues you listed, and on how big the
interfaces are in that batch.

**`sapia run -f` does not take an expression.** Verified on the prosapia in this workspace
(and in every cached revision of `dev`): `-f/--filter` is **a path to a Python module
defining `apply_filter(df) -> df`**. Passing the expression string fails with
`ImportError: Could not load module from epitope_status == "OK" …`. The CLAUDE.md line and
several sibling tool skills show `-f '<expression>'`; that form is **not** supported by the
library today. Write the module instead — it is three lines, and it is reusable:

```python
# filters/on_epitope.py
def apply_filter(df):
    return df.query('epitope_status == "OK" and epitope_hotspot_recall >= 0.5')
```

```bash
sapia run proteinmpnn <run_dir> -t table1 -i rfdiffusion3_path -f filters/on_epitope.py
```

The expression itself is a valid `DataFrame.query`, verified against a collected table: it
kept the on-epitope design, the inverse kept the bound-elsewhere design, and the
numbering-error design fell out of **both** — which is exactly what the error contract is
for. (Under Modal the module path must be readable inside the workstation, i.e. on the
runs Volume or in the working tree that was shipped.)

## Traps

**Read it after the target-landed gate, never before.** On a predicted complex this tool
measures where the binder sits *relative to the predicted target*. If the prediction moved
the target, the recall is precise and meaningless. Order: target landed (`usalign`) →
epitope → interface quality (`cms`, `pyrosetta`).

**The structure column must contain both sides.** Pointing `-i` at a `chainsel_path` that
holds only the binder gives `error: target chain(s) A absent`. That is the loud failure,
but note the quiet mirror image: naming the wrong chain as `--design-chains` (the target,
say) scores target-against-target happily. Guard with `epitope_design_chains` and with
`n_hotspots` / `hotspot_resnames`.

**Chain letters differ between generators.** rfd3 may call the binder `B` where a predictor
calls it `E`, and bindcraft2 records its own letter in `bindcraft2_binder_chain` (never
read BC2's internal `binder_chain` *setting*). An absent chain is an error, so you will
find out — but only after a submit.

**`-l/--dir-label` is effectively required.** The leaf `epitope` names both the output dir
and every column, so scoring `rfdiffusion3_path` and then `boltz_path` on the same table
with no label overwrites the first run's columns. Use `-l backbone` / `-l pred`, **pass the
same `-l` to `collect`**, and remember the filter must then use the labelled prefix
(`epitope_pred_hotspot_recall`).

**`hotspot_contact_frac` is weighted by ATOM COUNT, not by energy or by burial.** It
counts atom pairs, so a large side chain inserted into the binder (ARG, TRP, a histidine
wedged in a pocket) contributes more pairs than a small one at the same depth, and a
hotspot whose own side chain interpenetrates the binder inflates its own count. Treat it as
a ranking signal over a batch scored at one cutoff, not as a physical quantity: for how
much of the residue is actually buried use `cms_path` (per-residue CMS) or `pyrosetta`
(`if_dSASA`). It is also **insensitive to which residues you did not list** — listing one
hotspot instead of four changes the numerator but not the denominator, so shares from two
different `--hotspots` lists are not comparable.

**The cutoff moves the answer.** At a 4 Å gap, `--contact-cutoff 3.5` gives recall 0.0 and
`8.0` gives recall 1.0 on the same structure (measured in the test). 5.0 is a generous
heavy-atom contact. Keep one cutoff for a whole campaign or the recalls are not comparable
across tables — and `epitope_contact_cutoff` is in the table so you can check.

**`Submitting N designs` counts TASKS, not designs.** 60 designs at the default
`--designs-per-task 20` print `Submitting 3 designs`. Count real results with
`epitope_status`.

**Failures still exit 0.** Errors are recorded as data, so `.exit` files that are all `0`
do not mean every design succeeded. Read the status column. A worker crash writes
`error: worker crashed` for every design in that chunk.

**Rows drop out of the manifest silently.** A design whose input file does not exist is
printed as `MISSING … (skipping)` and left out; it collects as `missing`.

**Ligands and modified residues in a named chain count.** Contacts use every non-water
residue of the named chains, so a glycan or cofactor modelled inside chain A is part of the
target (and can legitimately be a hotspot). Hydrogens and waters are excluded.

**Hotspot ranges are integers only.** `A96-99` is fine; an insertion code inside a range
(`A96A-99`) is not, and raises at submit. Single residues may carry one (`A96A`).

## Verified (local, 2026-09-30)

`uv run python tools/epitope/test_epitope.py` — synthetic complexes with a hand-countable
answer (a target laid out 4 Å apart along x, a binder parked a known distance above a known
stretch). All checks pass, including:

- binder over target residues 96–101 → `hotspot_recall 1.0`, `iface_target_resnums
  A96,A97,A98,A99,A100,A101`, `min_dist_hotspot 4.0`, `hotspot_resnames A96=TRP,A99=GLU,
  A101=SER`, per-residue file covering all 21 target residues;
- binder elsewhere → status **OK** with `hotspot_recall 0.0`, `hotspot_hits none`,
  `min_dist_hotspot 28.3`, and the collector keeping that `0.0` as `0.0`;
- binder touching nothing → `n_iface_target_res 0`, `iface_target_resnums none`;
- `A500` on a 90–110 target → **`error:`** naming the residue, blaming the numbering and
  reporting the span, with every column NA through the collector;
- a renumbered target scoring fine while `hotspot_resnames` exposes what was really scored;
- hotspot on a non-target chain, absent design chain, absent target chain, missing file →
  all `error:` with NA columns;
- `--contact-cutoff` 3.5 vs 8.0 flipping recall on one structure;
- submit-side: `{expr}` resolution, range expansion, batching (5 designs at 2 per task → 3
  tasks), `gpus_per_task = 0`, and the five refusals that kill the submit.

**The 2026-09-30 additions** add seven cases to the same file (19 in total, all passing):

- **design side**: all 6 binder residues contacting → `n_iface_design_res 6`,
  `iface_design_resnums B1,B2,B3,B4,B5,B6`, `iface_design_resnames
  B1=GLY,…,B6=GLY` (the backbone really is poly-glycine — that is the warning made
  visible), round-tripping through the collector as real columns;
- **design side, nothing touched** → `0` / `none` / `none`, kept as `0` by the collector,
  while an `error:` row gives NA for the new columns too;
- **`--sequence-column`** → `iface_design_resnames B1=MET,B2=HIS,B3=LYS,B4=TYR,B5=TRP,
  B6=ASP` from the sequence while the poly-GLY coordinates still supply the geometry, and
  `hotspot_recall` is unchanged at `1.0`;
- **both error contracts** → `seq_len=3` vs `n_design_res_struct=6` named in the message
  with every column NA; a `/` string refused by name; an empty column refused by name;
- **lineage** → a structure column absent from the frame resolved from the ancestor, the
  sequence column likewise, and the row's own value winning when both exist;
- **unresolvable row** → present in the manifest with an empty path, collected as
  `error: could not resolve …` naming the column, never dropped;
- **backward compatibility** → every pre-existing column byte-identical with
  `--sequence-column` omitted, a 3-field (pre-edit) sub-manifest row still scoring, and the
  per-residue schema unchanged.

Also verified end to end **locally** (`sapia run` → real manifest + `.meta.json` → the task
`.sh` under the real prelude → `sapia collect`) on a fabricated 3-row `table0`, including a
`.cif` input staged through `ensure_pdb`, a `-l probe` labelled second run, and the
collected table showing `1.0` / `0.0` / `error:` on the three designs.

**Not yet run on Modal.** The image is `debian_slim + gemmi + numpy` and builds in seconds,
but treat the first `-e modal` run as a shakedown: score 2 designs and read
`epitope_status`, `epitope_hotspot_resnames` and `epitope_n_hotspots` before scaling.
**Not available on vib** — it would need a `SAPIA_ACTIVATE_EPITOPE` entry providing a
Python with gemmi + numpy.
