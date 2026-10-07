#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=linkpath

# Measure, for one batch of designs per array task, the shortest route between two
# chain termini that stays out of the protein, and write per-design files
# (<name>.tsv, <name>_path.pdb) for collect_linkpath.py to merge back. The manifest
# line points at a sub-manifest (name<TAB>structure<TAB>from_res<TAB>to_res per
# design) plus the run's flags. CPU-only; the manifest builder sets gpus_per_task =
# 0. The worker needs gemmi + numpy, so it runs under $PIPELINE_PYTHON (defaulting
# to `python`).

set -euo pipefail

# Shared scaffolding: sets MANIFEST/OUT_DIR/SAPIA_TASK_ID/SAPIA_LINE.
source "${SAPIA_PRELUDE:?}"

# Site-specific activation (required) — see docs/configuration.md.
sapia_activate SAPIA_ACTIVATE_LINKPATH

# The interpreter defaults to `python` (made right by the activation hook, or already
# on PATH — it must carry gemmi + numpy). Set PIPELINE_PYTHON to point straight at a
# specific interpreter.
PY=${PIPELINE_PYTHON:-python}

# --- manifest-line guard -----------------------------------------------------
# Copied from dimerfit.sh / rpxdock.sh, which earned it the hard way. The prelude
# does `SAPIA_LINE="$(sed -n "${SAPIA_TASK_ID}p" "$MANIFEST")"`, and `sed -n Np` on
# a file that does not (yet) show N lines EXITS 0 WITH EMPTY OUTPUT. `set -e` does
# not trip, and empty fields reach the worker as plausible-looking arguments.
#
# Measured 2026-10-02, run 20261002_143419_dimer_phase2: 18 of 37 rpxdock tasks hit
# this at once. The manifest on the Volume was provably intact afterwards, so this
# is a READ-side visibility failure in the task container, not a corrupt writer.
#
# So: retry briefly in case the view settles, then FAIL LOUDLY. A wrong-but-
# plausible run is far worse than a dead one.
sapia_read_manifest_line() {
    local attempt line nlines
    for attempt in 1 2 3 4 5; do
        line=$(sed -n "${SAPIA_TASK_ID}p" "$MANIFEST" || true)
        if [ -n "$line" ]; then
            printf '%s' "$line"
            return 0
        fi
        nlines=$(wc -l <"$MANIFEST" 2>/dev/null || echo "?")
        echo "[$(date +%T)] WARNING: manifest line $SAPIA_TASK_ID empty on attempt" \
             "$attempt/5 (manifest '$MANIFEST' shows $nlines lines); retrying" >&2
        sleep $((attempt * 5))
    done
    return 1
}

if [ -z "${SAPIA_LINE:-}" ]; then
    echo "[$(date +%T)] manifest line $SAPIA_TASK_ID came back EMPTY; re-reading" >&2
    SAPIA_LINE=$(sapia_read_manifest_line) || {
        echo "FATAL: task $SAPIA_TASK_ID could not read line $SAPIA_TASK_ID of" \
             "'$MANIFEST' after 5 attempts. The manifest is present but this" \
             "container does not see line $SAPIA_TASK_ID (stale Volume view)." \
             "Refusing to run with empty inputs." >&2
        exit 1
    }
fi

# NOTE: manifest-field locals are prefixed. Bash pre-sets names like GROUPS, UID
# and PIPESTATUS; assigning a command substitution to one fails with rc=1 and,
# under `set -e`, kills the task with EMPTY .out and .err.
LP_TASK_FILE=$(echo "$SAPIA_LINE" | cut -f1)
LP_N_DESIGNS=$(echo "$SAPIA_LINE" | cut -f2)
LP_FROM_CHAIN=$(echo "$SAPIA_LINE" | cut -f3)
LP_TO_CHAIN=$(echo "$SAPIA_LINE" | cut -f4)
LP_FROM_END=$(echo "$SAPIA_LINE" | cut -f5)
LP_TO_END=$(echo "$SAPIA_LINE" | cut -f6)
LP_OBSTACLES=$(echo "$SAPIA_LINE" | cut -f7)
LP_INCLUDE_H=$(echo "$SAPIA_LINE" | cut -f8)
LP_PROTEIN_ONLY=$(echo "$SAPIA_LINE" | cut -f9)
LP_SPACING=$(echo "$SAPIA_LINE" | cut -f10)
LP_PROBE=$(echo "$SAPIA_LINE" | cut -f11)
LP_PAD=$(echo "$SAPIA_LINE" | cut -f12)
LP_CARVE=$(echo "$SAPIA_LINE" | cut -f13)
LP_TAUT_RISE=$(echo "$SAPIA_LINE" | cut -f14)
LP_RELAXED_RISE=$(echo "$SAPIA_LINE" | cut -f15)
LP_BOTH_DIR=$(echo "$SAPIA_LINE" | cut -f16)
LP_RES_EST=$(echo "$SAPIA_LINE" | cut -f17)
LP_MODEL=$(echo "$SAPIA_LINE" | cut -f18)

# Field COUNT first: `cut -f2` on a line with NO TABS returns the whole line, not
# an empty string, so a one-field line would otherwise sail past the empty checks
# below with every variable holding the same text.
LP_NFIELDS=$(printf '%s' "$SAPIA_LINE" | awk -F'\t' '{print NF}')
if [ "$LP_NFIELDS" -ne 18 ]; then
    echo "FATAL: task $SAPIA_TASK_ID read a manifest line with $LP_NFIELDS" \
         "field(s), expected 18 (task_file, n_designs, from_chain, to_chain," \
         "from_end, to_end, obstacle_chains, include_h, protein_only, spacing," \
         "probe, pad, carve, taut_rise, relaxed_rise, both_directions," \
         "residue_estimate, model). line='$SAPIA_LINE'" >&2
    exit 1
fi

# Every field must be present. Catches a short line and a half-visible Volume as
# separate, named failures rather than one confusing one.
for _pair in "task_file=$LP_TASK_FILE" "n_designs=$LP_N_DESIGNS" \
             "from_chain=$LP_FROM_CHAIN" "to_chain=$LP_TO_CHAIN" \
             "from_end=$LP_FROM_END" "to_end=$LP_TO_END" \
             "obstacle_chains=$LP_OBSTACLES" "include_h=$LP_INCLUDE_H" \
             "protein_only=$LP_PROTEIN_ONLY" "spacing=$LP_SPACING" \
             "probe=$LP_PROBE" "pad=$LP_PAD" "carve=$LP_CARVE" \
             "taut_rise=$LP_TAUT_RISE" "relaxed_rise=$LP_RELAXED_RISE" \
             "both_directions=$LP_BOTH_DIR" "residue_estimate=$LP_RES_EST" \
             "model=$LP_MODEL"; do
    if [ -z "${_pair#*=}" ]; then
        echo "FATAL: task $SAPIA_TASK_ID read a malformed manifest line:" \
             "'${_pair%%=*}' is empty. line='$SAPIA_LINE'" >&2
        exit 1
    fi
done

# The sub-manifest must be visible AND non-empty. The worker additionally checks
# that it holds exactly $LP_N_DESIGNS lines, so a PARTIALLY visible task file is a
# named failure rather than a task that quietly measures fewer designs.
if [ ! -s "$LP_TASK_FILE" ]; then
    echo "FATAL: task $SAPIA_TASK_ID: task file '$LP_TASK_FILE' is missing or" \
         "empty. The manifest line parsed, but this container cannot see the" \
         "sub-manifest." >&2
    exit 1
fi
# --- end manifest-line guard -------------------------------------------------

# The two boolean-ish flags are passed as 0/1 columns and turned into real argv
# switches here, so the manifest never has to carry an empty field.
LP_FLAGS=()
[ "$LP_INCLUDE_H" = "1" ] && LP_FLAGS+=(--include-h)
[ "$LP_PROTEIN_ONLY" = "1" ] && LP_FLAGS+=(--protein-only)
[ "$LP_BOTH_DIR" = "1" ] || LP_FLAGS+=(--no-both-directions)
[ "$LP_RES_EST" = "1" ] || LP_FLAGS+=(--no-residue-estimate)

echo "[$(date +%T)] task $SAPIA_TASK_ID: linkpath on $LP_N_DESIGNS design(s)" \
     "($LP_FROM_CHAIN:$LP_FROM_END -> $LP_TO_CHAIN:$LP_TO_END, obstacles" \
     "$LP_OBSTACLES, probe $LP_PROBE, spacing $LP_SPACING)"

# The worker records per-design errors as data (status 'error: ...') and still exits
# 0, so one bad structure never fails the batch. If the worker itself crashes before
# it can do even that, write a fallback error TSV for every design of this task that
# has none, so collect still sees them.
if ! "$PY" "${SAPIA_TOOL_DIR:?}/linkpath_worker.py" \
        --task-file "$LP_TASK_FILE" \
        --n-designs "$LP_N_DESIGNS" \
        --from-chain "$LP_FROM_CHAIN" \
        --to-chain "$LP_TO_CHAIN" \
        --from-end "$LP_FROM_END" \
        --to-end "$LP_TO_END" \
        --obstacle-chains "$LP_OBSTACLES" \
        --spacing "$LP_SPACING" \
        --probe "$LP_PROBE" \
        --pad "$LP_PAD" \
        --carve "$LP_CARVE" \
        --taut-rise "$LP_TAUT_RISE" \
        --relaxed-rise "$LP_RELAXED_RISE" \
        --model "$LP_MODEL" \
        "${LP_FLAGS[@]+"${LP_FLAGS[@]}"}" \
        --out-dir "$OUT_DIR"; then
    while IFS=$'\t' read -r LP_NAME _REST || [ -n "$LP_NAME" ]; do
        [ -n "$LP_NAME" ] || continue
        if [ ! -f "$OUT_DIR/${LP_NAME}.tsv" ]; then
            printf 'name\tstatus\tpath\n%s\terror: worker crashed\t\n' \
                "$LP_NAME" >"$OUT_DIR/${LP_NAME}.tsv"
        fi
    done <"$LP_TASK_FILE"
    exit 1
fi
