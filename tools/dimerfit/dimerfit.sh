#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dimerfit

# Place one batch of C2 docks back into the binder/target frame per array task, and
# write per-design files (<name>.tsv, <name>_dimer.pdb, <name>_complex.pdb) for
# collect_dimerfit.py to merge back. The manifest line points at a sub-manifest
# (name<TAB>dock<TAB>ref<TAB>epitope<TAB>target_epitope per design) plus the run's
# flags. CPU-only; the manifest builder sets gpus_per_task = 0. The worker needs
# gemmi + numpy, so it runs under $PIPELINE_PYTHON (defaulting to `python`).

set -euo pipefail

# Shared scaffolding: sets MANIFEST/OUT_DIR/SAPIA_TASK_ID/SAPIA_LINE.
source "${SAPIA_PRELUDE:?}"

# Site-specific activation (required) — see docs/configuration.md.
sapia_activate SAPIA_ACTIVATE_DIMERFIT

# The interpreter defaults to `python` (made right by the activation hook, or already
# on PATH — it must carry gemmi + numpy). Set PIPELINE_PYTHON to point straight at a
# specific interpreter.
PY=${PIPELINE_PYTHON:-python}

# --- manifest-line guard -----------------------------------------------------
# Copied from rpxdock.sh, which earned it the hard way. The prelude does
# `SAPIA_LINE="$(sed -n "${SAPIA_TASK_ID}p" "$MANIFEST")"`, and `sed -n Np` on a
# file that does not (yet) show N lines EXITS 0 WITH EMPTY OUTPUT. `set -e` does
# not trip, and empty fields reach the worker as plausible-looking arguments.
#
# Measured 2026-10-02, run 20261002_143419_dimer_phase2: 18 of 37 rpxdock tasks
# hit this at once. The manifest on the Volume was provably intact afterwards, so
# this is a READ-side visibility failure in the task container, not a corrupt
# writer. prosapia's publish_manifest already uses batch_upload against exactly
# this hazard and it was not sufficient; one task re-read the file ~2 minutes
# later and STILL saw nothing, so the stale view can outlive a short backoff.
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
DF_TASK_FILE=$(echo "$SAPIA_LINE" | cut -f1)
DF_N_DESIGNS=$(echo "$SAPIA_LINE" | cut -f2)
DF_DOCK_CHAINS=$(echo "$SAPIA_LINE" | cut -f3)
DF_BINDER_CHAIN=$(echo "$SAPIA_LINE" | cut -f4)
DF_TARGET_CHAINS=$(echo "$SAPIA_LINE" | cut -f5)
DF_RESNUM_MATCH=$(echo "$SAPIA_LINE" | cut -f6)
DF_CLASH_CUTOFF=$(echo "$SAPIA_LINE" | cut -f7)
DF_OCCLUSION_CUTOFF=$(echo "$SAPIA_LINE" | cut -f8)
DF_DIMER_CUTOFF=$(echo "$SAPIA_LINE" | cut -f9)

# Field COUNT first: `cut -f2` on a line with NO TABS returns the whole line, not
# an empty string, so a one-field line would otherwise sail past the empty checks
# below with every variable holding the same text.
DF_NFIELDS=$(printf '%s' "$SAPIA_LINE" | awk -F'\t' '{print NF}')
if [ "$DF_NFIELDS" -ne 9 ]; then
    echo "FATAL: task $SAPIA_TASK_ID read a manifest line with $DF_NFIELDS" \
         "field(s), expected 9 (task_file, n_designs, dock_chains," \
         "binder_chain_in_ref, target_chains_in_ref, resnum_match, clash_cutoff," \
         "occlusion_cutoff, dimer_contact_cutoff). line='$SAPIA_LINE'" >&2
    exit 1
fi

# Every field must be present. Catches a short line and a half-visible Volume as
# separate, named failures rather than one confusing one.
for _pair in "task_file=$DF_TASK_FILE" "n_designs=$DF_N_DESIGNS" \
             "dock_chains=$DF_DOCK_CHAINS" "binder_chain=$DF_BINDER_CHAIN" \
             "target_chains=$DF_TARGET_CHAINS" "resnum_match=$DF_RESNUM_MATCH" \
             "clash_cutoff=$DF_CLASH_CUTOFF" \
             "occlusion_cutoff=$DF_OCCLUSION_CUTOFF" \
             "dimer_cutoff=$DF_DIMER_CUTOFF"; do
    if [ -z "${_pair#*=}" ]; then
        echo "FATAL: task $SAPIA_TASK_ID read a malformed manifest line:" \
             "'${_pair%%=*}' is empty. line='$SAPIA_LINE'" >&2
        exit 1
    fi
done

# The sub-manifest must be visible AND non-empty. The worker additionally checks
# that it holds exactly $DF_N_DESIGNS lines, so a PARTIALLY visible task file is a
# named failure rather than a task that quietly measures fewer designs.
if [ ! -s "$DF_TASK_FILE" ]; then
    echo "FATAL: task $SAPIA_TASK_ID: task file '$DF_TASK_FILE' is missing or" \
         "empty. The manifest line parsed, but this container cannot see the" \
         "sub-manifest." >&2
    exit 1
fi
# --- end manifest-line guard -------------------------------------------------

echo "[$(date +%T)] task $SAPIA_TASK_ID: dimerfit on $DF_N_DESIGNS design(s)" \
     "(dock chains $DF_DOCK_CHAINS, ref binder $DF_BINDER_CHAIN, ref target" \
     "$DF_TARGET_CHAINS)"

# The worker records per-design errors as data (status 'error: ...') and still exits
# 0, so one bad dock never fails the batch. If the worker itself crashes before it
# can do even that, write a fallback error TSV for every design of this task that has
# none, so collect still sees them.
if ! "$PY" "${SAPIA_TOOL_DIR:?}/dimerfit_worker.py" \
        --task-file "$DF_TASK_FILE" \
        --n-designs "$DF_N_DESIGNS" \
        --dock-chains "$DF_DOCK_CHAINS" \
        --binder-chain-in-ref "$DF_BINDER_CHAIN" \
        --target-chains-in-ref "$DF_TARGET_CHAINS" \
        --resnum-match "$DF_RESNUM_MATCH" \
        --clash-cutoff "$DF_CLASH_CUTOFF" \
        --occlusion-cutoff "$DF_OCCLUSION_CUTOFF" \
        --dimer-contact-cutoff "$DF_DIMER_CUTOFF" \
        --out-dir "$OUT_DIR"; then
    while IFS=$'\t' read -r DF_NAME _REST || [ -n "$DF_NAME" ]; do
        [ -n "$DF_NAME" ] || continue
        if [ ! -f "$OUT_DIR/${DF_NAME}.tsv" ]; then
            printf 'name\tstatus\tpath\tcomplex_path\n%s\terror: worker crashed\t\t\n' \
                "$DF_NAME" >"$OUT_DIR/${DF_NAME}.tsv"
        fi
    done <"$DF_TASK_FILE"
    exit 1
fi
