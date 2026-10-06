#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=cms

# Score one batch of designs per array task: contact molecular surface + shape
# complementarity via cms-cuda. The manifest line points at a sub-manifest
# (name<TAB>structure per design); the worker writes <name>.tsv and
# <name>_per_residue.tsv per design for collect_cms.py. The worker needs cms_cuda
# (+ CuPy for the GPU path), so it runs under $PIPELINE_PYTHON (default `python`).

set -euo pipefail

# Shared scaffolding: sets MANIFEST/OUT_DIR/SAPIA_TASK_ID/SAPIA_LINE.
source "${SAPIA_PRELUDE:?}"

# Site-specific activation (required) — see docs/configuration.md.
sapia_activate SAPIA_ACTIVATE_CMS

PY=${PIPELINE_PYTHON:-python}

TASK_FILE=$(echo "$SAPIA_LINE" | cut -f1)
BINDER_CHAINS=$(echo "$SAPIA_LINE" | cut -f2)
TARGET_CHAINS=$(echo "$SAPIA_LINE" | cut -f3)
EXCLUDE=$(echo "$SAPIA_LINE" | cut -f4)
SC_MODE=$(echo "$SAPIA_LINE" | cut -f5)
MAX_MODE=$(echo "$SAPIA_LINE" | cut -f6)
DEVICE=$(echo "$SAPIA_LINE" | cut -f7)

# store_true flags as `if` blocks: `[[ ... ]] && ...` returns 1 under `set -e` and
# would kill the task whenever the flag is off.
EXTRA=()
if [[ "$SC_MODE" == "nosc" ]]; then
    EXTRA+=(--no-sc)
fi
if [[ "$MAX_MODE" == "max" ]]; then
    EXTRA+=(--max-cms)
fi

echo "[$(date +%T)] task $SAPIA_TASK_ID: cms on $(wc -l <"$TASK_FILE") designs ($DEVICE)"

# If the worker itself crashes (before it can record error-as-data), write a fallback
# error TSV for every design of this task that has none, so collect still sees them.
WORKER_RC=0
"$PY" "${SAPIA_TOOL_DIR:?}/cms_worker.py" \
        --task-file "$TASK_FILE" \
        --binder-chains "$BINDER_CHAINS" \
        --target-chains "$TARGET_CHAINS" \
        --exclude-resnames "$EXCLUDE" \
        --device "$DEVICE" \
        ${EXTRA[@]+"${EXTRA[@]}"} \
        --out-dir "$OUT_DIR" || WORKER_RC=$?

if [ "$WORKER_RC" -ne 0 ]; then
    while IFS=$'\t' read -r NAME _SRC || [ -n "$NAME" ]; do
        [ -n "$NAME" ] || continue
        if [ ! -f "$OUT_DIR/${NAME}.tsv" ]; then
            printf 'name\tstatus\tpath\n%s\terror: worker crashed\t\n' "$NAME" \
                >"$OUT_DIR/${NAME}.tsv"
        fi
    done <"$TASK_FILE"
fi

exit "$WORKER_RC"
