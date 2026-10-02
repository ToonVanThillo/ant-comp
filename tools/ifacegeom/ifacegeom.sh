#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ifacegeom

# Measure one batch of binder/target complexes per array task: interface residue
# lists on both sides plus terminus geometry. The manifest line points at a
# sub-manifest (name<TAB>structure per design); the worker writes <name>.tsv and
# <name>_per_residue.tsv per design for collect_ifacegeom.py. CPU-only. The worker
# needs gemmi + numpy, so it runs under $PIPELINE_PYTHON (defaulting to `python`).

set -euo pipefail

# Shared scaffolding: sets MANIFEST/OUT_DIR/SAPIA_TASK_ID/SAPIA_LINE.
source "${SAPIA_PRELUDE:?}"

# Site-specific activation (required) — see docs/configuration.md.
sapia_activate SAPIA_ACTIVATE_IFACEGEOM

# The interpreter defaults to `python` (made right by the activation hook, or already
# on PATH — it must carry gemmi + numpy). Set PIPELINE_PYTHON to point straight at a
# specific interpreter.
PY=${PIPELINE_PYTHON:-python}

TASK_FILE=$(echo "$SAPIA_LINE" | cut -f1)
BINDER_CHAINS=$(echo "$SAPIA_LINE" | cut -f2)
TARGET_CHAINS=$(echo "$SAPIA_LINE" | cut -f3)
METHOD=$(echo "$SAPIA_LINE" | cut -f4)
CONTACT_CUTOFF=$(echo "$SAPIA_LINE" | cut -f5)
CB_DIST_CUT=$(echo "$SAPIA_LINE" | cut -f6)
VECTOR_DIST_CUT=$(echo "$SAPIA_LINE" | cut -f7)
VECTOR_ANGLE_CUT=$(echo "$SAPIA_LINE" | cut -f8)

echo "[$(date +%T)] task $SAPIA_TASK_ID: ifacegeom on $(wc -l <"$TASK_FILE") designs ($METHOD)"

# If the worker itself crashes (before it can record error-as-data), write a fallback
# error TSV for every design of this task that has none, so collect still sees them.
if ! "$PY" "${SAPIA_TOOL_DIR:?}/ifacegeom_worker.py" \
        --task-file "$TASK_FILE" \
        --binder-chains "$BINDER_CHAINS" \
        --target-chains "$TARGET_CHAINS" \
        --method "$METHOD" \
        --contact-cutoff "$CONTACT_CUTOFF" \
        --cb-dist-cut "$CB_DIST_CUT" \
        --vector-dist-cut "$VECTOR_DIST_CUT" \
        --vector-angle-cut "$VECTOR_ANGLE_CUT" \
        --out-dir "$OUT_DIR"; then
    while IFS=$'\t' read -r NAME _SRC || [ -n "$NAME" ]; do
        [ -n "$NAME" ] || continue
        if [ ! -f "$OUT_DIR/${NAME}.tsv" ]; then
            printf 'name\tstatus\tpath\n%s\terror: worker crashed\t\n' "$NAME" \
                >"$OUT_DIR/${NAME}.tsv"
        fi
    done <"$TASK_FILE"
fi
