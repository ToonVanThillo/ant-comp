#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=ringfit

# Score one design's fit into the reference assembly per array task, and write a
# per-design TSV (name, status, aligned_path, metrics) for collect_ringfit.py to
# merge back. CPU-only. The worker needs gemmi + numpy, so it runs under
# $PIPELINE_PYTHON (defaulting to `python`).

set -euo pipefail

# Shared scaffolding: sets MANIFEST/OUT_DIR/SAPIA_TASK_ID/SAPIA_LINE.
source "${SAPIA_PRELUDE:?}"

# Site-specific activation (required) — see docs/configuration.md.
sapia_activate SAPIA_ACTIVATE_RINGFIT

# The interpreter defaults to `python` (made right by the activation hook, or already
# on PATH — it must carry gemmi + numpy). Set PIPELINE_PYTHON to point straight at a
# specific interpreter.
PY=${PIPELINE_PYTHON:-python}

NAME=$(echo "$SAPIA_LINE" | cut -f1)
DESIGN=$(echo "$SAPIA_LINE" | cut -f2)
REF=$(echo "$SAPIA_LINE" | cut -f3)
TARGET_CHAINS=$(echo "$SAPIA_LINE" | cut -f4)
BINDER_CHAINS=$(echo "$SAPIA_LINE" | cut -f5)
REF_TARGET_CHAINS=$(echo "$SAPIA_LINE" | cut -f6)
HOTSPOTS=$(echo "$SAPIA_LINE" | cut -f7)
LIPID_RESNAMES=$(echo "$SAPIA_LINE" | cut -f8)
CLASH_CUTOFF=$(echo "$SAPIA_LINE" | cut -f9)
CONTACT_CUTOFF=$(echo "$SAPIA_LINE" | cut -f10)
HOTSPOTS_NUMBERING=$(echo "$SAPIA_LINE" | cut -f11)
RESNUM_MATCH=$(echo "$SAPIA_LINE" | cut -f12)

RESULT_TSV="$OUT_DIR/${NAME}.tsv"
OUT_PDB="$OUT_DIR/${NAME}_onref.pdb"

echo "[$(date +%T)] task $SAPIA_TASK_ID: ringfit for $NAME"

# The worker lives next to this .sh; SAPIA_TOOL_DIR (exported by the driver) points
# there, independent of the submit cwd. If the worker itself crashes (before it can
# record error-as-data), write a fallback error TSV so collect still sees this design.
# The worker's return code is captured EXPLICITLY and re-raised at the end: with a
# bare `if ! worker; then <fallback>; fi` the script's exit status would be the status
# of the FALLBACK's last command, so a task could exit 1 with every row written
# correctly (or, worse, exit 0 after a genuine crash). The .exit file is the only
# completion signal on Modal, so it has to mean what it says.
WORKER_RC=0
"$PY" "${SAPIA_TOOL_DIR:?}/ringfit_worker.py" \
        --name "$NAME" \
        --design "$DESIGN" \
        --ref "$REF" \
        --target-chains "$TARGET_CHAINS" \
        --binder-chains "$BINDER_CHAINS" \
        --ref-target-chains "$REF_TARGET_CHAINS" \
        --hotspots "$HOTSPOTS" \
        --hotspots-numbering "$HOTSPOTS_NUMBERING" \
        --resnum-match "$RESNUM_MATCH" \
        --lipid-resnames "$LIPID_RESNAMES" \
        --clash-cutoff "$CLASH_CUTOFF" \
        --contact-cutoff "$CONTACT_CUTOFF" \
        --out-pdb "$OUT_PDB" \
        --result-tsv "$RESULT_TSV" || WORKER_RC=$?

if [ "$WORKER_RC" -ne 0 ]; then
    printf 'name\tstatus\taligned_path\n' >"$RESULT_TSV"
    printf '%s\terror: worker crashed\t\n' "$NAME" >>"$RESULT_TSV"
fi

exit "$WORKER_RC"
