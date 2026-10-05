#!/bin/bash
#SBATCH --job-name=protonpottsmpnn
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

# Shared scaffolding: sets MANIFEST/OUT_DIR/SAPIA_TASK_ID/SAPIA_LINE.
source "${SAPIA_PRELUDE:?}"

# Site-specific activation (required off Modal) — see docs/configuration.md.
# Under Modal this is a no-op and the image supplies $PROTONPOTTSMPNN, $HBPLUS_PATH
# and the interpreter that can import `mpnn`.
sapia_activate SAPIA_ACTIVATE_PROTONPOTTSMPNN

# The engine is a library API, so the work is a Python step, not a command line.
# Locals are lowercase/prefixed on purpose: an uppercase name colliding with a bash
# special variable (GROUPS, UID, PIPESTATUS…) fails its assignment under `set -e` and
# kills the task before it prints anything. See the trap note in authoring-a-tool.
proton_python=${PROTONPOTTSMPNN_PYTHON:-python}

design_name=$(printf '%s' "$SAPIA_LINE" | cut -f1)
config_path=$(printf '%s' "$SAPIA_LINE" | cut -f2)

echo "Task ${SAPIA_TASK_ID}: ${design_name}"

# The worker records its own failures into designs.tsv as `status: error: ...` before
# exiting non-zero, so a failed design still collects with a readable reason.
"$proton_python" "${SAPIA_TOOL_DIR:?}/protonpottsmpnn_worker.py" "$config_path"
