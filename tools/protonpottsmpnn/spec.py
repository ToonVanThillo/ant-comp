from pathlib import Path

from prosapia.core import Tool

from .collect_protonpottsmpnn import collect_protonpottsmpnn
from .run_protonpottsmpnn import (
    add_run_protonpottsmpnn_args,
    build_protonpottsmpnn_manifest,
)

TOOL = Tool(
    name="protonpottsmpnn",
    action="create",
    description=(
        "Design pH-switchable binder sequences with Proton-PottsMPNN: pin protonated "
        "centres (HIS-P/ASP-P/GLU-P) and redesign their neighbourhood along the "
        "stability<->pH-selectivity trade-off."
    ),
    default_script=str(Path(__file__).parent / "protonpottsmpnn.sh"),
    # Same choice as the atomium tool: the bundled proteinmpnn's default is the older
    # `rfdiffusion_path` (no 3) and silently submits nothing after an rfd3 run, so this
    # defaults to the column rfd3 actually writes in this workspace. Point -i at
    # bindcraft2_traj_path / bindcraft2_path to redesign a BindCraft2 complex instead.
    default_input_column="rfdiffusion3_path",
    build_manifest_fn=build_protonpottsmpnn_manifest,
    add_run_args_fn=add_run_protonpottsmpnn_args,
    collect_fn=collect_protonpottsmpnn,
)
