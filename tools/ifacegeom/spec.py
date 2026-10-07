from pathlib import Path

from prosapia.core import Tool

from .collect_ifacegeom import collect_ifacegeom
from .run_ifacegeom import (
    NO_DEFAULT_COLUMN,
    add_run_ifacegeom_args,
    build_ifacegeom_manifest,
)

TOOL = Tool(
    name="ifacegeom",
    action="update",
    description="Interface footprint of a binder on its target, and where its "
    "termini sit relative to that interface.",
    default_script=str(Path(__file__).parent / "ifacegeom.sh"),
    # No honest default: the complex column differs per campaign, and a wrong one
    # fails silently. The builder refuses the run unless -i names a real column.
    default_input_column=NO_DEFAULT_COLUMN,
    build_manifest_fn=build_ifacegeom_manifest,
    add_run_args_fn=add_run_ifacegeom_args,
    collect_fn=collect_ifacegeom,
)
