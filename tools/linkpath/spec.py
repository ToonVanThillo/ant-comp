from pathlib import Path

from prosapia.core import Tool

from .collect_linkpath import collect_linkpath
from .run_linkpath import (
    NO_DEFAULT_COLUMN,
    add_run_linkpath_args,
    build_linkpath_manifest,
)

TOOL = Tool(
    name="linkpath",
    action="update",
    description="Measure the shortest route between two chain termini that stays "
    "out of the protein -- the length a flexible linker would actually have to span.",
    default_script=str(Path(__file__).parent / "linkpath.sh"),
    # No honest default: the structure column differs per campaign, and a wrong one
    # fails silently. The builder refuses the run unless -i names a real column.
    default_input_column=NO_DEFAULT_COLUMN,
    build_manifest_fn=build_linkpath_manifest,
    add_run_args_fn=add_run_linkpath_args,
    collect_fn=collect_linkpath,
)
