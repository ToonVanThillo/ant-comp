from pathlib import Path

from prosapia.core import Tool

from .collect_rpxdock import collect_rpxdock
from .run_rpxdock import (
    NO_DEFAULT_COLUMN,
    add_run_rpxdock_args,
    build_rpxdock_manifest,
)

TOOL = Tool(
    name="rpxdock",
    action="create",
    description=(
        "Rigid-body dock a scaffold into a ONE-COMPONENT symmetric architecture "
        "(C2-C17, one-component cages, Dx_y) and score each pose with RPX motif "
        "tables. One scaffold row fans out to N dock rows in a child table. "
        "Multi-component architectures (T32, I32, AXLE_*, PLUG_*, ASYM, layers) "
        "are not wired and are refused by name at submit time."
    ),
    default_script=str(Path(__file__).parent / "rpxdock.sh"),
    # Sentinel, as in usalign/chainsel/cms: no structure column is a defensible
    # default (the scaffold may be rfdiffusion3_path, boltz_path or chainsel_path),
    # so the manifest builder raises unless -i/--input-column is given.
    default_input_column=NO_DEFAULT_COLUMN,
    build_manifest_fn=build_rpxdock_manifest,
    add_run_args_fn=add_run_rpxdock_args,
    collect_fn=collect_rpxdock,
)
