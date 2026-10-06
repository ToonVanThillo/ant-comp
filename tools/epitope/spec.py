from pathlib import Path

from prosapia.core import Tool

from .collect_epitope import collect_epitope
from .run_epitope import (
    NO_DEFAULT_COLUMN,
    add_run_epitope_args,
    build_epitope_manifest,
)

TOOL = Tool(
    name="epitope",
    action="update",
    description=(
        "Which target residues a binder actually contacts, and how much of the "
        "requested hotspot epitope it covers (heavy-atom contact only)."
    ),
    default_script=str(Path(__file__).parent / "epitope.sh"),
    # Sentinel, as in cms/chainsel/ssprofile: no structure column is a defensible
    # default (rfdiffusion3_path, boltz_path and bindcraft2_path are all normal
    # inputs), so the manifest builder raises unless -i/--input-column names a
    # column the table actually has.
    default_input_column=NO_DEFAULT_COLUMN,
    build_manifest_fn=build_epitope_manifest,
    add_run_args_fn=add_run_epitope_args,
    collect_fn=collect_epitope,
)
