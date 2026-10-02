from pathlib import Path

from prosapia.core import Tool

from .collect_hbdesigner import collect_hbdesigner
from .run_hbdesigner import add_run_hbdesigner_args, build_hbdesigner_manifest

TOOL = Tool(
    name="hbdesigner",
    action="create",
    description=(
        "Design buried hydrogen-bond networks onto existing backbones with "
        "HBDesigner (GNN + PyRosetta). Keeps --top-k ranked networks per backbone; "
        "every non-network position comes back as glycine, so the output feeds "
        "sequence design, it does not replace it."
    ),
    default_script=str(Path(__file__).parent / "hbdesigner.sh"),
    # The intended chain is rfd3 -> hbdesigner -> proteinmpnn, so the default is
    # the column rfd3 actually writes in this workspace. Unlike the bundled
    # proteinmpnn tool (whose default `rfdiffusion_path` matches nothing after an
    # rfd3 run and silently submits zero tasks), the builder RAISES when the
    # column is absent from the table.
    default_input_column="rfdiffusion3_path",
    build_manifest_fn=build_hbdesigner_manifest,
    add_run_args_fn=add_run_hbdesigner_args,
    collect_fn=collect_hbdesigner,
)
