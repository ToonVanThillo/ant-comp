"""User-side SHADOW of the bundled ``boltz`` tool: per-chain MSA policy.

Rung 5 of ``editing-a-tool``: ``get_builtin("boltz")`` with two hooks replaced.
The name stays ``boltz``, so this SHADOWS the built-in wherever this ``tools/``
dir is on ``$PROSAPIA_TOOLS_DIR`` -- deliberately, because the columns
(``boltz_path``, ``boltz_confidence_score``, ...) and the output dir must stay
identical for every downstream step and every existing run_dir.

Not overridden, and therefore still tracking upstream: ``default_script``
(the bundled ``boltz.sh``), its ``modal_image.py``, ``collect_fn``, ``action``
and ``default_input_column``. See ``run_boltz.py`` for what the two replaced
hooks add and refuse.
"""

from prosapia.core import get_builtin

from .run_boltz import add_run_boltz_msa_args, build_boltz_msa_manifest

TOOL = get_builtin("boltz").with_overrides(
    description=(
        "Run Boltz structure predictions (per-chain MSA policy via "
        "--msa-empty-chains)"
    ),
    add_run_args_fn=add_run_boltz_msa_args,
    build_manifest_fn=build_boltz_msa_manifest,
)
