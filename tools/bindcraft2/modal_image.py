"""Modal image for bindcraft2 (used by ``--executor modal``).

Built the way BindCraft2's own ``containers/Dockerfile`` builds it, for the same two
reasons that file gives:

* **Editable install.** ``settings/`` and ``scaffolds/`` live at the repository root,
  not inside the package, and the runtime resolves them relative to it. A plain
  ``pip install`` copies only the package and orphans them, so the checkout has to
  stay in place and on the import path.
* **ldconfig.** The jax cuda13 wheels keep their CUDA and cuDNN libraries under
  ``nvidia/*/lib`` inside site-packages, where the loader does not look. Without the
  ``ld.so.conf.d`` entry jax silently falls back to the CPU -- it warns once, runs a
  hundred times slower, and the campaign still starts. The ``ldconfig -p | grep
  libcupti`` at the end fails the *build* instead of someone's campaign.

The AlphaFold parameters (~5.3 GB) are NOT baked in: they download on first use into
``$XDG_CACHE_HOME/bindcraft``, which is a Volume
(``SAPIA_MODAL_VOLUME_BINDCRAFT_CACHE``, default ``bindcraft-cache``) so later tasks
reuse them. Warm it with ``sapia run bindcraft2 <run_dir> --fetch-weights-only``
before fanning out, or several first containers will each pull the same 5.3 GB.
``BINDCRAFT_AF2_PARAMS`` (the parameters alone) or ``BINDCRAFT_WEIGHTS`` (the whole
cache root) in ``.env`` override the location entirely.

The same Volume also collects jax's compiled graphs, under
``$XDG_CACHE_HOME/bindcraft/compile_cache/<card>`` -- BindCraft2 sets
``JAX_COMPILATION_CACHE_DIR`` itself (``cli.use_campaign_compile_cache``), keyed by
GPU model, so later campaigns on the same card skip the compile.

GPUs: ``RESOURCES`` asks for one card, which is one serial campaign. A campaign can
instead fan its trajectories across several (``design_workers``), for which the task
needs the cards too -- ``-g 4`` makes the request ``A100:4``. The default stays at
one on purpose: the cost of a fan-out should be asked for, not inherited.
"""

import modal

from prosapia.core.executors.modal import get_named_volume

CACHE_DIR = "/bindcraft_cache"

BINDCRAFT_REPO = "https://github.com/PacesaLab/BindCraft2.git"
# Pinned to a release so a rebuild can't silently pick up new work on main. Bump
# deliberately: v1.0.3, ce3150f8d1132a0c900d66ad8f6d85ec08c9ad17.
BINDCRAFT_REF = "v1.0.3"

# A campaign is long by construction: it keeps spending trajectories until
# number_of_final_designs are accepted or max_trajectories are used up. The 12 h
# timeout is a guard, not an estimate -- size it with -T/--time per run.
RESOURCES = {"gpu": "A100", "cpu": 8, "memory": "32G", "timeout": "12:00:00"}

# The Dockerfile's ldconfig step, verbatim in spirit: point the loader at the CUDA
# libraries the jax wheels carry inside site-packages.
_LDCONFIG = (
    "python -c \"import nvidia, pathlib; print('\\n'.join(sorted(str(path) "
    "for root in nvidia.__path__ for path in pathlib.Path(root).glob('*/lib'))))\" "
    "> /etc/ld.so.conf.d/bindcraft-cuda.conf "
    "&& ldconfig && ldconfig -p | grep -q libcupti"
)


def image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        # build-essential is here for the same reason the Dockerfile keeps it:
        # biotraj (pulled in by biotite) does not publish a wheel everywhere.
        .apt_install("git", "build-essential")
        .run_commands(
            f"git clone --depth 1 --branch {BINDCRAFT_REF} {BINDCRAFT_REPO} /opt/bindcraft",
            "pip install -e '/opt/bindcraft[cuda13]'",
            _LDCONFIG,
            # Fail the build, not the campaign, if the wheels don't line up.
            "python -c \"import jax, bindcraft.proteinmpnn; print('jax', jax.__version__)\"",
            "bindcraft --help > /dev/null",
        )
        .env(
            {
                # Where `bindcraft` looks for the AlphaFold parameters it downloads.
                "XDG_CACHE_HOME": CACHE_DIR,
                "NVIDIA_VISIBLE_DEVICES": "all",
                "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            }
        )
    )


def volumes() -> dict[str, modal.Volume]:
    return {
        CACHE_DIR: get_named_volume(
            "SAPIA_MODAL_VOLUME_BINDCRAFT_CACHE", "bindcraft-cache"
        )
    }
