"""Modal image for hbdesigner (used by ``--executor modal``).

Four constraints shape this image, each read out of upstream's pyproject.toml
rather than assumed:

1. **Python 3.10, strictly** (``requires-python = ">=3.10,<3.11"``). It therefore
   cannot share the built-in pyrosetta tool's image, which is 3.12.
2. **The torch stack is pinned as a set.** This uses upstream's ``gpu-cu124``
   extra: torch 2.6.0+cu124 plus the prebuilt ``torch_cluster`` / ``torch_scatter``
   wheels built against exactly that torch. They are binary wheels ONLY for that
   version -- let pip resolve a different torch and they fall back to compiling
   from source (upstream sets ``no-build-isolation-package`` for both for the same
   reason). So torch is installed first, from its own URL, and the two PyG wheels
   are installed from their URLs with ``--no-deps``; nothing is allowed to
   re-resolve them.
3. **PyRosetta comes from the pinned public wheel** named in upstream's
   ``[tool.uv.sources]`` (``PyRosetta4.Release.python310...2024.39``), not from
   ``pyrosetta-installer`` (which is the 3.12 route the built-in pyrosetta tool
   uses). Free for academic use; a commercial user needs their own licence.
4. **``numpy==1.26.4``** is pinned upstream; respected here.

The repo is cloned at a PINNED COMMIT and installed EDITABLE, and both halves of
that matter:

  * pinned, so a rebuild cannot silently pick up new work on ``main``;
  * editable, because ``inference_hbdesigner`` resolves its weights as
    ``Path(__file__).parents[2]/model_weights/<model>.pt``. A non-editable install
    puts ``__file__`` in site-packages, where that path does not exist -- and
    setuptools' ``packages.find`` would also drop ``hbdesigner/scripts/`` (no
    ``__init__.py``), taking ``graft_seq.py`` with it. Editable keeps both working
    off the baked-in checkout. ``--no-deps`` is used because the dependency set is
    curated here (``pyrosetta`` is a bare name upstream resolves through a uv
    source that pip does not read).

**Upstream UNDER-DECLARES its dependencies**, so ``--no-deps`` cannot be trusted to
reproduce what the code actually imports: ``[project].dependencies`` at the pinned
commit lists only pyrosetta, numpy, omegaconf, biopython, wandb, networkx, pandas
and pebble. An AST scan of all 29 source files under ``hbdesigner/``, plus a
transitive internal-import trace from ``hbdesigner.inference.inference_hbdesigner``,
found these undeclared imports, all installed explicitly below:

  * ``scipy`` -- MODULE-LEVEL in ``data/hbnet.py``, ``data/protein.py`` and
    ``inference/protein_ops.py``. Its absence is what broke the first build.
    Pinned ``<1.16`` because scipy 1.16 dropped Python 3.10 and this image is
    3.10 by upstream's ``requires-python``. Do not loosen that bound.
  * ``git`` (GitPython) -- MODULE-LEVEL in ``train/trainer.py``, which
    ``inference_hbdesigner`` reaches transitively (the same quirk that makes
    ``wandb`` a hard dependency of an inference-only run).
  * ``biotite`` and ``hydride`` -- LAZY imports, inside ``biotite_hbond_detect()``
    and ``run_hydride()`` in ``data/hbnet.py``. Upstream has them in
    ``[dependency-groups].dev`` only. ``biotite_hbond_detect()`` IS the scoring
    function behind this tool's ``hb_score_full`` / ``hb_score_hb`` /
    ``avg_burial`` / ``saturation`` / ``buried_heavy_unsats`` /
    ``buried_unsat_hpol`` columns, so without them the build PASSES and a GPU task
    dies mid-scoring instead.
  * ``tqdm`` -- module-level in ``scripts/merge_networks.py`` only, which is off
    the inference path. Installed anyway; it is tiny.

The ``run_commands`` smoke tests are the guard. ``run_hbdesigner --help`` cannot
catch a lazily imported runtime dependency, which is exactly how biotite/hydride
would otherwise have reached a GPU task -- hence the separate import test that
forces the lazy set to resolve at build time. **Re-run the import scan when the
pinned commit is bumped**: new undeclared imports will not show up in upstream's
metadata.

The ~160 MB of model weights are plain git files (no LFS), so the clone brings
them along and they are baked into the image: no Volume, no manual weight
population, nothing for an agent to install by hand.

``wandb`` is a hard import dependency -- ``inference_hbdesigner`` imports the
training module, which imports ``wandb`` at module level -- so the image disables
it rather than letting a task container try to phone home.

Resources: the GNN sampling wants a GPU, but packing and Rosetta scoring are
CPU-parallel (``--n-workers``, which defaults to the CPU count), so this is a
1-GPU / 16-CPU box with a generous timeout: ``--n-samples 100+`` with PyRosetta
packing is not fast.
"""

import modal

# HEAD of RosettaCommons/HBDesigner `main`, 2026-08-31 (merge of #9,
# "ReadmeAndPixiUpdates"), resolved with `git ls-remote` on 2026-10-01. Bump
# deliberately -- a moving branch would change what a campaign ran.
HBDESIGNER_REPO = "https://github.com/RosettaCommons/HBDesigner.git"
HBDESIGNER_COMMIT = "ed65fa053786394a33efcedd5624c80ffbfef12b"
HBDESIGNER_DIR = "/opt/HBDesigner"

# Upstream's `gpu-cu124` extra, verbatim.
TORCH_WHEEL = (
    "https://download.pytorch.org/whl/cu124/"
    "torch-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl"
)
TORCH_CLUSTER_WHEEL = (
    "https://data.pyg.org/whl/torch-2.6.0%2Bcu124/"
    "torch_cluster-1.6.3%2Bpt26cu124-cp310-cp310-linux_x86_64.whl"
)
TORCH_SCATTER_WHEEL = (
    "https://data.pyg.org/whl/torch-2.6.0%2Bcu124/"
    "torch_scatter-2.1.2%2Bpt26cu124-cp310-cp310-linux_x86_64.whl"
)
# Upstream's [tool.uv.sources] pyrosetta pin.
PYROSETTA_WHEEL = (
    "https://west.rosettacommons.org/pyrosetta/release/release/"
    "PyRosetta4.Release.python310.linux.cxx11thread.serialization.wheel/"
    "pyrosetta-2024.39%2Brelease.59628fb-cp310-cp310-linux_x86_64.whl"
)

RESOURCES = {"gpu": "L4", "cpu": 16, "memory": "32G", "timeout": "04:00:00"}


def image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git")
        # setuptools/wheel are for the --no-build-isolation editable install below.
        .pip_install("setuptools>=61", "wheel", "numpy==1.26.4", TORCH_WHEEL)
        # Prebuilt against torch 2.6.0+cu124: --no-deps keeps pip from touching
        # the torch that was just installed.
        .pip_install(TORCH_CLUSTER_WHEEL, TORCH_SCATTER_WHEEL, extra_options="--no-deps")
        .pip_install(
            # Re-asserted here (already installed above) so the resolver treats it
            # as a hard constraint and cannot drift numpy while solving scipy /
            # biotite. Nothing in this layer may re-resolve torch* -- it does not,
            # because every package here is satisfied by what is already present.
            "numpy==1.26.4",
            "torch-geometric==2.7.0",
            PYROSETTA_WHEEL,
            "omegaconf>=2.3.0,<3",
            "biopython>=1.86,<2",
            "wandb>=0.25.0,<0.26",
            "networkx>=3.4.2,<4",
            "pandas>=2.3.3,<3",
            "pebble>=5.2.0,<6",
            # Undeclared upstream (see the module docstring). scipy and GitPython
            # are module-level imports on the inference path; biotite and hydride
            # are lazy imports inside the scoring path, so only the extra smoke
            # test below catches them at build time.
            "scipy>=1.10,<1.16",  # <1.16: scipy 1.16 dropped py3.10. Load-bearing.
            "GitPython>=3.1,<4",  # provides the `git` module
            "biotite>=1.2.0,<2",  # upstream's own dev pin
            "hydride>=1.1.2,<2",  # upstream's own dev pin
            "tqdm",
            # For hbdesigner_worker.py (residue mapping / graft verification), the
            # house structure library. Not an upstream dependency.
            "gemmi",
        )
        .run_commands(
            f"git clone {HBDESIGNER_REPO} {HBDESIGNER_DIR}",
            f"git -C {HBDESIGNER_DIR} checkout {HBDESIGNER_COMMIT}",
            f"git -C {HBDESIGNER_DIR} rev-parse HEAD > {HBDESIGNER_DIR}/COMMIT",
            # Weights ship in the repo as plain files; fail the BUILD, not a task,
            # if that ever stops being true.
            f"test -f {HBDESIGNER_DIR}/model_weights/design_020.pt",
            f"test -f {HBDESIGNER_DIR}/model_weights/design_002.pt",
            f"test -f {HBDESIGNER_DIR}/model_weights/pack.pt",
            f"pip install --no-deps --no-build-isolation -e {HBDESIGNER_DIR}",
            # Smoke tests: the console script exists, the graft helper is
            # importable (it lives in a dir with no __init__.py), and the torch
            # stack loads together.
            "run_hbdesigner --help > /dev/null",
            "python -c 'import hbdesigner.scripts.graft_seq'",
            "python -c 'import torch, torch_cluster, torch_scatter, pyrosetta, gemmi'",
            # The three tests above exercise only what is imported at module level
            # on the inference path. biotite and hydride are imported INSIDE
            # biotite_hbond_detect() / run_hydride(), i.e. inside scoring, so their
            # absence passes every test above and kills a GPU task hours later.
            # This line forces the whole undeclared set to resolve at BUILD time.
            "python -c 'import scipy, git, biotite.structure, hydride, tqdm'",
        )
        .env(
            {
                "HBDESIGNER": HBDESIGNER_DIR,
                "WANDB_MODE": "disabled",
                "WANDB_SILENT": "true",
            }
        )
    )
