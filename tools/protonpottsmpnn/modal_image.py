"""Modal image for protonpottsmpnn (used by ``--executor modal``).

The repo is PUBLIC, so unlike atomium this needs no Modal Secret -- it is cloned at
BUILD time and pinned to a commit, which also means tasks never re-clone.

Weights ship IN the repo and total ~64 MB, so there is no weights Volume:

    checkpoints/…/epoch-0125.ckpt                       21 MB   the v6 Potts model
    …/mpnn/transforms/ev6/weights/{his,acid}/*.pkl      43 MB   the FLAML labeller folds

HBPLUS -- the one thing you must supply by hand
-----------------------------------------------
Proton-PottsMPNN needs the HBPLUS binary AT DESIGN TIME. The repo's README says only the
labeller needs it; that is wrong. ``prepare_potts_input`` builds its inference pipeline
through ``get_protonation_state_transforms``, which runs ``CalculateHbondsPlus``
unconditionally (``pipelines/potts_mpnn.py:269``), and the v6 vocabulary consumes those
bonds as its feature pool. Every single design call shells out to it.

HBPLUS is a C program by Ian McDonald (UCL/EBI). It is NOT on PyPI (404) or conda-forge
(no such package), and the EBI download sits behind a licence form -- the plain tarball
URL serves an HTML page -- so no image build can fetch it. It is therefore VENDORED into
this tool:

    tools/protonpottsmpnn/vendor/hbplus.tar.gz      sha256 937467447bd2e429… (148 KB)
    tools/protonpottsmpnn/vendor/HBPLUS_README.md   upstream readme, kept for provenance

Pure C, no Fortran: the bundled ``accall.f`` belongs to the separate accessibility
program and the default make target builds ``hbplus`` alone, from gcc + ``-lm``. The
binary needs no runtime data files (``standard.data`` / ``vdw.radii`` are accall's); it
only optionally reads ``$HOME/.hbplusrc``. Verified by compiling it.

The build untars, compiles and exports ``HBPLUS_PATH``. With the tarball absent, image()
fails immediately with that instruction rather than producing an image whose every task
dies hundreds of frames deep in a FileNotFoundError on the original author's laptop path
(the fallback hardcoded in ``bond_annotation.calculate_hbonds``).
"""

from pathlib import Path

import modal

# Pinned so a rebuild cannot silently pick up new work on a freshly published repo.
# Bump deliberately: HEAD of `main`, 2026-09-30 "title update".
PROTON_REPO = "https://github.com/christian-creator/ProtonPottsMPNN.git"
PROTON_COMMIT = "09682abfa7d20e0abcdeea0490b7a4b1c190aee3"
PROTON_DIR = "/opt/protonpottsmpnn"

# The HBPLUS source tarball you supply (see the module docstring).
VENDOR_HBPLUS = Path(__file__).parent / "vendor" / "hbplus.tar.gz"

# CPU-only: the design engine forks a pool over the lambda ladder, and forking after
# CUDA init is unsafe, so run_ph_redesign only parallelises on a CPU device. The manifest
# builder also forces gpus_per_task = 0, so callers never need -g 0.
RESOURCES = {"cpu": 8, "memory": "16G", "timeout": "02:00:00"}


def image() -> modal.Image:
    if not VENDOR_HBPLUS.is_file():
        raise FileNotFoundError(
            f"protonpottsmpnn needs the HBPLUS source tarball at {VENDOR_HBPLUS}.\n"
            f"Proton-PottsMPNN's protonation labeller runs HBPLUS on EVERY design call "
            f"(pipelines/potts_mpnn.py:269), HBPLUS is not pip/conda installable, and "
            f"its download is behind an academic licence form, so the image build "
            f"cannot fetch it for you.\n"
            f"Get it from https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/ and save "
            f"the tarball to that path."
        )

    return (
        # foundry (rc-foundry) pins requires-python >=3.12,<3.13.
        modal.Image.debian_slim(python_version="3.12")
        # libgomp1 is the OpenMP runtime xgboost links at import. MEASURED: without an
        # OpenMP runtime the task dies at `AnnotateProtonationStates` with
        # "XGBoost Library could not be loaded", i.e. AFTER HBPLUS has already run --
        # so it looks like a protonation-labelling bug rather than a missing apt package.
        # gcc pulls it in on debian_slim today; named explicitly so a slimmer base or a
        # future image that drops the compiler cannot silently take it away.
        .apt_install("git", "gcc", "make", "libc6-dev", "libgomp1")
        .run_commands(
            f"git clone {PROTON_REPO} {PROTON_DIR}",
            f"git -C {PROTON_DIR} checkout {PROTON_COMMIT}",
            f"git -C {PROTON_DIR} rev-parse HEAD > {PROTON_DIR}/COMMIT",
            f"rm -rf {PROTON_DIR}/.git",
            # Fail the BUILD, not every task, if the shipped weights are missing.
            f"test -f {PROTON_DIR}/checkpoints/potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt",
            f"test -f {PROTON_DIR}/foundry/models/mpnn/src/mpnn/transforms/ev6/weights/his/fold1.pkl",
        )
        # One resolution for the foundry core AND the extras: install.sh is explicit that
        # installing them separately lets the resolver prune the extras. The FLAML stack
        # is NOT optional here -- the v6 protonation labeller is a pickled FLAML model, so
        # a design call imports flaml/xgboost/lightgbm/scikit-learn.
        .run_commands(
            f"pip install -e {PROTON_DIR}/foundry -r {PROTON_DIR}/requirements-extra.txt",
            f"python -P -c \"import mpnn, inspect; assert 'protonpottsmpnn' in "
            f"inspect.getfile(mpnn).lower(), inspect.getfile(mpnn)\"",
        )
        # HBPLUS: compile the tarball you vendored. hbplus ships a plain Makefile.
        .add_local_file(str(VENDOR_HBPLUS), "/tmp/hbplus.tar.gz", copy=True)
        .run_commands(
            "mkdir -p /opt/hbplus && tar xzf /tmp/hbplus.tar.gz -C /opt/hbplus --strip-components=1",
            "cd /opt/hbplus && make",
            "test -x /opt/hbplus/hbplus",
        )
        .env(
            {
                "PROTONPOTTSMPNN": PROTON_DIR,
                "HBPLUS_PATH": "/opt/hbplus/hbplus",
                # The engine forks a CPU pool; letting each worker grab every core
                # oversubscribes the container badly.
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
    )
