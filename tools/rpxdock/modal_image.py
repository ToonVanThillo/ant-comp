"""Modal image for rpxdock (used by ``--executor modal``).

CPU-ONLY. There is no ``gpu`` key in ``RESOURCES`` and the manifest builder forces
``gpus_per_task = 0``, so callers do not need ``-g 0``.

Four things in this image are load-bearing, each one measured rather than assumed
(reproduced in a ``python:3.12-slim-bookworm`` container, which is what
``debian_slim`` builds on):

* **A C++ toolchain, and a prewarm.** RPXdock ships 21 C++ extensions (BVH, xbin,
  phmap, the hierarchical samplers, the motif maps) and compiles them with
  ``cppimport`` on FIRST IMPORT, writing the ``.so`` files next to the sources
  inside site-packages. Measured: ~7 minutes. Left to run time, every cold
  container repays that and several containers race to write the same files, so
  the build does one plain ``import rpxdock`` and then ``CPPIMPORT_RELEASE_MODE=1``
  forbids any further compiling at run time -- a missing extension then fails
  loudly as an ImportError instead of being silently rebuilt per task.

* **A verification that fails the BUILD.** Following bindcraft2's precedent. It is
  not optional paranoia: upstream's own ``rpxdock/util/parallel_build_modules.py``
  submits builds to a process pool and never calls ``.result()``, and its
  ``maybe_build`` wraps the call in ``except None:`` -- so a failed compile there
  is silent. That helper is deliberately NOT used (it also derives cppimport module
  names by string-mangling absolute paths, which does not resolve for an installed
  package: measured, it compiles nothing at all from site-packages). The
  verification here checks every needed ``.so`` exists and imports, checks
  PyRosetta is absent, checks the ``tar``/``bzip2`` binaries ``result_to_tarball``
  shells out to, and then runs a real C2 dock on RPXdock's own shipped fixtures.

* **A neutral working directory for the prewarm.** ``cppimport`` shells out to
  setuptools, which runs flat-layout package discovery against the CURRENT
  DIRECTORY; from a directory holding stray top-level dirs the build dies with
  "Multiple top-level packages discovered in a flat-layout" and never touches
  RPXdock (measured, locally).

* **``numpy<2``.** ``setup.py`` has ``numpy<2`` commented out, and on numpy 2.4
  the one-component CAGE/DIHEDRAL path dies for every input at
  ``sampling/xform_hier.py:59``::

      ang_nstep = int(np.ceil(ang / angresl))
      TypeError: only 0-dimensional arrays can be converted to Python scalars

  (``DockSpec1CompCage`` stores ``nfold`` as a 1-element array, and numpy 2 made
  ``int()`` on that an error.) Pinning numpy 1.26 restores it. The pin costs one
  thing, also measured: ``Body.filter_pairs`` uses ``np.ones(..., dtype=np.bool)``,
  and ``np.bool`` exists in numpy 2 but NOT in 1.26 -- so RPXdock's
  ``--score_only_sspair`` raises ``AttributeError`` under this pin. That option is
  therefore not exposed as a flag; it is reachable only through ``--set``, with the
  trap documented in the skill.

PyRosetta is deliberately NOT installed, and the ``[pyrosetta]`` extra is not used.
That is the configuration RPXdock's CLI defaults to (``--use_rosetta`` is a
``store_true``, so False -- verified in-container). What it forgoes: full-atom
Rosetta poses, Rosetta's DSSP (willutil's pure-Python ``wu.dssp`` is used instead),
and the helix-termini accessibility options (``--term_access*`` /
``--termini_dir*``), which print "no pyrosetta, helix termini stuff diabled" and
are skipped. It also means ``rp.search.result_from_tarball`` cannot reopen a
result's bodies (``AttributeError: module 'rpxdock.rosetta.triggers_init' has no
attribute 'pose_from_file'``, verified) -- see ``rpxdock_worker.py``, which is
written around that.

The motif tables ("hscore") are NOT baked in: they live on the ``rpxdock-hscore``
Volume (``SAPIA_MODAL_VOLUME_RPXDOCK_HSCORE``) mounted at ``/rpxdock_files``, which
is also RPXdock's own ``--hscore_data_dir`` default, so upstream defaults work
untouched. The Volume holds three lowercase alias dirs: ``ilv_h`` (~365 MB, the
tool default), ``ailv_h`` (~1.4 GB: 475 MB of ``.txz`` plus 923 MB of
``.txz.pickle`` sidecars, which take precedence over the tarballs and are
untested), ``afilmv_ehl`` (~5.7 GB, SS-dependent).
"""

import base64

import modal

from prosapia.core.executors.modal import get_named_volume

HSCORE_DIR = "/rpxdock_files"

RPXDOCK_REPO = "https://github.com/willsheffler/rpxdock"
# Pinned to an explicit commit of main so a rebuild cannot silently pick up new
# upstream work -- RPXdock publishes no releases and main moves. Bump
# deliberately: 2026-04-01, "Fix log-normal sigma parameter in score functions".
RPXDOCK_COMMIT = "61264c6cbaae882bf52d6547cd2d40b6ed4de0d9"

# See the module docstring: numpy 2 breaks the one-component cage/dihedral sampler.
NUMPY_PIN = "numpy<2"

# Where the prewarm and the verification run: empty, so setuptools' flat-layout
# discovery has nothing to trip over.
BUILD_DIR = "/opt/rpxdock_build"

# CPU-only. The hierarchical search is effectively single-process (the protocols
# use rp.util.InProcessExecutor), so the cores are for BLAS and the compiled
# kernels rather than for fan-out; the memory is sized for the hscore load, which
# decompresses ~365 MB of xz motif tables per container for the default ilv_h and
# ~5.7 GB for afilmv_ehl. Raise --mem before using afilmv_ehl.
RESOURCES = {"cpu": 4, "memory": "16G", "timeout": "02:00:00"}

# The compiled extensions `import rpxdock` is expected to produce. Anything not in
# this list (bvh_nd, bvh_xform, _orientations, and the *_test modules) is not built
# by a plain import and is not on the docking path -- verified by listing the .so
# files after a prewarm.
REQUIRED_EXTENSIONS = (
    "rpxdock.bvh.bvh",
    "rpxdock.cluster.cookie_cutter",
    "rpxdock.geom.bcc",
    "rpxdock.geom.expand_xforms",
    "rpxdock.geom.miniball",
    "rpxdock.geom.xform_dist",
    "rpxdock.motif._motif",
    "rpxdock.phmap.phmap",
    "rpxdock.sampling.xform_hierarchy",
    "rpxdock.xbin.smear",
    "rpxdock.xbin.xbin",
    "rpxdock.xbin.xbin_util",
)

# Build-time verification. Written with double quotes only, so base64 round-trips
# it through the shell without a quoting fight. Fails the build, not a campaign.
_VERIFY = (
    """
import importlib, os, pathlib, shutil, sys, sysconfig
import numpy as np
import rpxdock as rp

required = %r
site = pathlib.Path(sysconfig.get_paths()["purelib"])
for dotted in required:
    parts = dotted.split(".")
    srcdir = site.joinpath(*parts[:-1])
    if not list(srcdir.glob(parts[-1] + "*.so")):
        sys.exit("cppimport extension did not build: " + dotted)
    importlib.import_module(dotted)
print("extensions OK:", len(required))

import rpxdock.rosetta.triggers_init as ti
if ti.HAVE_PYROSETTA:
    sys.exit("pyrosetta is present; this image is built for the pyrosetta-free "
             "path and the worker assumes it")

for exe in ("tar", "bzip2"):
    if shutil.which(exe) is None:
        sys.exit("missing binary needed by result_to_tarball: " + exe)

# A real dock on RPXdock's own shipped fixtures: the 860 KB small_ilv_h TEST table
# and a C3 asymmetric unit. Proves the compiled stack, the no-PyRosetta body
# loader, the willutil DSSP and dump_pdb all work together.
hscore_dir = os.path.join(os.path.dirname(rp.data.__file__), "hscore")
pdb = os.path.join(rp.data.pdbdir, "C3_1na0-1_1.pdb.gz")
argv = ["--architecture", "C2", "--inputs1", pdb,
        "--hscore_files", "small_ilv_h", "--hscore_data_dir", hscore_dir,
        "--beam_size", "5000", "--save_results_as_pickle", "False",
        "--overwrite_existing_results", "--output_prefix", "/tmp/verify/rpx_"]
kw = rp.options.get_cli_args(argv)
if kw.use_rosetta:
    sys.exit("expected --use_rosetta to default to False")
hscore = rp.RpxHier(kw.hscore_files, **kw)
from rpxdock.app import dock as app
result = app.dock_cyclic(hscore, stack=False, **kw)
scores = np.asarray(result.data["scores"].data)
if len(scores) == 0 or not scores.max() > 0:
    sys.exit("smoke dock produced no scoring pose")
body = result.bodies[0][0]
ss = np.asarray(body.ss)
if not (ss == "H").mean() > 0.5:
    sys.exit("willutil dssp found almost no helix in a helical fixture")
out = "/tmp/verify/smoke.pdb"
os.makedirs("/tmp/verify", exist_ok=True)
result.dump_pdb(int(np.argmax(scores)), fname=out, sym="C2", **kw)
if not os.path.getsize(out) > 0:
    sys.exit("dump_pdb wrote nothing")
rp.search.result_to_tarball(result, "/tmp/verify/smoke.txz", overwrite=True)
print("rpxdock image OK: numpy", np.__version__, "ndock", len(scores),
      "best", float(scores.max()))
"""
    % (REQUIRED_EXTENSIONS,)
)

_VERIFY_B64 = base64.b64encode(_VERIFY.encode()).decode()


def image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        # g++ is mandatory: the 21 extensions are compiled, not shipped. bzip2 is
        # what result_to_tarball's `tar cjf` shells out to.
        .apt_install("git", "build-essential", "bzip2")
        .pip_install(NUMPY_PIN, f"git+{RPXDOCK_REPO}@{RPXDOCK_COMMIT}")
        .run_commands(
            f"mkdir -p {BUILD_DIR}",
            # Prewarm from an empty cwd (see the module docstring). ~7 min.
            f"cd {BUILD_DIR} && python -c 'import rpxdock'",
            f"cd {BUILD_DIR} && echo {_VERIFY_B64} | base64 -d > verify.py"
            f" && python verify.py",
        )
        .env(
            {
                # RPXdock's own default, and the Volume mount point.
                "RPXDOCK_HSCORE_DIR": HSCORE_DIR,
                # Everything is prebuilt: forbid run-time compiling so a missing
                # extension is an ImportError rather than a silent rebuild racing
                # across containers. Set AFTER the build steps on purpose.
                "CPPIMPORT_RELEASE_MODE": "1",
            }
        )
    )


def volumes() -> dict[str, modal.Volume]:
    return {
        HSCORE_DIR: get_named_volume(
            "SAPIA_MODAL_VOLUME_RPXDOCK_HSCORE", "rpxdock-hscore"
        )
    }
