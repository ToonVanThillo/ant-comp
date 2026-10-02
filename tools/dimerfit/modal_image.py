"""Modal image for dimerfit (used by ``--executor modal``).

The worker is pure Python -- gemmi for structure parsing and writing, numpy for the
Kabsch fit and the distance scans -- so the image is a slim Debian with those two
wheels and nothing else: no GPU, no torch, no scipy, no biopython, and it builds in
seconds. The superposition is implemented here (``kabsch``) rather than pulled in as
a dependency, which is the precedent ringfit and ifacegeom set.

The manifest builder sets ``gpus_per_task = 0``, so callers never need ``-g 0``, and
a task measures a whole batch of designs (``--designs-per-task``) because the cold
start costs more than the computation.
"""

import modal

RESOURCES = {"cpu": 2, "memory": "8G", "timeout": "00:30:00"}


def image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.12").pip_install("gemmi", "numpy")
