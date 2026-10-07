"""Modal image for ifacegeom (used by ``--executor modal``).

The worker is pure Python -- gemmi for structure parsing, numpy for the distance
and angle work -- so the image is a slim Debian with those two wheels and nothing
else: no GPU, no torch, and it builds in seconds. The manifest builder sets
``gpus_per_task = 0``, and a task measures a whole batch of designs
(``--designs-per-task``) because the cold start costs more than the computation.
"""

import modal

RESOURCES = {"cpu": 2, "memory": "8G", "timeout": "00:30:00"}


def image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.12").pip_install("gemmi", "numpy")
