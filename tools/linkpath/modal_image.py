"""Modal image for linkpath (used by ``--executor modal``).

The worker is pure Python -- gemmi for structure parsing, numpy for the occupancy
grid and the distance scans, and a stdlib ``heapq`` A* -- so the image is a slim
Debian with those two wheels and nothing else: no GPU, no torch, no scipy, no
biopython, and it builds in seconds. The prototype this was ported from used
Biopython; dropping it was a deliberate choice to match dimerfit/ringfit/ifacegeom,
and the two implementations were checked against each other on a real file (see the
"Verification done" section of the skill).

The manifest builder sets ``gpus_per_task = 0``, so callers never need ``-g 0``, and
a task measures a whole batch of designs (``--designs-per-task``) because the cold
start costs ~60x more than one design's measurement.
"""

import modal

RESOURCES = {"cpu": 2, "memory": "8G", "timeout": "00:30:00"}


def image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.12").pip_install("gemmi", "numpy")
