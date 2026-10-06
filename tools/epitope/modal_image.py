"""Modal image for epitope (used by ``--executor modal``).

The worker is pure Python -- gemmi to read PDB/mmCIF, numpy for the heavy-atom
distance matrix -- so the image is a slim Debian with those two wheels and nothing
else: no GPU, no scipy, no biopython, and it builds in seconds. The measurement is
geometry; anything heavier would be a dependency bought for nothing. The manifest
builder sets ``gpus_per_task = 0``, so callers never need ``-g 0``.
"""

import modal

RESOURCES = {"cpu": 2, "memory": "4G", "timeout": "00:30:00"}


def image() -> modal.Image:
    return modal.Image.debian_slim(python_version="3.12").pip_install("gemmi", "numpy")
