#!/usr/bin/env python3
"""
Measure the shortest route between two chain termini that stays out of the protein --
the length a flexible linker would actually have to span.

This is the per-array-task step of the linkpath tool (and works standalone). The
premise, in one sentence: *a straight-line C-term-to-N-term distance is only a LOWER
BOUND -- if the direct vector passes through the protein body the linker must go
around, so compute the shortest path through solvent-accessible space.*

Method (ported from the verified prototype ``linker_path.py``, Biopython -> gemmi):

1. Build a 3D occupancy grid over the obstacle atoms, padded so a path can leave the
   surface.
2. Mark a voxel BLOCKED when its centre lies within (vdW radius + ``--probe``) of any
   obstacle atom. ``probe`` approximates the thickness of the linker backbone.
3. Free a small bubble (``--carve``) around each anchor atom so the chain can leave
   its own terminus, then A* over the free voxels (26-connected, true Euclidean edge
   costs, admissible Euclidean heuristic) from the start anchor to the goal anchor.
4. Report the route length and convert it into a residue count.

Residue conversion::

    n_res_min     = ceil(d / --taut-rise)     3.5 A/res: fully extended, strained
    n_res_relaxed = ceil(d / --relaxed-rise)  2.1 A/res: <=60% of contour, a coil

Scope limits -- where these numbers mean nothing
------------------------------------------------
* **The obstacle set is whatever chains you point it at** (``--obstacle-chains``).
  That choice *is* the measurement: route a dimer through only its own two protomers
  and you get a different -- and, if a target is really there, a dishonestly short --
  answer than routing it through the whole complex. The resolved list is written back
  as ``obstacle_chains`` so a later reader can audit it without re-deriving anything.
* **The path is computed on a RIGID structure.** Real linkers are flexible and real
  termini move; this is a geometric floor, not a thermodynamic statement. It says
  nothing about whether a linker of that length is stable, soluble or entropically
  affordable.
* **The grid path is itself a LOWER BOUND.** A real chain has volume and cannot hug
  the surface exactly, and the route is a 26-connected voxel path, not a smooth
  curve. Add margin.
* **Existing unresolved or flexible terminal residues already contribute length** a
  designer may not need to add. The anchors are the last/first *resolved* amino acid;
  anything disordered beyond them is free length this tool cannot see.
* **No energy, no sequence, no flexibility.** It does not relax, repack, score or
  check that a linker of n residues is synthesisable. It is pure geometry.

"No free path found" is a RESULT, not a failure: the design is unlinkable at that
probe radius. The status stays ``OK``, ``path_found`` is False, the distance and
residue-count columns are NA and ``note`` says why. Genuine failures (a missing
chain, a chain with no amino acids, an unreadable file, ``auto`` on a structure that
does not hold exactly two protein chains) are recorded as data with a status starting
``error:`` and every metric NA, so a partial array still collects and the batch still
exits 0.

Writes a one-row TSV (name, status, path, then the metrics) that collect_linkpath.py
merges back into the table, plus ``<name>_path.pdb``: the route as a chain of
pseudo-atoms, loadable alongside the structure in PyMOL (chain X = forward route,
chain Y = the reverse route when ``--both-directions``).

Normally invoked per array task by the linkpath tool -- run ``sapia run linkpath``
rather than calling this directly.

Usage (standalone, one task file of ``name<TAB>structure<TAB>from_res<TAB>to_res``
lines, where ``-`` means "the chain's own terminus"):
    python tools/linkpath/linkpath_worker.py \\
        --task-file task_0.tsv --out-dir out/ \\
        --from-chain A --to-chain B --obstacle-chains all --probe 2.0
"""

import argparse
import csv
import heapq
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

# van der Waals radii (A). Copied verbatim from the prototype so the occupancy grid
# is bit-for-bit the same; DEFAULT_VDW covers anything not listed.
VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.73, "ZN": 1.39, "FE": 2.00, "CA": 2.31, "NA": 2.27,
    "K": 2.75, "MN": 2.05, "CU": 1.40, "NI": 1.63, "CO": 2.00,
}  # fmt: skip
DEFAULT_VDW = 1.70

# Solvent is never an obstacle: a crystallographic water is not something a linker
# has to route around.
WATERS = {"HOH", "WAT", "DOD", "H2O", "TIP", "SOL"}

# 'auto' terminus selection is only defined for a two-chain structure: with one chain
# there is nothing to link to, and with three there is no single answer. Guessing
# would produce a plausible number for a question nobody asked.
AUTO_CHAIN_COUNT = 2

# path_dist >= straight_dist is a mathematical invariant (the A* cost is the length of
# a polyline between the two anchors, and a polyline is never shorter than its chord).
# A violation means the grid or the bookkeeping is broken, so it is an error row. The
# tolerance absorbs the float32 g_score accumulator the prototype uses.
INVARIANT_TOL = 1e-3

# How far below ``--probe`` the measured route clearance may fall before the row is
# downgraded to 'warn:'. Two free voxel centres a body diagonal apart can have their
# midpoint dip ~0.1 A closer to a nearby atom than either endpoint (chord vs arc), so
# a small slack is geometry, not tunnelling. Anything beyond it is.
CLEARANCE_SLACK = 0.25

# Metric columns of the per-design TSV, in order (bare names: collect_linkpath.py
# hands them to the driver, which leaf-prefixes them to linkpath_<name>).
METRIC_COLUMNS = [
    # --- provenance / trust: what was actually measured ------------------------
    "from_res",
    "to_res",
    "from_atom",
    "to_atom",
    "from_chain",
    "to_chain",
    "obstacle_chains",
    "n_obstacle_atoms",
    "probe",
    "spacing",
    "grid_shape",
    "direct_clear",
    # --- geometry --------------------------------------------------------------
    "straight_dist",
    "straight_ca_dist",
    "path_dist",
    "detour_ratio",
    "path_found",
    "min_clearance",
    "rev_path_dist",
    "path_asymmetry",
    # Each rise constant sits beside the count it produced: a residue count is
    # meaningless without it, and the rise is the MORE arbitrary of the two -- 2.1
    # A/res is a convention about how taut a coil may be, not a physical constant,
    # and a later reader will want to re-derive the count at a different one. Both
    # are NA whenever the count they explain is NA (--no-residue-estimate, or no
    # route), because recording a rise that was not applied to anything would be
    # worse than recording nothing.
    "n_res_min",
    "taut_rise",
    "n_res_relaxed",
    "relaxed_rise",
    "note",
    "seconds",
]
RESULT_COLUMNS = ["name", "status", "path", *METRIC_COLUMNS]


# ---------------------------------------------------------------------------
# Structure handling
# ---------------------------------------------------------------------------


def _is_amino_acid(res: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info and info.is_amino_acid())


def element_of(atom: gemmi.Atom) -> str:
    """Element symbol, upper-case. Falls back to the first letter of the atom name,
    exactly as the prototype did, for files that leave the element column blank."""
    el = atom.element.name.strip().upper()
    if el and el != "X":
        return el
    name = atom.name.strip().upper()
    return name[0] if name else "C"


def read_model(path: Path, model_index: int) -> gemmi.Model:
    """First (or ``--model``-th) model of a PDB/mmCIF file."""
    if not path.is_file():
        raise FileNotFoundError(f"structure missing: {path}")
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if len(st) == 0:
        raise ValueError(f"no models in {path}")
    if model_index < 0 or model_index >= len(st):
        return st[0]
    return st[model_index]


def protein_chain_names(model: gemmi.Model) -> list[str]:
    """Names of every chain holding at least one amino acid, in FILE ORDER.

    File order is what ``--from-chain auto`` means: the first chain donates the
    C-terminus, the second accepts the N-terminus.
    """
    seen: list[str] = []
    for chain in model:
        if chain.name not in seen and any(_is_amino_acid(r) for r in chain):
            seen.append(chain.name)
    return seen


def all_chain_names(model: gemmi.Model) -> list[str]:
    """Every chain name in file order, protein or not -- what ``--obstacle-chains
    all`` resolves to before the water/hydrogen/ligand filters are applied."""
    seen: list[str] = []
    for chain in model:
        if chain.name not in seen:
            seen.append(chain.name)
    return seen


@dataclass
class Obstacles:
    """The obstacle atom cloud, plus enough bookkeeping to exclude a residue."""

    coords: np.ndarray  # (N, 3)
    radii: np.ndarray  # (N,) vdW radius per atom
    owner: list[tuple[str, int, str]]  # (chain, resnum, icode) per atom
    chains: list[str]  # chains that actually contributed an atom


def collect_obstacles(
    model: gemmi.Model,
    chain_names: list[str],
    include_h: bool,
    protein_only: bool,
) -> Obstacles:
    """Every atom a linker has to route around, from the named chains.

    Waters are never obstacles -- a crystallographic water is not something a chain
    has to go around. ``protein_only`` drops everything that is not an amino acid
    (ligands, cofactors, ions, nucleic acids), which makes the route *shorter*, so it
    is off by default: a cofactor in the way is in the way.
    """
    coords: list[list[float]] = []
    radii: list[float] = []
    owner: list[tuple[str, int, str]] = []
    used: list[str] = []
    wanted = set(chain_names)
    for chain in model:
        if chain.name not in wanted:
            continue
        contributed = False
        for res in chain:
            if res.name.strip().upper() in WATERS:
                continue
            if protein_only and not _is_amino_acid(res):
                continue
            key = (chain.name, res.seqid.num, res.seqid.icode.strip())
            for atom in res:
                el = element_of(atom)
                if el == "H" and not include_h:
                    continue
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                radii.append(VDW.get(el, DEFAULT_VDW))
                owner.append(key)
                contributed = True
        if contributed and chain.name not in used:
            used.append(chain.name)
    if not coords:
        raise ValueError(
            f"no obstacle atoms collected from chain(s) {','.join(chain_names)} "
            f"-- check --obstacle-chains / --protein-only"
        )
    return Obstacles(
        np.asarray(coords, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        owner,
        used,
    )


def resolve_obstacle_chains(model: gemmi.Model, spec: str, src: str) -> list[str]:
    """``--obstacle-chains`` -> an explicit chain list.

    ``all`` means every chain in the file. A chain named but absent is an error for
    this design, never a smaller selection: a silently narrower obstacle set makes
    the route *shorter*, i.e. fails in the direction that looks like a good design.
    """
    present = all_chain_names(model)
    if spec.strip().lower() == "all":
        return present
    wanted = [tok.strip() for tok in spec.split(",") if tok.strip()]
    if not wanted:
        raise ValueError("--obstacle-chains must name at least one chain, or 'all'")
    if missing := [c for c in wanted if c not in present]:
        raise ValueError(
            f"--obstacle-chains {missing} absent from {src} "
            f"(chains present: {','.join(present) or 'none'})"
        )
    # De-duplicated, in the order the file lists them, so the recorded column is
    # comparable between designs regardless of how the flag was written.
    return [c for c in present if c in set(wanted)]


def ordered_residues(model: gemmi.Model, chain_id: str, src: str) -> list[gemmi.Residue]:
    """Amino acids of one chain that have a CA, in file order."""
    names = [c.name for c in model]
    if chain_id not in names:
        raise ValueError(
            f"chain {chain_id!r} not found in {src} "
            f"(chains present: {','.join(names) or 'none'})"
        )
    out: list[gemmi.Residue] = []
    for chain in model:
        if chain.name != chain_id:
            continue
        for res in chain:
            if _is_amino_acid(res) and res.find_atom("CA", "*") is not None:
                out.append(res)
    if not out:
        raise ValueError(f"chain {chain_id!r} in {src} contains no amino-acid residues")
    return out


def pick_residue(
    residues: list[gemmi.Residue], chain_id: str, which: str, resnum: int | None
) -> gemmi.Residue:
    """``which='C'`` -> the last residue, ``'N'`` -> the first, unless ``resnum``
    names one explicitly."""
    if resnum is not None:
        for r in residues:
            if r.seqid.num == resnum:
                return r
        raise ValueError(
            f"residue {resnum} not found in chain {chain_id} "
            f"(amino acids present: {residues[0].seqid.num}..{residues[-1].seqid.num})"
        )
    return residues[-1] if which == "C" else residues[0]


def anchor_atom(res: gemmi.Residue, which: str) -> gemmi.Atom:
    """The backbone atom a linker actually grows from (C) or into (N)."""
    for name in (["C", "CA"] if which == "C" else ["N", "CA"]):
        atom = res.find_atom(name, "*")
        if atom is not None:
            return atom
    raise ValueError(f"residue {res.name}{res.seqid.num} lacks backbone atoms (C/CA/N)")


def res_label(chain_id: str, res: gemmi.Residue) -> str:
    """``'A:111'`` -- the chain:resnum[icode] form every tool in this campaign
    parses (ifacegeom, dimerfit). Not the prototype's ``A/ALA111``."""
    return f"{chain_id}:{res.seqid.num}{res.seqid.icode.strip()}"


def atom_xyz(atom: gemmi.Atom) -> np.ndarray:
    return np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def ca_xyz(res: gemmi.Residue) -> np.ndarray | None:
    atom = res.find_atom("CA", "*")
    return None if atom is None else atom_xyz(atom)


# ---------------------------------------------------------------------------
# Occupancy grid  (verbatim from the prototype, numpy-vectorised per atom)
# ---------------------------------------------------------------------------


def build_grid(
    coords: np.ndarray, radii: np.ndarray, spacing: float, probe: float, pad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = coords.min(axis=0) - pad
    maxs = coords.max(axis=0) + pad
    shape = np.ceil((maxs - mins) / spacing).astype(int) + 1
    blocked = np.zeros(tuple(shape), dtype=bool)

    for c, r in zip(coords, radii):
        rr = r + probe
        lo = np.floor((c - rr - mins) / spacing).astype(int)
        hi = np.ceil((c + rr - mins) / spacing).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, shape - 1)
        if np.any(lo > hi):
            continue
        xs = mins[0] + np.arange(lo[0], hi[0] + 1) * spacing - c[0]
        ys = mins[1] + np.arange(lo[1], hi[1] + 1) * spacing - c[1]
        zs = mins[2] + np.arange(lo[2], hi[2] + 1) * spacing - c[2]
        d2 = xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2
        blocked[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] |= d2 <= rr * rr

    return blocked, mins, shape


def carve_sphere(
    blocked: np.ndarray,
    mins: np.ndarray,
    spacing: float,
    centre: np.ndarray,
    radius: float,
) -> None:
    """Free a small bubble so the chain can leave its anchor atom.

    Without it the start voxel is blocked by the anchor atom itself -- the anchor is
    part of the protein, so it is always inside its own (vdW + probe) shell.
    """
    shape = np.array(blocked.shape)
    lo = np.maximum(np.floor((centre - radius - mins) / spacing).astype(int), 0)
    hi = np.minimum(np.ceil((centre + radius - mins) / spacing).astype(int), shape - 1)
    if np.any(lo > hi):
        return
    xs = mins[0] + np.arange(lo[0], hi[0] + 1) * spacing - centre[0]
    ys = mins[1] + np.arange(lo[1], hi[1] + 1) * spacing - centre[1]
    zs = mins[2] + np.arange(lo[2], hi[2] + 1) * spacing - centre[2]
    d2 = xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2
    sub = blocked[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
    sub[d2 <= radius * radius] = False


def nearest_free(
    blocked: np.ndarray,
    mins: np.ndarray,
    spacing: float,
    point: np.ndarray,
    max_r: float = 12.0,
) -> tuple[int, int, int] | None:
    """Closest free voxel to ``point``, or None within ``max_r``."""
    shape = np.array(blocked.shape)
    base = np.round((point - mins) / spacing).astype(int)
    base = np.clip(base, 0, shape - 1)
    if not blocked[tuple(base)]:
        return tuple(int(v) for v in base)  # type: ignore[return-value]
    max_steps = int(math.ceil(max_r / spacing))
    for radius in range(1, max_steps + 1):
        lo = np.maximum(base - radius, 0)
        hi = np.minimum(base + radius, shape - 1)
        sub = blocked[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1]
        free_idx = np.argwhere(~sub)
        if free_idx.size:
            cand = free_idx + lo
            pos = mins + cand * spacing
            d = np.linalg.norm(pos - point, axis=1)
            return tuple(int(v) for v in cand[int(np.argmin(d))])  # type: ignore[return-value]
    return None


def polyline_clearance(
    points: list[np.ndarray],
    coords: np.ndarray,
    radii: np.ndarray,
    anchors: list[np.ndarray],
    carve: float,
    step: float = 0.25,
) -> float | None:
    """Smallest distance from the route to any ATOM SURFACE, in A, or None.

    Densely samples the polyline (not just its vertices, which are free voxel centres
    and therefore clear by construction) and returns ``min(|p - atom| - vdW)``.

    Read it against ``probe``: a route that honestly stays in solvent-accessible
    space never comes closer to an atom than the probe radius. A value well below
    ``probe`` means the 26-connected grid **cut a corner or tunnelled through a thin
    wall** between two free voxel centres -- which makes ``path_dist`` too SHORT,
    i.e. wrong in the direction that looks like a good design.

    Only segments with BOTH endpoints outside every ``carve`` bubble are measured.
    ``carve_sphere`` deliberately frees those bubbles so the chain can leave its own
    terminus, so inside them the route is *expected* to be inside the protein -- and
    the segment that crosses the boundary inherits that. Measured on the C2 barnase
    dimer: without this exclusion every route scores about -1.4 A and the metric says
    nothing at all; excluding samples (rather than whole segments) still leaks the
    boundary-crossing segment and scores 0.7 A.
    """
    if len(points) < 2 or coords.size == 0:
        return None
    verts = np.asarray(points)
    if anchors:
        anc = np.asarray(anchors)
        delta = verts[:, None, :] - anc[None, :, :]
        outside = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)).min(axis=1) > carve
    else:
        outside = np.ones(verts.shape[0], dtype=bool)

    samples: list[np.ndarray] = []
    for i in range(verts.shape[0] - 1):
        if not (outside[i] and outside[i + 1]):
            continue
        a, b = verts[i], verts[i + 1]
        seg = b - a
        n = max(int(math.ceil(float(np.linalg.norm(seg)) / step)), 1)
        for t in range(n + 1):
            samples.append(a + (t / n) * seg)
    if not samples:
        return None
    pts = np.asarray(samples)
    best = np.inf
    for start in range(0, pts.shape[0], 256):
        block = pts[start : start + 256]
        delta = block[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)) - radii[None, :]
        best = min(best, float(dist.min()))
    return best


def segment_is_clear(
    p0: np.ndarray,
    p1: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    probe: float,
    exclude: np.ndarray,
) -> bool:
    """Is the straight anchor-to-anchor vector clear of the protein body?

    Analytic point-to-segment distance against every obstacle atom -- no grid, no
    sampling. ``exclude`` masks out the two ANCHOR RESIDUES' own atoms, which the
    segment necessarily starts and ends inside; without that exclusion this would be
    False for every design (the prototype's grid-sampled ``line_is_clear`` has
    exactly that defect, measured, and this is the one place linkpath deliberately
    departs from it).
    """
    keep = ~exclude
    if not keep.any():
        return True
    pts, rads = coords[keep], radii[keep]
    seg = p1 - p0
    length2 = float(seg @ seg)
    if length2 <= 0.0:
        return True
    t = np.clip(((pts - p0) @ seg) / length2, 0.0, 1.0)
    closest = p0[None, :] + t[:, None] * seg[None, :]
    delta = pts - closest
    dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return bool(np.all(dist >= rads + probe))


# ---------------------------------------------------------------------------
# A* over free voxels  (verbatim from the prototype)
# ---------------------------------------------------------------------------


def astar(
    blocked: np.ndarray,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    spacing: float,
) -> tuple[list[tuple[int, int, int]] | None, float | None]:
    """26-connected A* with true Euclidean edge costs and the (admissible) Euclidean
    straight-line heuristic. Returns ``(voxel path, length)`` or ``(None, None)``."""
    nx, ny, nz = blocked.shape
    nyz = ny * nz
    total = nx * nyz
    flat_blocked = blocked.reshape(-1)

    offsets = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                cost = math.sqrt(di * di + dj * dj + dk * dk) * spacing
                offsets.append((di, dj, dk, di * nyz + dj * nz + dk, cost))

    s = start[0] * nyz + start[1] * nz + start[2]
    g_idx = goal[0] * nyz + goal[1] * nz + goal[2]
    gi, gj, gk = goal

    g_score = np.full(total, np.inf, dtype=np.float32)
    came = np.full(total, -1, dtype=np.int64)
    closed = np.zeros(total, dtype=bool)
    g_score[s] = 0.0

    def h(i: int, j: int, k: int) -> float:
        return math.sqrt((i - gi) ** 2 + (j - gj) ** 2 + (k - gk) ** 2) * spacing

    heap = [(h(*start), s)]
    while heap:
        _f, cur = heapq.heappop(heap)
        if closed[cur]:
            continue
        closed[cur] = True
        if cur == g_idx:
            break
        ci = cur // nyz
        rem = cur - ci * nyz
        cj = rem // nz
        ck = rem - cj * nz
        gc = g_score[cur]
        for di, dj, dk, doff, cost in offsets:
            i, j, k = ci + di, cj + dj, ck + dk
            if i < 0 or j < 0 or k < 0 or i >= nx or j >= ny or k >= nz:
                continue
            nb = cur + doff
            if flat_blocked[nb] or closed[nb]:
                continue
            tentative = gc + cost
            if tentative < g_score[nb]:
                g_score[nb] = tentative
                came[nb] = cur
                heapq.heappush(heap, (tentative + h(i, j, k), nb))

    if not np.isfinite(g_score[g_idx]):
        return None, None

    path = []
    cur = g_idx
    while cur != -1:
        ci = cur // nyz
        rem = cur - ci * nyz
        cj = rem // nz
        ck = rem - cj * nz
        path.append((int(ci), int(cj), int(ck)))
        if cur == s:
            break
        cur = came[cur]
    path.reverse()
    return path, float(g_score[g_idx])


# ---------------------------------------------------------------------------
# Routing one terminus pair
# ---------------------------------------------------------------------------


@dataclass
class Route:
    """One anchor-to-anchor routing attempt."""

    straight: float
    dist: float | None  # None = no free path at this probe/spacing
    points: list[np.ndarray]  # anchor + voxel centres + anchor (what gets written)
    grid_points: list[np.ndarray]  # the A* part alone (what clearance is measured on)
    anchors: list[np.ndarray]  # the two anchor atoms, i.e. the carve-bubble centres
    carve: float  # the EFFECTIVE bubble radius actually used (see effective_carve)
    note: str


def effective_carve(carve: float, probe: float, anchor_vdw: float) -> float:
    """The carve radius actually used, which is never smaller than the anchor atom's
    own probe shell.

    An anchor atom sits at the CENTRE of its bubble and blocks a sphere of radius
    ``vdW + probe`` around itself. If ``--carve`` is smaller than that, the carved
    bubble is wrapped in a complete, unbroken blocked shell of thickness
    ``vdW + probe - carve`` -- and with the campaign defaults (carve 3.0, probe 2.0,
    backbone C at 1.70) that shell is 0.70 A thick around EVERY anchor. The route can
    then only escape by a 26-connected step jumping the shell, which happens at
    ``--spacing 1.0`` and fails at 0.5: measured on a C2 barnase dimer, spacing 1.0
    reported a 57.3 A route and spacing 0.5 reported no path at all, for the same
    structure.

    That shell is a pure artifact: a linker is covalently bonded to the anchor atom,
    so the anchor cannot be the thing blocking the linker's exit. Growing the bubble
    just past it removes the artifact without touching anything else about the grid.
    """
    return max(carve, anchor_vdw + probe + 1e-6)


def route(
    blocked0: np.ndarray,
    mins: np.ndarray,
    spacing: float,
    carve: float,
    p_start: np.ndarray,
    p_goal: np.ndarray,
) -> Route:
    """A* from ``p_start`` to ``p_goal`` on a private copy of the occupancy grid.

    The copy matters: ``--both-directions`` routes a second pair of anchors, and the
    carve bubbles of the first must not free space for the second.
    """
    straight = float(np.linalg.norm(p_goal - p_start))
    blocked = blocked0.copy()
    carve_sphere(blocked, mins, spacing, p_start, carve)
    carve_sphere(blocked, mins, spacing, p_goal, carve)

    s_vox = nearest_free(blocked, mins, spacing, p_start)
    g_vox = nearest_free(blocked, mins, spacing, p_goal)
    if s_vox is None or g_vox is None:
        return Route(
            straight,
            None,
            [],
            [],
            [p_start, p_goal],
            carve,
            "an anchor could not be placed in free space; raise --carve or lower "
            "--probe",
        )

    voxels, grid_len = astar(blocked, s_vox, g_vox, spacing)
    if voxels is None or grid_len is None:
        return Route(straight, None, [], [], [p_start, p_goal], carve, "")

    pts = [mins + np.array(v, dtype=np.float64) * spacing for v in voxels]
    # Exactly the prototype's `total`: the two short hops from each anchor atom to
    # its first free voxel, plus the voxel path between them.
    dist = (
        float(np.linalg.norm(pts[0] - p_start))
        + grid_len
        + float(np.linalg.norm(p_goal - pts[-1]))
    )
    return Route(
        straight, dist, [p_start, *pts, p_goal], pts, [p_start, p_goal], carve, ""
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_path_pdb(
    filename: Path, forward: list[np.ndarray], reverse: list[np.ndarray]
) -> None:
    """The route(s) as a chain of connected pseudo-atoms, for PyMOL.

    Chain X is the forward route, chain Y the reverse one (``--both-directions``).
    Load it alongside the structure and the detour is visible directly.
    """
    with open(filename, "w") as fh:
        serial = 0
        for chain_id, points in (("X", forward), ("Y", reverse)):
            first = serial + 1
            for p in points:
                serial += 1
                fh.write(
                    "HETATM%5d  C   PTH %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                    % (serial, chain_id, min(serial, 9999), p[0], p[1], p[2])
                )
            for n in range(first, serial):
                fh.write("CONECT%5d%5d\n" % (n, n + 1))
        fh.write("END\n")


def _fmt(value: object) -> str:
    """TSV cell: NA (None) becomes empty, floats get 3 decimals."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


@dataclass
class Design:
    """One sub-manifest line."""

    name: str
    structure: Path
    from_res: int | None  # None = the chain's own terminus
    to_res: int | None


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def compute(
    args: "Args", design: Design
) -> tuple[str, dict[str, object], list[np.ndarray], list[np.ndarray]]:
    started = time.time()
    src = str(design.structure)
    model = read_model(design.structure, args.model)

    # --- chains --------------------------------------------------------------
    protein = protein_chain_names(model)
    from_chain, to_chain = args.from_chain.strip(), args.to_chain.strip()
    if from_chain.lower() == "auto" or to_chain.lower() == "auto":
        if len(protein) != AUTO_CHAIN_COUNT:
            raise ValueError(
                f"--from-chain/--to-chain 'auto' needs exactly {AUTO_CHAIN_COUNT} "
                f"protein chains; {src} holds {len(protein)} "
                f"({','.join(protein) or 'none'}). Name the two chains explicitly -- "
                f"guessing a terminus pair here would answer a question nobody asked."
            )
        if from_chain.lower() == "auto":
            from_chain = protein[0]
        if to_chain.lower() == "auto":
            to_chain = protein[1]

    # --- obstacles -----------------------------------------------------------
    obstacle_chains = resolve_obstacle_chains(model, args.obstacle_chains, src)
    obstacles = collect_obstacles(
        model, obstacle_chains, args.include_h, args.protein_only
    )
    coords, radii = obstacles.coords, obstacles.radii

    # --- anchors -------------------------------------------------------------
    from_residues = ordered_residues(model, from_chain, src)
    to_residues = ordered_residues(model, to_chain, src)
    from_res = pick_residue(from_residues, from_chain, args.from_end, design.from_res)
    to_res = pick_residue(to_residues, to_chain, args.to_end, design.to_res)
    from_atom = anchor_atom(from_res, args.from_end)
    to_atom = anchor_atom(to_res, args.to_end)
    p_start, p_goal = atom_xyz(from_atom), atom_xyz(to_atom)
    straight = float(np.linalg.norm(p_goal - p_start))
    if straight <= 0.0:
        raise ValueError(
            f"the two anchors are the same atom ({res_label(from_chain, from_res)} "
            f"{from_atom.name}); there is nothing to span"
        )

    ca_from, ca_to = ca_xyz(from_res), ca_xyz(to_res)
    straight_ca = (
        float(np.linalg.norm(ca_to - ca_from))
        if ca_from is not None and ca_to is not None
        else None
    )

    # Is the direct vector unobstructed? The two anchor residues' own atoms are
    # excluded -- see segment_is_clear.
    anchor_keys = {
        (from_chain, from_res.seqid.num, from_res.seqid.icode.strip()),
        (to_chain, to_res.seqid.num, to_res.seqid.icode.strip()),
    }
    anchor_atoms = np.fromiter(
        (key in anchor_keys for key in obstacles.owner), dtype=bool, count=len(obstacles.owner)
    )
    direct_clear = segment_is_clear(
        p_start, p_goal, coords, radii, args.probe, anchor_atoms
    )

    # --- grid ----------------------------------------------------------------
    blocked0, mins, shape = build_grid(coords, radii, args.spacing, args.probe, args.pad)

    # One effective bubble radius for both anchors: the larger of the two, so neither
    # anchor atom can seal itself in. See effective_carve.
    carve_eff = max(
        effective_carve(args.carve, args.probe, VDW.get(element_of(a), DEFAULT_VDW))
        for a in (from_atom, to_atom)
    )
    forward = route(blocked0, mins, args.spacing, carve_eff, p_start, p_goal)

    notes: list[str] = []
    if carve_eff > args.carve + 1e-6:
        # Logged, not noted: with the campaign defaults this fires on every design
        # (3.0 -> 3.70 A for a backbone C at probe 2.0), and a note that is always
        # there would drown the per-design findings `note` exists to carry. It is
        # fully determined by the echoed `probe` and `from_atom`/`to_atom`.
        print(
            f"{design.name}: carve grown {args.carve:g} -> {carve_eff:.2f} A so the "
            f"anchor atom's own probe shell does not seal it in"
        )
    if forward.note:
        notes.append(forward.note)
    elif forward.dist is None:
        notes.append(f"no free path at probe {args.probe:g} / spacing {args.spacing:g}")

    # --- the reverse route: a free trust metric under C2 ----------------------
    # Always C-terminus of the TO chain -> N-terminus of the FROM chain, using each
    # chain's own terminus (never --from-res/--to-res, which describe the forward
    # pair). Under exact C2 symmetry the two routes must be equal, so their
    # difference bounds the discretisation error and catches a broken frame.
    rev_dist: float | None = None
    rev_points: list[np.ndarray] = []
    rev_grid_points: list[np.ndarray] = []
    rev_anchors: list[np.ndarray] = []
    if args.both_directions:
        if from_chain == to_chain:
            notes.append("reverse route skipped: from-chain == to-chain")
        else:
            rev_from = pick_residue(to_residues, to_chain, "C", None)
            rev_to = pick_residue(from_residues, from_chain, "N", None)
            reverse = route(
                blocked0,
                mins,
                args.spacing,
                carve_eff,
                atom_xyz(anchor_atom(rev_from, "C")),
                atom_xyz(anchor_atom(rev_to, "N")),
            )
            rev_dist = reverse.dist
            rev_points = reverse.points
            rev_grid_points = reverse.grid_points
            rev_anchors = reverse.anchors
            if reverse.note:
                notes.append(f"reverse: {reverse.note}")
            elif reverse.dist is None:
                notes.append(
                    f"no free reverse path at probe {args.probe:g} / spacing "
                    f"{args.spacing:g}"
                )

    path_found = forward.dist is not None

    # How close the route really comes to the protein, measured on the CONTINUOUS
    # polyline rather than on voxel centres (which are free by construction). This is
    # the tunnelling check: see polyline_clearance.
    clearances = [
        c
        for c in (
            polyline_clearance(
                forward.grid_points, coords, radii, forward.anchors, carve_eff
            ),
            polyline_clearance(rev_grid_points, coords, radii, rev_anchors, carve_eff),
        )
        if c is not None
    ]
    clearance = min(clearances) if clearances else None

    # The invariant. A polyline between the anchors can never be shorter than the
    # straight line; if it is, the grid or the bookkeeping is wrong and every number
    # here is worthless, so this is an error row rather than a quietly short answer.
    if forward.dist is not None and forward.dist < straight - INVARIANT_TOL:
        raise ValueError(
            f"invariant violated: path_dist {forward.dist:.3f} < straight_dist "
            f"{straight:.3f} (grid {shape[0]}x{shape[1]}x{shape[2]} @ "
            f"{args.spacing} A). The occupancy grid or the path bookkeeping is broken."
        )
    if rev_dist is not None:
        rev_straight = (
            float(np.linalg.norm(rev_points[-1] - rev_points[0])) if rev_points else 0.0
        )
        if rev_dist < rev_straight - INVARIANT_TOL:
            raise ValueError(
                f"invariant violated on the reverse route: {rev_dist:.3f} < "
                f"{rev_straight:.3f}"
            )

    metrics: dict[str, object] = {
        "from_res": res_label(from_chain, from_res),
        "to_res": res_label(to_chain, to_res),
        "from_atom": from_atom.name,
        "to_atom": to_atom.name,
        "from_chain": from_chain,
        "to_chain": to_chain,
        # The RESOLVED list, never the literal 'all': this is the auditable answer to
        # "was the target in there?".
        "obstacle_chains": ",".join(obstacles.chains),
        "n_obstacle_atoms": int(coords.shape[0]),
        "probe": float(args.probe),
        "spacing": float(args.spacing),
        "grid_shape": f"{int(shape[0])}x{int(shape[1])}x{int(shape[2])}",
        "direct_clear": bool(direct_clear),
        "straight_dist": straight,
        "straight_ca_dist": straight_ca,
        "path_dist": forward.dist,
        "detour_ratio": (forward.dist / straight) if forward.dist is not None else None,
        "path_found": path_found,
        "min_clearance": clearance,
        "rev_path_dist": rev_dist,
        "path_asymmetry": (
            abs(forward.dist - rev_dist)
            if forward.dist is not None and rev_dist is not None
            else None
        ),
        # The counts and their rise constants are written together or not at all --
        # see the comment on METRIC_COLUMNS.
        "n_res_min": None,
        "taut_rise": None,
        "n_res_relaxed": None,
        "relaxed_rise": None,
        "note": "; ".join(notes),
        "seconds": time.time() - started,
    }
    if args.residue_estimate and forward.dist is not None:
        metrics["n_res_min"] = int(math.ceil(forward.dist / args.taut_rise))
        metrics["taut_rise"] = float(args.taut_rise)
        metrics["n_res_relaxed"] = int(math.ceil(forward.dist / args.relaxed_rise))
        metrics["relaxed_rise"] = float(args.relaxed_rise)

    # 'no free path' is a RESULT (the design is unlinkable at this probe), not a
    # failure: the status stays OK so a standard `== "OK"` trust filter does not
    # silently delete the finding. Read path_found, not the status.
    #
    # A route that passed measurably closer to an atom than the probe allows is a
    # different matter: the 26-connected grid cut a corner or tunnelled through a
    # thin wall between two free voxel centres, so path_dist is too SHORT -- the
    # direction that looks like a good design. Every metric is still written (so the
    # suspicion can be judged from the table), but the 'warn:' prefix keeps the row
    # out of any `== "OK"` selection, exactly as dimerfit does.
    status = "OK"
    if clearance is not None and clearance < args.probe - CLEARANCE_SLACK:
        status = (
            f"warn: route clearance {clearance:.2f} A < probe {args.probe:g} A "
            f"(grid tunnelling at spacing {args.spacing:g}; lower --spacing)"
        )
    return status, metrics, forward.points, rev_points


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_result(
    result_tsv: Path, name: str, status: str, path: str, metrics: dict
) -> None:
    result_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(result_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(RESULT_COLUMNS)
        writer.writerow(
            [name, status, path] + [_fmt(metrics.get(c)) for c in METRIC_COLUMNS]
        )


def _parse_resnum(token: str, what: str) -> int | None:
    """``'-'`` (the sentinel for "the chain's own terminus") -> None."""
    token = token.strip()
    if token in ("", "-"):
        return None
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"{what} must be an integer or '-', got {token!r}") from exc


def read_task_file(path: Path, expect: int | None) -> list[Design]:
    """``name<TAB>structure<TAB>from_res<TAB>to_res`` lines -> designs.

    ``expect`` is the design count the manifest said this task carries. A mismatch is
    FATAL: a half-visible sub-manifest on a Volume reads as a short task rather than
    as an error, and that is exactly how 18 of 37 rpxdock tasks once ran with empty
    inputs (see the guard in linkpath.sh).
    """
    if not path.is_file():
        raise FileNotFoundError(f"task file missing: {path}")
    designs: list[Design] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(
                f"malformed task line {lineno} in {path}: expected 4 tab-separated "
                f"fields (name, structure, from_res, to_res), got {len(fields)}: "
                f"{line!r}"
            )
        name, structure, from_res, to_res = fields
        if not name or not structure:
            raise ValueError(f"task line {lineno} in {path} has an empty field: {line!r}")
        designs.append(
            Design(
                name,
                Path(structure),
                _parse_resnum(from_res, "--from-res"),
                _parse_resnum(to_res, "--to-res"),
            )
        )
    if expect is not None and len(designs) != expect:
        raise ValueError(
            f"{path} holds {len(designs)} design(s) but the manifest says this task "
            f"carries {expect}. Refusing to run on a partially visible task file."
        )
    return designs


class Args(argparse.Namespace):
    task_file: Path
    n_designs: int
    from_chain: str
    to_chain: str
    from_end: str
    to_end: str
    obstacle_chains: str
    include_h: bool
    protein_only: bool
    spacing: float
    probe: float
    pad: float
    carve: float
    taut_rise: float
    relaxed_rise: float
    both_directions: bool
    residue_estimate: bool
    model: int
    out_dir: Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shortest solvent-accessible route between two chain termini, "
        "for a batch of designs."
    )
    ap.add_argument(
        "--task-file",
        type=Path,
        required=True,
        help="Sub-manifest: one 'name<TAB>structure<TAB>from_res<TAB>to_res' line per "
        "design ('-' = the chain's own terminus).",
    )
    ap.add_argument(
        "--n-designs",
        type=int,
        default=-1,
        help="Design count the manifest says this task carries; -1 skips the check.",
    )
    ap.add_argument("--from-chain", default="auto")
    ap.add_argument("--to-chain", default="auto")
    ap.add_argument("--from-end", choices=("C", "N"), default="C")
    ap.add_argument("--to-end", choices=("C", "N"), default="N")
    ap.add_argument("--obstacle-chains", default="all")
    ap.add_argument("--include-h", action="store_true")
    ap.add_argument("--protein-only", action="store_true")
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--probe", type=float, default=2.0)
    ap.add_argument("--pad", type=float, default=15.0)
    ap.add_argument("--carve", type=float, default=3.0)
    ap.add_argument("--taut-rise", type=float, default=3.5)
    ap.add_argument("--relaxed-rise", type=float, default=2.1)
    ap.add_argument(
        "--both-directions",
        dest="both_directions",
        action="store_true",
        default=True,
    )
    ap.add_argument("--no-both-directions", dest="both_directions", action="store_false")
    ap.add_argument(
        "--residue-estimate",
        dest="residue_estimate",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-residue-estimate", dest="residue_estimate", action="store_false"
    )
    ap.add_argument("--model", type=int, default=0)
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Where the per-design files go."
    )
    args = ap.parse_args(namespace=Args())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    designs = read_task_file(
        args.task_file, None if args.n_designs < 0 else args.n_designs
    )

    failures = 0
    for design in designs:
        result_tsv = args.out_dir / f"{design.name}.tsv"
        path_pdb = args.out_dir / f"{design.name}_path.pdb"
        try:
            status, metrics, forward, reverse = compute(args, design)
            if forward:
                write_path_pdb(path_pdb, forward, reverse)
            else:
                path_pdb.unlink(missing_ok=True)
            print(
                f"{design.name}: {status} -- {metrics['from_res']} "
                f"({metrics['from_atom']}) -> {metrics['to_res']} "
                f"({metrics['to_atom']}), straight {_fmt(metrics['straight_dist'])} A, "
                f"path {_fmt(metrics['path_dist'])} A "
                f"(x{_fmt(metrics['detour_ratio'])}), direct_clear "
                f"{metrics['direct_clear']}, n_res_relaxed "
                f"{_fmt(metrics['n_res_relaxed'])}, asymmetry "
                f"{_fmt(metrics['path_asymmetry'])}"
                + (f" [{metrics['note']}]" if metrics["note"] else "")
            )
            write_result(
                result_tsv,
                design.name,
                status,
                str(path_pdb) if forward else "",
                metrics,
            )
        except Exception as e:  # noqa: BLE001 - errors are recorded as data
            failures += 1
            print(f"{design.name}: ERROR {e}", file=sys.stderr)
            # No partial file left behind: a half-written path would be worse than
            # none, since the column would point at something unusable.
            path_pdb.unlink(missing_ok=True)
            write_result(result_tsv, design.name, f"error: {e}", "", {})

    # Every design has a result either way; exiting non-zero would only hide that
    # behind a task-level failure, so the batch succeeds and the table carries the
    # errors.
    print(f"linkpath: {failures} failed of {len(designs)}")


if __name__ == "__main__":
    main()
