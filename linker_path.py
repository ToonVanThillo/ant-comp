#!/usr/bin/env python3
"""
linker_path.py -- SUPERSEDED PROTOTYPE. DO NOT RUN. Use the `linkpath` tool.

    sapia run linkpath <run_dir> -t <table> -i <structure_column>

Kept only as the origin of that tool. Two numerical defects were found in this file
while porting it, both confirmed against a known answer -- see
`campaigns/20261002_egfr_ph_dimer_binder.md` section 5:

1. ANCHOR SEALED IN A BLOCKED SHELL. The anchor atom sits at the centre of the
   bubble `--carve` frees, but it still blocks a sphere of (vdW + probe) around
   itself. With carve 3.0 / probe 2.0 / backbone C vdW 1.70 the bubble is wrapped
   in a complete 0.70 A blocked shell, and the route escapes only through a grid
   artifact. The reported length therefore moves with --spacing and gets WORSE as
   the grid gets finer: measured 57.29 A at spacing 1.0 and NO PATH at 0.5 on the
   same structure. On a free-space pair whose answer is exactly 27.10 A this file
   returns 27.93 A (probe 1.4) / 29.39 A (probe 2.0); `linkpath` returns 27.10 A.
2. `direct path clear` IS ALWAYS "obstructed". `line_is_clear` samples the straight
   segment including the two anchor residues' own atoms, which the segment starts
   and ends inside. Verified: it reports obstructed for two residues alone in empty
   space.

Are old numbers from this script usable? Lengths yes, verdicts no.

  * A recorded LENGTH is a safe UPPER BOUND, never short. Correcting defect 1 only
    ever frees voxels, so the corrected free set is a strict superset of the buggy
    one and the start/goal voxels are identical -- the corrected optimum is a
    minimum over a superset of the same paths. Measured over 72 paired runs
    (6 structures x 4 spacings x 3 probes): 25 inflated by up to 4.76 A, 20
    identical, and ZERO cases where the corrected route came back longer. So if an
    old number cleared your linker budget it still clears it; if it failed, it may
    be a false reject by up to ~5 A. The inflation is NOT a constant you can
    subtract (0.00-4.76 A, depending on local geometry at the anchor).
  * A recorded "No free path found" is WORTHLESS. 15 of those same 72 runs reported
    no route where one exists, and it gets worse as the grid gets finer or the probe
    larger -- every --spacing 0.5 --probe 2.0 case in the sweep was a false negative.
    Never conclude a design is unlinkable from a run of this script.

--- original docstring follows ---

obstruction-aware C-to-N distance for fusion-linker design.

A straight-line C(term) -> N(term) distance is only a lower bound: if the
direct vector passes through the protein, the linker has to go around.
This script computes the shortest path that stays in solvent-accessible
space.

Method
------
1. Build a 3D grid over the structure (+ padding so paths can leave the
   surface).
2. Mark a voxel as BLOCKED if its centre lies within (vdW radius + probe)
   of any atom.  `probe` approximates the thickness of the linker backbone.
3. A* search over the free voxels (26-connected, true Euclidean edge costs)
   from the C-terminal anchor atom to the N-terminal anchor atom.
4. Report the path length and convert it into a residue count.

Residue conversion
------------------
  contour length of a polypeptide ~ 3.5 A / residue (fully extended)
  n_min   = d / 3.5     absolute floor: a taut, strained linker
  n_design = d / 2.1    keeps the chain at <=60% of contour length,
                        i.e. a relaxed coil rather than a stretched one

Usage
-----
  python linker_path.py model.pdb --c-chain A --n-chain B
  python linker_path.py model.cif --c-chain A --n-chain B --path-out path.pdb

Dependencies: numpy, biopython  (scipy optional, not required)
"""

import argparse
import heapq
import math
import os
import sys

import numpy as np

try:
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    sys.exit("Biopython is required:  pip install biopython")


# ----------------------------------------------------------------------
# van der Waals radii (A)
# ----------------------------------------------------------------------
VDW = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.73, "ZN": 1.39, "FE": 2.00, "CA": 2.31, "NA": 2.27,
    "K": 2.75, "MN": 2.05, "CU": 1.40, "NI": 1.63, "CO": 2.00,
}
DEFAULT_VDW = 1.70

WATERS = {"HOH", "WAT", "DOD", "H2O", "TIP", "SOL"}


def element_of(atom):
    el = (atom.element or "").strip().upper()
    if el:
        return el
    name = atom.get_name().strip().upper()
    return name[0] if name else "C"


# ----------------------------------------------------------------------
# Structure handling
# ----------------------------------------------------------------------
def load_structure(path, model_id):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("s", path)
    models = list(structure)
    if not models:
        sys.exit("No models found in %s" % path)
    for m in models:
        if m.get_id() == model_id:
            return m
    return models[0]


def collect_atoms(model, include_h, protein_only, skip_waters=True):
    """Return (coords Nx3, radii N) for every obstructing atom."""
    coords, radii = [], []
    for chain in model:
        for res in chain:
            hetflag, _, _ = res.get_id()
            resname = res.get_resname().strip().upper()
            if skip_waters and resname in WATERS:
                continue
            if protein_only and hetflag.strip():
                continue
            for atom in res:
                if atom.is_disordered():
                    atom = atom.disordered_get()
                el = element_of(atom)
                if el == "H" and not include_h:
                    continue
                coords.append(atom.get_coord())
                radii.append(VDW.get(el, DEFAULT_VDW))
    if not coords:
        sys.exit("No atoms collected -- check the input file / filters.")
    return np.asarray(coords, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def ordered_residues(chain):
    out = []
    for res in chain:
        if is_aa(res, standard=False) and "CA" in res:
            out.append(res)
    return out


def pick_residue(model, chain_id, which, resnum=None):
    """which = 'C' (last residue) or 'N' (first residue)."""
    if chain_id not in [c.get_id() for c in model]:
        sys.exit("Chain '%s' not found. Chains present: %s"
                 % (chain_id, ", ".join(c.get_id() for c in model)))
    chain = model[chain_id]
    residues = ordered_residues(chain)
    if not residues:
        sys.exit("Chain '%s' contains no amino-acid residues." % chain_id)
    if resnum is not None:
        for r in residues:
            if r.get_id()[1] == resnum:
                return r
        sys.exit("Residue %d not found in chain %s." % (resnum, chain_id))
    return residues[-1] if which == "C" else residues[0]


def anchor_atom(res, which):
    """Backbone atom the linker actually grows from / into."""
    prefer = ["C", "CA"] if which == "C" else ["N", "CA"]
    for name in prefer:
        if name in res:
            a = res[name]
            return (a.disordered_get() if a.is_disordered() else a)
    sys.exit("Residue %s lacks backbone atoms." % str(res.get_id()))


def res_label(chain_id, res):
    return "%s/%s%d" % (chain_id, res.get_resname().strip(), res.get_id()[1])


# ----------------------------------------------------------------------
# Occupancy grid
# ----------------------------------------------------------------------
def build_grid(coords, radii, spacing, probe, pad):
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
        d2 = (xs[:, None, None] ** 2 + ys[None, :, None] ** 2
              + zs[None, None, :] ** 2)
        blocked[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] |= (d2 <= rr * rr)

    return blocked, mins, shape


def carve_sphere(blocked, mins, spacing, centre, radius):
    """Free up a small bubble so the chain can leave its anchor atom."""
    shape = np.array(blocked.shape)
    lo = np.maximum(np.floor((centre - radius - mins) / spacing).astype(int), 0)
    hi = np.minimum(np.ceil((centre + radius - mins) / spacing).astype(int), shape - 1)
    if np.any(lo > hi):
        return
    xs = mins[0] + np.arange(lo[0], hi[0] + 1) * spacing - centre[0]
    ys = mins[1] + np.arange(lo[1], hi[1] + 1) * spacing - centre[1]
    zs = mins[2] + np.arange(lo[2], hi[2] + 1) * spacing - centre[2]
    d2 = (xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2)
    sub = blocked[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    sub[d2 <= radius * radius] = False


def nearest_free(blocked, mins, spacing, point, max_r=12.0):
    """Closest free voxel to `point`; returns (i,j,k) or None."""
    shape = np.array(blocked.shape)
    base = np.round((point - mins) / spacing).astype(int)
    base = np.clip(base, 0, shape - 1)
    if not blocked[tuple(base)]:
        return tuple(base)
    max_steps = int(math.ceil(max_r / spacing))
    for radius in range(1, max_steps + 1):
        lo = np.maximum(base - radius, 0)
        hi = np.minimum(base + radius, shape - 1)
        sub = blocked[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        free_idx = np.argwhere(~sub)
        if free_idx.size:
            cand = free_idx + lo
            pos = mins + cand * spacing
            d = np.linalg.norm(pos - point, axis=1)
            return tuple(cand[int(np.argmin(d))])
    return None


def line_is_clear(blocked, mins, spacing, p0, p1):
    """Sample the straight segment; True if it never enters blocked space."""
    d = np.linalg.norm(p1 - p0)
    n = max(int(d / (spacing * 0.5)), 2)
    shape = np.array(blocked.shape)
    for t in np.linspace(0.0, 1.0, n):
        p = p0 + t * (p1 - p0)
        idx = np.round((p - mins) / spacing).astype(int)
        if np.any(idx < 0) or np.any(idx >= shape):
            continue
        if blocked[tuple(idx)]:
            return False
    return True


# ----------------------------------------------------------------------
# A* over free voxels
# ----------------------------------------------------------------------
def astar(blocked, start, goal, spacing):
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

    def h(i, j, k):
        return math.sqrt((i - gi) ** 2 + (j - gj) ** 2 + (k - gk) ** 2) * spacing

    heap = [(h(*start), s)]
    while heap:
        f, cur = heapq.heappop(heap)
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
        path.append((ci, cj, ck))
        if cur == s:
            break
        cur = came[cur]
    path.reverse()
    return path, float(g_score[g_idx])


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def write_path_pdb(filename, points):
    with open(filename, "w") as fh:
        for n, p in enumerate(points, start=1):
            fh.write("HETATM%5d  C   PTH X%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                     % (n, n, p[0], p[1], p[2]))
        for n in range(1, len(points)):
            fh.write("CONECT%5d%5d\n" % (n, n + 1))
        fh.write("END\n")


def main():
    ap = argparse.ArgumentParser(
        description="Obstruction-aware C-to-N distance and linker length estimate.")
    ap.add_argument("structure", help="PDB or mmCIF file")
    ap.add_argument("--c-chain", required=True, help="chain donating the C-terminus")
    ap.add_argument("--n-chain", required=True, help="chain receiving the N-terminus")
    ap.add_argument("--c-res", type=int, default=None,
                    help="residue number to use as C-terminal anchor (default: last)")
    ap.add_argument("--n-res", type=int, default=None,
                    help="residue number to use as N-terminal anchor (default: first)")
    ap.add_argument("--spacing", type=float, default=1.0, help="grid spacing, A (default 1.0)")
    ap.add_argument("--probe", type=float, default=1.4,
                    help="probe radius, A: 1.4 = water, ~2.0 = chain-sized channel")
    ap.add_argument("--pad", type=float, default=15.0,
                    help="padding around the structure, A (default 15)")
    ap.add_argument("--carve", type=float, default=3.0,
                    help="radius freed around each anchor atom, A (default 3.0)")
    ap.add_argument("--model", type=int, default=0, help="model index (default 0)")
    ap.add_argument("--include-h", action="store_true", help="treat hydrogens as obstacles")
    ap.add_argument("--protein-only", action="store_true",
                    help="ignore ligands/cofactors as obstacles")
    ap.add_argument("--path-out", default=None,
                    help="write the path as a PDB of pseudo-atoms for PyMOL")
    args = ap.parse_args()

    model = load_structure(args.structure, args.model)

    c_res = pick_residue(model, args.c_chain, "C", args.c_res)
    n_res = pick_residue(model, args.n_chain, "N", args.n_res)
    c_atom = anchor_atom(c_res, "C")
    n_atom = anchor_atom(n_res, "N")
    p_start = np.asarray(c_atom.get_coord(), dtype=np.float64)
    p_goal = np.asarray(n_atom.get_coord(), dtype=np.float64)

    straight = float(np.linalg.norm(p_goal - p_start))

    coords, radii = collect_atoms(model, args.include_h, args.protein_only)
    blocked, mins, shape = build_grid(coords, radii, args.spacing, args.probe, args.pad)

    clear = line_is_clear(blocked, mins, args.spacing, p_start, p_goal)

    carve_sphere(blocked, mins, args.spacing, p_start, args.carve)
    carve_sphere(blocked, mins, args.spacing, p_goal, args.carve)

    s_vox = nearest_free(blocked, mins, args.spacing, p_start)
    g_vox = nearest_free(blocked, mins, args.spacing, p_goal)
    if s_vox is None or g_vox is None:
        sys.exit("Could not place an anchor in free space -- increase --carve "
                 "or lower --probe.")

    path, grid_len = astar(blocked, s_vox, g_vox, args.spacing)

    print("structure            : %s (model %d)" % (args.structure, args.model))
    print("C-terminal anchor    : %s  atom %s" % (res_label(args.c_chain, c_res),
                                                  c_atom.get_name()))
    print("N-terminal anchor    : %s  atom %s" % (res_label(args.n_chain, n_res),
                                                  n_atom.get_name()))
    print("grid                 : %d x %d x %d voxels @ %.2f A, probe %.2f A"
          % (shape[0], shape[1], shape[2], args.spacing, args.probe))
    print("")
    print("straight-line C->N   : %6.1f A" % straight)
    print("direct path clear    : %s" % ("yes" if clear else "NO - obstructed"))

    if path is None:
        print("")
        print("No free path found. The termini may be in an enclosed pocket;")
        print("try a smaller --probe or a larger --pad.")
        return

    pts = [mins + np.array(v) * args.spacing for v in path]
    total = (np.linalg.norm(pts[0] - p_start)
             + grid_len
             + np.linalg.norm(p_goal - pts[-1]))

    print("accessible path      : %6.1f A  (%.2f x straight line)"
          % (total, total / straight if straight > 0 else float("nan")))
    print("")
    n_min = math.ceil(total / 3.5)
    n_design = math.ceil(total / 2.1)
    print("linker residues")
    print("  minimum (taut)     : %d   (3.5 A/res, fully extended)" % n_min)
    print("  recommended        : %d   (<=60%% of contour length, relaxed coil)" % n_design)
    print("  e.g. GGGGS repeats : %d  (%d residues)"
          % (math.ceil(n_design / 5.0), 5 * math.ceil(n_design / 5.0)))
    print("")
    print("Note: the grid path is a lower bound -- a real chain has volume and")
    print("cannot hug the surface exactly. Add margin, and remember unresolved")
    print("terminal residues already contribute length you may not need to add.")

    if args.path_out:
        write_path_pdb(args.path_out, pts)
        print("")
        print("path written to %s  (load alongside the structure in PyMOL)" % args.path_out)


if __name__ == "__main__":
    main()
