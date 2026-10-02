#!/usr/bin/env python3
"""
Measure where a binder's interface with its target sits, and where the termini sit
relative to it.

This is the per-array-task step of the ifacegeom tool (and works standalone). It
answers two questions about a binder/target complex we did NOT design, and whose
epitope therefore has to be discovered rather than looked up:

1. *Which residues form the interface, on both sides?* The binder-side list is the
   epitope footprint every later step needs -- it is what gets rotamer-grafted back
   onto the dimer and what the H-bond network must avoid overlapping. The
   target-side list carries the histidine count: a target epitope containing
   histidines is one whose contacts change when the pH drops, which is exactly what
   a pH-responsive binder must not rely on.
2. *Where is the C-terminus relative to that interface?* Wet-lab validation fuses a
   GFP/split-strep tag to the binder's C-terminus, so a C-terminus sitting on the
   binding face would put the tag in the epitope. Measured as a signed projection
   onto the binder-COM -> interface-COM axis: the plane through the binder COM
   normal to that axis splits the binder into an interface side (proj > 0) and a
   far side (proj < 0). The N-terminus gets the same treatment, because phase 2
   needs it reachable from the partner monomer's C-terminus across the dimerization
   interface -- the requirement is two-sided, so both numbers are recorded and
   neither is filtered on here.

Interface selection
-------------------
``--method vector`` (default) reproduces the *idea* of Rosetta's
``InterGroupInterfaceByVector``: a residue is at the interface either because it is
in physical contact with the other side, or because its side chain points at it.

    nearby   any heavy atom of i within --contact-cutoff of any heavy atom of j
             -> both i and j are selected
    vector   CB(i)-CB(j) within --vector-dist-cut AND the angle between i's
             CA->CB direction and the CB(i)->CB(j) direction is below
             --vector-angle-cut -> i is selected (and symmetrically for j)

Both tests only consider pairs whose CB atoms are within ``--cb-dist-cut``. The
vector test is applied **per residue**, not per pair, so each side's list stands on
its own: binder residue i is listed because i points at the target, regardless of
whether the target residue happens to point back. Glycine has no CB, so a virtual
one is built from N/CA/C with the standard construction.

This is a reimplementation of the concept, not a port of Rosetta's code -- there is
no PyRosetta in this image and the numbers are not claimed to be bit-identical. The
defaults are Rosetta's documented ones (5.5 / 9.0 / 11.0 A, 75 deg).

``--method heavy`` drops the vector test and keeps the ``nearby`` criterion alone,
which is the plain heavy-atom contact definition.

Writes a one-row TSV (name, status, path, then the metrics) that
collect_ifacegeom.py merges back into the table, plus a per-residue TSV
(``<name>_per_residue.tsv``) listing every residue near the other side with how it
was selected and its distances -- the evidence behind the lists, and what the
epitope can be cross-checked against. Errors are recorded as data (a status
starting with 'error:') rather than only crashing, so partial array runs still
collect.

Normally invoked per array task by the ifacegeom tool -- run ``sapia run ifacegeom``
rather than calling this directly.

Usage (standalone, one task file of name<TAB>structure lines):
    python tools/ifacegeom/ifacegeom_worker.py \\
        --task-file task_0.tsv --binder-chains A --target-chains auto \\
        --out-dir out/
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
import numpy as np

METHODS = ("vector", "heavy")

# Every histidine naming convention we might meet: Rosetta/Amber protonation states
# (HID/HIE/HIP) and the CHARMM spellings (HSD/HSE/HSP). The whole campaign turns on
# counting histidines correctly, so none of these may fall through as 'X'.
HIS_RESNAMES = {"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"}
# Residue names gemmi does not map to a one-letter code but we do.
CODE_ALIASES = {name: "H" for name in HIS_RESNAMES}

# Virtual-CB construction from backbone N, CA, C (the standard trRosetta/AlphaFold
# coefficients). Used for glycine, and for any residue whose CB is missing.
_CB_A, _CB_B, _CB_C = -0.58273431, 0.56802827, -0.54067466

# Metric columns of the per-design TSV, in order (bare names: collect_ifacegeom.py
# hands them to the driver, which leaf-prefixes them to ifacegeom_<name>).
METRIC_COLUMNS = [
    "method",
    "binder_chains",
    "target_chains",
    "binder_len",
    "target_len",
    "n_binder_res",
    "binder_res",
    "binder_res_seq",
    "n_binder_his",
    "binder_his",
    "n_target_res",
    "target_res",
    "n_target_his",
    "target_his",
    "binder_com",
    "binder_iface_com",
    "iface_com",
    "axis_len",
    "nterm_res",
    "cterm_res",
    "nterm_proj",
    "cterm_proj",
    "nterm_iface_dist",
    "cterm_iface_dist",
    "nterm_iface_min_dist",
    "cterm_iface_min_dist",
    "seconds",
]
RESULT_COLUMNS = ["name", "status", "path", *METRIC_COLUMNS]

PER_RESIDUE_COLUMNS = [
    "side",
    "label",
    "chain",
    "resnum",
    "icode",
    "resname",
    "code",
    "selected",
    "selected_by",
    "min_heavy_dist",
    "min_cb_dist",
]


# ---------------------------------------------------------------------------
# Residue bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class Res:
    """One amino acid, reduced to what the geometry needs."""

    label: str  # "A:12" (chain:resnum[icode]) -- the format every later tool parses
    chain: str
    seqid: int
    icode: str
    resname: str
    code: str  # one-letter, 'X' when unknown
    heavy: np.ndarray  # (M, 3) heavy-atom coordinates
    masses: np.ndarray  # (M,) atomic weights, for the centre of mass
    ca: np.ndarray | None  # CA coordinate
    cb: np.ndarray | None  # CB, real or virtual; None when no backbone at all
    # Filled in by the selection pass.
    selected: bool = False
    by: set[str] = field(default_factory=set)
    min_heavy: float | None = None
    min_cb: float | None = None

    @property
    def is_his(self) -> bool:
        return self.resname.upper() in HIS_RESNAMES


def _is_heavy(atom: gemmi.Atom) -> bool:
    """Non-hydrogen. Tested on the element name rather than ``Element.is_hydrogen``,
    which is a method in some gemmi versions and a property in others (0.7.x)."""
    return atom.element.name.upper() not in ("H", "D")


def _is_amino_acid(res: gemmi.Residue) -> bool:
    info = gemmi.find_tabulated_residue(res.name)
    return bool(info and info.is_amino_acid())


def one_letter(res: gemmi.Residue) -> str:
    """One-letter code, with the histidine protonation variants mapped explicitly."""
    if (alias := CODE_ALIASES.get(res.name.upper())) is not None:
        return alias
    info = gemmi.find_tabulated_residue(res.name)
    code = (info.one_letter_code if info else "").upper()
    return code if code.isalpha() else "X"


def _virtual_cb(
    n: np.ndarray | None, ca: np.ndarray | None, c: np.ndarray | None
) -> np.ndarray | None:
    """CB position built from the backbone (glycine, or a missing CB)."""
    if n is None or ca is None or c is None:
        return None
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return _CB_A * a + _CB_B * b + _CB_C * cc + ca


def _residue(chain_name: str, res: gemmi.Residue) -> "Res | None":
    """One gemmi residue -> ``Res``, or None when it has no heavy atoms."""
    coords: list[list[float]] = []
    masses: list[float] = []
    named: dict[str, np.ndarray] = {}
    for atom in res:
        if not _is_heavy(atom):
            continue
        pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        coords.append([pos[0], pos[1], pos[2]])
        masses.append(float(atom.element.weight))
        # First occurrence wins; alternative conformers are already removed.
        named.setdefault(atom.name, pos)
    if not coords:
        return None

    icode = res.seqid.icode.strip()
    ca = named.get("CA")
    cb = named.get("CB")
    if cb is None:
        cb = _virtual_cb(named.get("N"), ca, named.get("C"))
    return Res(
        label=f"{chain_name}:{res.seqid.num}{icode}",
        chain=chain_name,
        seqid=res.seqid.num,
        icode=icode,
        resname=res.name.upper(),
        code=one_letter(res),
        heavy=np.asarray(coords, dtype=float),
        masses=np.asarray(masses, dtype=float),
        ca=ca,
        cb=cb,
    )


def chain_residues(model: gemmi.Model, chain_names: list[str]) -> list[Res]:
    """Amino acids of the named chains, in the order the chains were named."""
    out: list[Res] = []
    for name in chain_names:
        for chain in model:
            if chain.name != name:
                continue
            for res in chain:
                if _is_amino_acid(res) and (r := _residue(chain.name, res)) is not None:
                    out.append(r)
    return out


def protein_chain_names(model: gemmi.Model) -> list[str]:
    """Names of every chain holding at least one amino-acid residue, in file order."""
    seen: list[str] = []
    for chain in model:
        if chain.name not in seen and any(_is_amino_acid(r) for r in chain):
            seen.append(chain.name)
    return seen


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def centre_of_mass(residues: list[Res]) -> np.ndarray | None:
    """Mass-weighted centre of the residues' heavy atoms."""
    if not residues:
        return None
    coords = np.concatenate([r.heavy for r in residues])
    masses = np.concatenate([r.masses for r in residues])
    total = float(masses.sum())
    if total <= 0:
        return None
    return (coords * masses[:, None]).sum(axis=0) / total


def min_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance between two small point sets."""
    delta = a[:, None, :] - b[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)).min())


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float | None:
    """Angle between two vectors in degrees, None when either is degenerate."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return None
    cos = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def select_interface(
    binder: list[Res],
    target: list[Res],
    method: str,
    contact_cutoff: float,
    cb_dist_cut: float,
    vector_dist_cut: float,
    vector_angle_cut: float,
) -> None:
    """Mark the interface residues on both sides, in place.

    Only pairs whose CB atoms are within ``cb_dist_cut`` are considered at all; that
    prefilter is what keeps the heavy-atom distance work small. Each candidate pair
    is then tested by contact (``nearby``) and, under ``--method vector``, by whether
    either residue's side chain points at the other (``vector``).
    """
    b_cb = [r.cb if r.cb is not None else r.heavy.mean(axis=0) for r in binder]
    t_cb = [r.cb if r.cb is not None else r.heavy.mean(axis=0) for r in target]
    if not b_cb or not t_cb:
        return
    b_arr, t_arr = np.asarray(b_cb), np.asarray(t_cb)

    delta = b_arr[:, None, :] - t_arr[None, :, :]
    cb_dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))

    for i, j in np.argwhere(cb_dist <= cb_dist_cut):
        bi, tj = binder[int(i)], target[int(j)]
        d_cb = float(cb_dist[i, j])
        bi.min_cb = d_cb if bi.min_cb is None else min(bi.min_cb, d_cb)
        tj.min_cb = d_cb if tj.min_cb is None else min(tj.min_cb, d_cb)

        d_heavy = min_dist(bi.heavy, tj.heavy)
        bi.min_heavy = d_heavy if bi.min_heavy is None else min(bi.min_heavy, d_heavy)
        tj.min_heavy = d_heavy if tj.min_heavy is None else min(tj.min_heavy, d_heavy)

        if d_heavy <= contact_cutoff:
            for r in (bi, tj):
                r.selected = True
                r.by.add("nearby")
            continue

        if method != "vector" or d_cb > vector_dist_cut:
            continue
        # Per residue, not per pair: each side's list stands on its own.
        for src, dst in ((bi, tj), (tj, bi)):
            if src.ca is None or src.cb is None or dst.cb is None:
                continue
            angle = _angle_deg(src.cb - src.ca, dst.cb - src.cb)
            if angle is not None and angle < vector_angle_cut:
                src.selected = True
                src.by.add("vector")


# ---------------------------------------------------------------------------
# Per-design computation
# ---------------------------------------------------------------------------


def split_list(value: str) -> list[str]:
    """Comma-joined CLI list -> items, dropping blanks."""
    return [v.strip() for v in value.split(",") if v.strip()]


def load_structure(path: Path) -> gemmi.Structure:
    """Read a PDB or mmCIF file into a single-model, single-conformer structure."""
    if not path.exists():
        raise FileNotFoundError(f"structure missing: {path}")
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    if len(st) == 0:
        raise ValueError(f"no model in {path}")
    return st


def _fmt(value: object) -> str:
    """TSV cell: NA (None) becomes empty, floats get 3 decimals."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_point(point: np.ndarray | None) -> str | None:
    """A coordinate as ``x,y,z`` with 3 decimals (NA when absent)."""
    if point is None:
        return None
    return ",".join(f"{v:.3f}" for v in point)


def resolve_chains(
    model: gemmi.Model, binder_spec: str, target_spec: str, src_name: str
) -> tuple[list[str], list[str]]:
    """``--binder-chains``/``--target-chains`` -> the two disjoint chain lists.

    A chain named but absent is an error for this design, never a smaller selection:
    a silently narrower binder would make every number below it wrong.
    """
    present = protein_chain_names(model)
    binder_chains = split_list(binder_spec)
    if not binder_chains:
        raise ValueError("--binder-chains must name at least one chain")
    if missing := [c for c in binder_chains if c not in present]:
        raise ValueError(
            f"--binder-chains {missing} absent from {src_name} "
            f"(protein chains present: {present or '(none)'})"
        )

    if target_spec.strip().lower() == "auto":
        target_chains = [c for c in present if c not in set(binder_chains)]
        if not target_chains:
            raise ValueError(
                f"--target-chains auto found no protein chain outside "
                f"--binder-chains {binder_chains} in {src_name}; this file holds no "
                f"target, so there is no interface to measure"
            )
    else:
        target_chains = split_list(target_spec)
        if not target_chains:
            raise ValueError("--target-chains must name at least one chain, or 'auto'")
        if missing := [c for c in target_chains if c not in present]:
            raise ValueError(
                f"--target-chains {missing} absent from {src_name} "
                f"(protein chains present: {present or '(none)'})"
            )

    if overlap := sorted(set(binder_chains) & set(target_chains)):
        raise ValueError(
            f"chains {overlap} are on both sides; an interface needs two disjoint sides"
        )
    return binder_chains, target_chains


def compute(
    args: "Args", src: Path
) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    """Every ifacegeom metric for one complex -> (status, metrics, per-residue rows).

    Raises on anything unusable (the caller records the message as data).
    """
    started = time.time()
    model = load_structure(src)[0]
    binder_chains, target_chains = resolve_chains(
        model, args.binder_chains, args.target_chains, src.name
    )

    binder = chain_residues(model, binder_chains)
    target = chain_residues(model, target_chains)
    if not binder or not target:
        raise ValueError(
            f"no amino acids on one side (binder {len(binder)}, target {len(target)})"
        )

    select_interface(
        binder,
        target,
        args.method,
        args.contact_cutoff,
        args.cb_dist_cut,
        args.vector_dist_cut,
        args.vector_angle_cut,
    )
    b_iface = [r for r in binder if r.selected]
    t_iface = [r for r in target if r.selected]

    # Termini: the first residue of the first binder chain and the last of the last,
    # in --binder-chains order. Recorded by label so every projection below can be
    # checked against the structure by hand.
    nterm, cterm = binder[0], binder[-1]

    binder_com = centre_of_mass(binder)
    binder_iface_com = centre_of_mass(b_iface)
    iface_com = centre_of_mass(b_iface + t_iface)

    axis_len: float | None = None
    unit: np.ndarray | None = None
    if binder_com is not None and iface_com is not None:
        axis = iface_com - binder_com
        axis_len = float(np.linalg.norm(axis))
        # A degenerate axis means the interface COM sits on the binder COM; the
        # plane is then undefined and the projections are NA rather than noise.
        unit = axis / axis_len if axis_len > 1e-6 else None

    iface_ca = np.asarray([r.ca for r in b_iface if r.ca is not None])

    def terminus(res: Res) -> tuple[float | None, float | None, float | None]:
        """(signed projection, distance to iface COM, distance to nearest epitope CA)."""
        if res.ca is None:
            return None, None, None
        proj = (
            float(np.dot(res.ca - binder_com, unit))
            if unit is not None and binder_com is not None
            else None
        )
        dist = (
            float(np.linalg.norm(res.ca - iface_com)) if iface_com is not None else None
        )
        near = min_dist(res.ca.reshape(1, 3), iface_ca) if iface_ca.size else None
        return proj, dist, near

    n_proj, n_dist, n_near = terminus(nterm)
    c_proj, c_dist, c_near = terminus(cterm)

    metrics: dict[str, object] = {
        "method": args.method,
        "binder_chains": ",".join(binder_chains),
        "target_chains": ",".join(target_chains),
        "binder_len": len(binder),
        "target_len": len(target),
        "n_binder_res": len(b_iface),
        "binder_res": ",".join(r.label for r in b_iface),
        "binder_res_seq": "".join(r.code for r in b_iface),
        "n_binder_his": sum(r.is_his for r in b_iface),
        "binder_his": ",".join(r.label for r in b_iface if r.is_his),
        "n_target_res": len(t_iface),
        "target_res": ",".join(r.label for r in t_iface),
        "n_target_his": sum(r.is_his for r in t_iface),
        "target_his": ",".join(r.label for r in t_iface if r.is_his),
        "binder_com": _fmt_point(binder_com),
        "binder_iface_com": _fmt_point(binder_iface_com),
        "iface_com": _fmt_point(iface_com),
        "axis_len": axis_len,
        "nterm_res": nterm.label,
        "cterm_res": cterm.label,
        "nterm_proj": n_proj,
        "cterm_proj": c_proj,
        "nterm_iface_dist": n_dist,
        "cterm_iface_dist": c_dist,
        "nterm_iface_min_dist": n_near,
        "cterm_iface_min_dist": c_near,
        "seconds": time.time() - started,
    }

    # Every residue that had a candidate partner, selected or not: the evidence
    # behind the two lists, and what the epitope is cross-checked against.
    per_residue: list[dict[str, object]] = []
    for side, residues in (("binder", binder), ("target", target)):
        for r in residues:
            if r.min_cb is None:
                continue
            per_residue.append(
                {
                    "side": side,
                    "label": r.label,
                    "chain": r.chain,
                    "resnum": r.seqid,
                    "icode": r.icode,
                    "resname": r.resname,
                    "code": r.code,
                    "selected": int(r.selected),
                    "selected_by": "+".join(sorted(r.by)),
                    "min_heavy_dist": r.min_heavy,
                    "min_cb_dist": r.min_cb,
                }
            )

    # An empty interface is almost always the wrong --binder-chains/--target-chains
    # rather than a real non-contact, and every geometry column below it is NA. Say
    # so in the status (which keeps the row out of any `== "OK"` selection) while
    # still recording the lengths and chain names that make the mistake obvious.
    status = "OK" if b_iface and t_iface else "warn: no interface residues"
    return status, metrics, per_residue


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


def write_per_residue(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(PER_RESIDUE_COLUMNS)
        for row in rows:
            writer.writerow([_fmt(row.get(c)) for c in PER_RESIDUE_COLUMNS])


def read_task_file(path: Path) -> list[tuple[str, Path]]:
    """``name<TAB>structure`` lines -> pairs."""
    designs: list[tuple[str, Path]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise ValueError(f"malformed task line in {path}: {line!r}")
        designs.append((fields[0], Path(fields[1])))
    return designs


class Args(argparse.Namespace):
    task_file: Path
    binder_chains: str
    target_chains: str
    method: str
    contact_cutoff: float
    cb_dist_cut: float
    vector_dist_cut: float
    vector_angle_cut: float
    out_dir: Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure binder/target interface geometry for a batch of designs."
    )
    ap.add_argument(
        "--task-file",
        type=Path,
        required=True,
        help="Sub-manifest: one 'name<TAB>structure' line per design.",
    )
    ap.add_argument("--binder-chains", default="A")
    ap.add_argument(
        "--target-chains",
        default="auto",
        help="'auto' (default) takes every protein chain that is not a binder chain.",
    )
    ap.add_argument("--method", choices=METHODS, default="vector")
    ap.add_argument("--contact-cutoff", type=float, default=5.5)
    ap.add_argument("--cb-dist-cut", type=float, default=11.0)
    ap.add_argument("--vector-dist-cut", type=float, default=9.0)
    ap.add_argument("--vector-angle-cut", type=float, default=75.0)
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Where the per-design TSVs go."
    )
    args = ap.parse_args(namespace=Args())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    designs = read_task_file(args.task_file)
    failures = 0
    for name, src in designs:
        result_tsv = args.out_dir / f"{name}.tsv"
        per_residue_tsv = args.out_dir / f"{name}_per_residue.tsv"
        try:
            status, metrics, per_residue = compute(args, src)
            write_per_residue(per_residue_tsv, per_residue)
            print(
                f"{name}: {status} -- {metrics['n_binder_res']} binder / "
                f"{metrics['n_target_res']} target interface residues, "
                f"{metrics['n_target_his']} target His, cterm_proj "
                f"{_fmt(metrics['cterm_proj'])} A"
            )
            write_result(result_tsv, name, status, str(per_residue_tsv), metrics)
        except Exception as e:  # noqa: BLE001 - errors are recorded as data
            failures += 1
            print(f"{name}: ERROR {e}", file=sys.stderr)
            write_result(result_tsv, name, f"error: {e}", "", {})

    # Every design has a result either way; exiting non-zero would only hide that
    # behind a task-level failure, so the batch succeeds and the table carries the
    # errors.
    print(f"ifacegeom: {failures} failed of {len(designs)}")


if __name__ == "__main__":
    main()
