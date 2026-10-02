"""One-off: generate afilmv_ehl .pickle caches on the rpxdock-hscore Volume.

  modal run scripts/rpxdock_pickle/gen_pickles.py::generate   # stage + verify
  modal run scripts/rpxdock_pickle/gen_pickles.py::promote    # move into alias dir

Only in-container paths are used inside the functions.
"""
import sys
from pathlib import Path

import modal

ROOT = "/rpxdock_files"
ALIAS = "afilmv_ehl"
STAGING = f"{ROOT}/.pickle_staging_afilmv"

if modal.is_local():
    from dotenv import load_dotenv

    repo = Path(__file__).resolve().parents[2]
    load_dotenv(repo / ".env")
    sys.path.insert(0, str(repo / "tools" / "rpxdock"))
    import modal_image as mi

    _image, _vols = mi.image(), mi.volumes()
else:
    _image = modal.Image.debian_slim()  # placeholder; the container uses the image it was built with
    _vols = {ROOT: modal.Volume.from_name("rpxdock-hscore")}

app = modal.App("rpxdock-pickle-gen", image=_image)


def _vol():
    return modal.Volume.from_name("rpxdock-hscore")


@app.function(volumes=_vols, cpu=4, memory=98304, timeout=7200)
def gen(extra_args: list[str]):
    import os, shutil, subprocess, time

    t0 = time.time()
    os.makedirs(STAGING, exist_ok=True)
    leftovers = os.listdir(STAGING)
    print("staging dir:", STAGING, "initial contents:", leftovers, flush=True)
    assert not leftovers, "staging dir not empty"
    print("alias dir:", sorted(os.listdir(f"{ROOT}/{ALIAS}")), flush=True)
    cmd = ["python", "-m", "rpxdock", "--hscore_files", ALIAS,
           "--hscore_data_dir", ROOT, "--generate_hscore_pickle_files"] + extra_args
    print("cwd:", STAGING, "cmd:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=STAGING)
    print("generator returncode:", r.returncode, "elapsed s:", round(time.time() - t0), flush=True)
    for f in sorted(os.listdir(STAGING)):
        print("staged", f, os.path.getsize(os.path.join(STAGING, f)), flush=True)
    import modal as _m
    _m.Volume.from_name("rpxdock-hscore").commit()
    print("committed", flush=True)
    return r.returncode


@app.function(volumes=_vols, cpu=1, memory=2048, timeout=1800)
def move():
    import os, shutil

    st, al = STAGING, f"{ROOT}/{ALIAS}"
    names = sorted(os.listdir(st))
    assert len(names) == 6 and all(n.endswith(".pickle") for n in names), names
    for n in names:
        os.rename(os.path.join(st, n), os.path.join(al, n))
    print("moved:", names, flush=True)
    os.rmdir(st)
    modal.Volume.from_name("rpxdock-hscore").commit()
    print("committed", flush=True)
    for f in sorted(os.listdir(al)):
        print(f, os.path.getsize(os.path.join(al, f)))
    print("count:", len(os.listdir(al)))


@app.function(volumes=_vols, cpu=1, memory=2048, timeout=600)
def listing():
    import os
    for d in (STAGING, f"{ROOT}/{ALIAS}"):
        print(d)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                print(" ", f, os.path.getsize(os.path.join(d, f)))


@app.local_entrypoint()
def generate(extra: str = ""):
    print("rc =", gen.remote(extra.split() if extra else []))


@app.local_entrypoint()
def promote():
    move.remote()


@app.local_entrypoint()
def ls():
    listing.remote()


def _sig(obj, path="obj", depth=0, out=None):
    """Flatten an object into {path: description}; numpy arrays get shape/dtype/md5."""
    import hashlib
    import numpy as np

    out = {} if out is None else out
    t = type(obj).__name__
    if isinstance(obj, np.ndarray):
        d = f"ndarray {obj.shape} {obj.dtype}"
        if obj.dtype != object:
            d += " md5=" + hashlib.md5(np.ascontiguousarray(obj).tobytes()).hexdigest()
        out[path] = d
    elif t == "Dataset" or t == "DataArray":
        out[path] = f"{t} dims={dict(obj.sizes)}"
        if t == "Dataset":
            for k in obj.variables:
                _sig(np.asarray(obj[k].values), f"{path}[{k}]", depth + 1, out)
        else:
            _sig(np.asarray(obj.values), f"{path}.values", depth + 1, out)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        out[path] = f"{t} {obj!r}"[:120]
    elif isinstance(obj, (list, tuple)) and depth < 3:
        out[path] = f"{t} len={len(obj)}"
        for i, v in enumerate(obj[:8]):
            _sig(v, f"{path}[{i}]", depth + 1, out)
    elif isinstance(obj, dict) and depth < 3:
        out[path] = f"{t} len={len(obj)}"
        for k, v in list(obj.items())[:20]:
            _sig(v, f"{path}[{k!r}]", depth + 1, out)
    else:
        try:
            n = len(obj)
            out[path] = f"{t} len={n}"
        except Exception:
            out[path] = t
        if depth < 3 and hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                _sig(v, f"{path}.{k}", depth + 1, out)
    return out


@app.function(volumes=_vols, cpu=4, memory=98304, timeout=7200)
def verify():
    import gc, os, time
    import rpxdock as rp

    st, al = STAGING, f"{ROOT}/{ALIAS}"
    names = sorted(os.listdir(st))
    print("staged:", len(names), "total bytes:", sum(os.path.getsize(f"{st}/{n}") for n in names), flush=True)
    for n in names:
        p = f"{st}/{n}"
        t0 = time.time()
        try:
            obj = rp.util.load(p)
        except BaseException as e:
            print(f"FAIL {n}: {type(e).__name__}: {e!r}", flush=True)
            continue
        print(f"LOADED {n} size={os.path.getsize(p)} type={type(obj).__module__}.{type(obj).__name__} in {time.time()-t0:.0f}s", flush=True)
        sig = _sig(obj)
        for k, v in sig.items():
            print("   ", k, v, flush=True)
        if "_base.rpx" in n:
            o2 = rp.util.load(f"{al}/{n[:-len('.pickle')]}")
            sig2 = _sig(o2)
            diffs = [k for k in set(sig) | set(sig2) if sig.get(k) != sig2.get(k)]
            print(f"BASE vs original .txz: {len(sig)} vs {len(sig2)} entries; differing: {diffs}", flush=True)
            for k in diffs[:20]:
                print("   ", k, "| pickle:", sig.get(k), "| txz:", sig2.get(k), flush=True)
            del o2, sig2
        del obj, sig
        gc.collect()


@app.local_entrypoint()
def check():
    verify.remote()


@app.function(volumes=_vols, cpu=4, memory=98304, timeout=3600)
def base_check():
    import rpxdock as rp
    n = "pdb_res_pair_data_si30_rots_EHL_AFILMV_SSdep_p0.5_b1_base.rpx.txz"
    o2 = rp.util.load(f"{ROOT}/{ALIAS}/{n}")  # primes xarray's dynamic class
    sig2 = _sig(o2)
    print("txz loaded:", type(o2).__name__, len(sig2), "entries", flush=True)
    o = rp.util.load(f"{STAGING}/{n}.pickle")
    sig = _sig(o)
    print("pickle loaded (after priming):", type(o).__name__, len(sig), "entries", flush=True)
    diffs = [k for k in set(sig) | set(sig2) if sig.get(k) != sig2.get(k)]
    print("differing entries:", diffs, flush=True)
    for k in sorted(sig2):
        print("  ", k, "|", sig2[k], "| same" if k not in diffs else "| DIFF " + str(sig.get(k)), flush=True)


@app.local_entrypoint()
def basecheck():
    base_check.remote()


BASE = "pdb_res_pair_data_si30_rots_EHL_AFILMV_SSdep_p0.5_b1_base.rpx.txz"


def _lazy_datasets(obj, path="obj", depth=0, found=None):
    found = [] if found is None else found
    t = type(obj).__name__
    if t in ("Dataset", "DataArray"):
        found.append((path, obj))
    elif isinstance(obj, (list, tuple)) and depth < 3:
        for i, v in enumerate(obj):
            _lazy_datasets(v, f"{path}[{i}]", depth + 1, found)
    elif isinstance(obj, dict) and depth < 3:
        for k, v in obj.items():
            _lazy_datasets(v, f"{path}[{k!r}]", depth + 1, found)
    elif hasattr(obj, "__dict__") and depth < 3:
        for k, v in vars(obj).items():
            _lazy_datasets(v, f"{path}.{k}", depth + 1, found)
    return found


@app.function(volumes=_vols, cpu=4, memory=98304, timeout=3600)
def regen():
    import os
    import rpxdock as rp

    src = f"{ROOT}/{ALIAS}/{BASE}"
    final = f"{STAGING}/{BASE}.pickle"
    tmp = f"{STAGING}/{BASE}.tmp_regen"  # not *.pickle, and outside the alias dir
    print("old staged file:", final, os.path.getsize(final), flush=True)
    obj = rp.util.load(src)
    print("loaded", type(obj).__name__, flush=True)
    for path, ds in _lazy_datasets(obj):
        print("dataset found:", path, type(ds).__name__, "dims", dict(ds.sizes), flush=True)
    # materialise every Dataset/DataArray found, then reassign it onto its parent
    for k, v in list(vars(obj).items()):
        if type(v).__name__ in ("Dataset", "DataArray"):
            import numpy as np
            import xarray as xr
            import pickle as _pk
            loaded = v.load()
            print("close hook before:", getattr(loaded, "_close", "n/a"), flush=True)
            # .load() alone is NOT enough: Dataset._close still holds the scipy store.
            # Rebuild a Dataset that owns plain numpy arrays and has no close hook.
            if type(loaded).__name__ == "Dataset":
                fresh_ds = xr.Dataset(
                    {n: xr.Variable(c.dims, np.array(c.values), dict(c.attrs))
                     for n, c in loaded.data_vars.items()},
                    coords={n: xr.Variable(c.dims, np.array(c.values), dict(c.attrs))
                            for n, c in loaded.coords.items()},
                    attrs=dict(loaded.attrs),
                )
            else:
                fresh_ds = loaded.copy(deep=True)
                fresh_ds.set_close(None)
            setattr(obj, k, fresh_ds)
            print("materialised obj." + k, "close hook now:", getattr(fresh_ds, "_close", "n/a"), flush=True)
    import pickle as _pk
    blob = _pk.dumps(obj)
    leaked = [m for m in (b"scipy_", b"_PickleWorkaround", b"flush_only", b"xarray.backends") if m in blob]
    print("in-memory pickle bytes:", len(blob), "leaked markers:", leaked, flush=True)
    del blob
    if leaked:
        print("ABORT: still references the netcdf store; staged file left untouched", flush=True)
        return
    # re-inspect: report anything still referencing a netcdf store
    for path, ds in _lazy_datasets(obj):
        print("after load:", path, "encoding source:", ds.encoding.get("source"), flush=True)
    import pickle
    blob_ok = True
    try:
        rp.dump(obj, tmp)
    except BaseException as e:
        blob_ok = False
        print("DUMP FAILED:", repr(e), flush=True)
        if os.path.exists(tmp):
            os.remove(tmp)
    if blob_ok:
        os.replace(tmp, final)
        print("replaced", final, "new size", os.path.getsize(final), flush=True)
    modal.Volume.from_name("rpxdock-hscore").commit()
    print("committed; staging now:", flush=True)
    tot = 0
    for f in sorted(os.listdir(STAGING)):
        s = os.path.getsize(f"{STAGING}/{f}")
        tot += s
        print("  ", f, s, flush=True)
    print("staging total bytes:", tot, flush=True)
    print("alias dir:", sorted(os.listdir(f"{ROOT}/{ALIAS}")), flush=True)
    txz = os.path.getsize(src)
    print("ratio new/txz:", os.path.getsize(final) / txz, flush=True)


@app.function(volumes=_vols, cpu=4, memory=98304, timeout=3600)
def fresh_load():
    import rpxdock as rp
    o = rp.util.load(f"{STAGING}/{BASE}.pickle")  # nothing else loaded in this process
    print("FRESH LOAD OK:", type(o).__module__ + "." + type(o).__name__, flush=True)


@app.local_entrypoint()
def regen_base():
    regen.remote()


@app.local_entrypoint()
def fresh():
    fresh_load.remote()
