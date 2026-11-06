# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Phase 0: link a toy rocKE device function into a FlyDSL 0.4.0 kernel.

Skips when rocKE is absent from ``PYTHONPATH`` (``platform/python`` +
``library``, ``ROCKE_BACKEND=python``). Measures:

* rocKE sidecar ``ffi_types`` vs FlyDSL's type map
* ``ROCKE_LLVM_FLAVOR`` / the emitted flavor vs FlyDSL's bundled LLVM
* whether an extern-linked build reaches FlyDSL's disk cache
* cold and warm compile times

Note: this file deliberately omits ``from __future__ import annotations`` so
FlyDSL sees real ``Int64`` / ``Int32`` / ``fx.Stream`` types on ``@flyc.jit``
parameters (stringified annotations truncate device pointers to Int32 and
fault).
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

flydsl = pytest.importorskip("flydsl")
flyc = pytest.importorskip("flydsl.compiler")
pytest.importorskip("flydsl.compiler.extern_link")
pytest.importorskip("flydsl.expr.extern")

try:
    from rocke.core.ir import I32, IRBuilder, PtrType
except ImportError:
    pytest.skip("rocKE not on PYTHONPATH", allow_module_level=True)

try:
    from kernels.gfx950.qwen38_mono.device_function import (
        DeviceFunctionKnobs,
        lower_device_function,
        write_device_function,
    )
except ImportError:
    pytest.skip("rocKE qwen38_mono device_function missing", allow_module_level=True)

import flydsl.expr as fx
from flydsl.compiler.extern_link import link_extern
from flydsl.expr.extern import ffi
from flydsl.expr.typing import Int32, Int64

SYMBOL = "rocke_ml4_smoke_add"
THREADS = 64
N = 64


def _toy_kernel():
    b = IRBuilder("smoke_add")
    b.kernel.attrs["max_workgroup_size"] = THREADS
    out = b.param("out", PtrType(I32, "global"), noalias=True, align=16)
    x = b.param("x", I32)
    tid = b.thread_id_x()
    b.global_store(out, tid, b.add(tid, x), align=4)
    b.ret()
    return b.kernel


def _generate(out_dir: Path) -> dict:
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR")
    fn = lower_device_function(
        _toy_kernel(),
        arch="gfx950",
        symbol=SYMBOL,
        llvm_flavor=flavor,
        knobs=DeviceFunctionKnobs(lds="global", inline="always"),
    )
    paths = write_device_function(fn, str(out_dir), stem="smoke_add", bitcode="auto")
    meta = json.loads(Path(paths["json"]).read_text())
    return {"fn": fn, "paths": paths, "meta": meta}


def _cache_roots():
    roots = []
    for key in ("FLYDSL_CACHE_DIR", "FLYDSL_CACHE", "XDG_CACHE_HOME"):
        v = os.environ.get(key)
        if v:
            roots.append(Path(v))
    roots.append(Path.home() / ".cache" / "flydsl")
    roots.append(Path.home() / ".cache")
    return roots


def _fingerprint_caches():
    digests = {}
    for root in _cache_roots():
        if not root.is_dir():
            continue
        files = sorted(p for p in root.rglob("*") if p.is_file())
        if not files:
            continue
        h = hashlib.sha256()
        for f in files:
            try:
                rel = str(f.relative_to(root))
            except ValueError:
                rel = str(f)
            h.update(rel.encode())
            h.update(f.read_bytes())
        digests[str(root)] = h.hexdigest()
    return digests


def _build_launcher(bitcode_path, name):
    """Compile a 64-thread kernel that calls the rocKE smoke function."""
    ext = link_extern(ffi(SYMBOL, ["int64", "int32"], "void"), bitcode_path=bitcode_path)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def smoke(out: Int64, x: Int32):
        ext(out, x)

    @flyc.jit
    def launch(
        out: Int64,
        x: Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        smoke(out, x).launch(grid=(1,), block=(THREADS,), stream=stream)

    return launch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP GPU required")
def test_mono_extern_smoke(tmp_path, monkeypatch):
    cache = tmp_path / "flydsl_cache"
    cache.mkdir()
    monkeypatch.setenv("FLYDSL_CACHE_DIR", str(cache))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    (tmp_path / "xdg").mkdir()

    gen = _generate(tmp_path / "rocke")
    meta = gen["meta"]
    fn = gen["fn"]
    link_path = gen["paths"]["bc"] or gen["paths"]["ll"]

    assert meta["ffi_types"] == ["int64", "int32"] == fn.ffi_types()
    assert meta["symbol"] == SYMBOL

    flydsl_ver = getattr(flydsl, "__version__", "?")
    flavor = meta.get("flavor") or fn.flavor
    env_flavor = os.environ.get("ROCKE_LLVM_FLAVOR")
    print(f"FLYDSL_VERSION={flydsl_ver}")
    print(f"ROCKE_LLVM_FLAVOR(env)={env_flavor}")
    print(f"ROCKE_EMITTED_FLAVOR={flavor}")
    print(f"LINK_PATH={link_path}")

    device = torch.device("cuda:0")
    out = torch.zeros(N, dtype=torch.int32, device=device)
    x = 7
    expect = torch.arange(N, dtype=torch.int32, device=device) + x

    before = _fingerprint_caches()
    t0 = time.perf_counter()
    launch1 = _build_launcher(link_path, name="mono_extern_smoke_cold")
    launch1(Int64(out.data_ptr()), Int32(x))
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - t0
    after_cold = _fingerprint_caches()
    torch.testing.assert_close(out, expect)

    out.zero_()
    t1 = time.perf_counter()
    launch2 = _build_launcher(link_path, name="mono_extern_smoke_warm")
    launch2(Int64(out.data_ptr()), Int32(x))
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - t1
    after_warm = _fingerprint_caches()
    torch.testing.assert_close(out, expect)

    cache_grew_cold = after_cold != before
    cache_grew_warm = after_warm != after_cold
    print(f"COMPILE_COLD_S={cold_s:.3f}")
    print(f"COMPILE_WARM_S={warm_s:.3f}")
    print(f"DISK_CACHE_GREW_ON_COLD={cache_grew_cold}")
    print(f"DISK_CACHE_GREW_ON_WARM={cache_grew_warm}")
    print(f"CACHE_ROOTS={list(after_warm)}")
    report = {
        "flydsl_version": flydsl_ver,
        "rocke_llvm_flavor_env": env_flavor,
        "rocke_emitted_flavor": flavor,
        "ffi_types": meta["ffi_types"],
        "compile_cold_s": cold_s,
        "compile_warm_s": warm_s,
        "disk_cache_grew_on_cold": cache_grew_cold,
        "disk_cache_grew_on_warm": cache_grew_warm,
        "link_path": link_path,
    }
    report_path = Path(
        os.environ.get("MONO_EXTERN_SMOKE_REPORT", str(tmp_path / "report.json"))
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"REPORT={report_path}")


if __name__ == "__main__":
    import tempfile

    class _Dummy:
        def setenv(self, k, v):
            os.environ[k] = v

    with tempfile.TemporaryDirectory() as d:
        test_mono_extern_smoke(Path(d), _Dummy())
