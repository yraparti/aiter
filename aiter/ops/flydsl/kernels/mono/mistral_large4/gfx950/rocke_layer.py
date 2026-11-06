# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Launch the rocKE ML4 decoder-layer kernel in place of the FlyDSL one.

``AITER_ML4_ROCKE=<rocKE tree>`` makes ``Glm5MonoKernel`` build the rocKE
layer kernel for its configuration and launch it, through the HIP module API,
with the FlyDSL launcher's arguments unchanged. The rocKE kernel places its
tasks on the CTAs the FlyDSL build would, and reads and writes the same
scratch, symmetric buffer and caches.

rocKE compiles in a subprocess (FlyDSL embeds its own LLVM; a second one in
this process can abort it). Code objects are cached under
``AITER_ML4_ROCKE_CACHE`` (default ``~/.cache/aiter_ml4_rocke``), keyed by the
spec and the rocKE layer sources.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import pathlib
import struct
import subprocess
import sys

import torch

from .variant import KV_SHARE, NOPE_LAYER

ENV = "AITER_ML4_ROCKE"
CACHE_ENV = "AITER_ML4_ROCKE_CACHE"
SPEC_ENV = "AITER_ML4_ROCKE_SPEC"
# directory that receives each built spec as <name>.json (offline rebuilds)
SPEC_DUMP_ENV = "AITER_ML4_ROCKE_SPEC_DUMP"
# mb_numa: a fixed copy mask (an integer) instead of the calibrated one
NUMA_MASK_ENV = "AITER_ML4_ROCKE_NUMA_MASK"
LAYER_PKG = ("library", "kernels", "gfx950", "ml4_mono", "layer")
GRID, THREADS = 256, 512

# mb_numa geometry (the layer's ctx.py): a header (word 0 the copy mask) at
# scratch + numa_hdr, then from the next 2 MiB boundary every logical byte x
# below numa_hdr at x + (x & ~8191) (copy 0) and 8 KiB past it (copy 1)
NUMA_HDR_BYTES, NUMA_PAGE, NUMA_CHUNK = 64, 1 << 21, 8192
NUMA_MAGIC = 0x4E554D41
# regions the layer reads or writes with plain loads / stores (ctx.NUMA_PLAIN)
NUMA_PLAIN = ("indices", "xqd")


def rocke_root() -> str | None:
    return os.environ.get(ENV) or None


def _loaded_hip() -> str:
    """Path of the HIP runtime this process already mapped: a bare soname
    could load a second runtime next to it."""
    torch.cuda.init()
    with open("/proc/self/maps") as f:
        for line in f:
            if "libamdhip64.so" in line:
                return line.split()[-1]
    raise RuntimeError("libamdhip64 is not loaded in this process")


class Hip:
    """hipModule loading and launching through this process's HIP runtime."""

    def __init__(self):
        self.lib = ctypes.CDLL(_loaded_hip())
        self.lib.hipGetErrorString.restype = ctypes.c_char_p
        self.keep = []

    def _ok(self, rc, what):
        if rc:
            raise RuntimeError(f"{what}: {self.lib.hipGetErrorString(rc).decode()}")

    def function(self, hsaco: bytes, name: str):
        image = ctypes.create_string_buffer(hsaco, len(hsaco))
        module = ctypes.c_void_p()
        self._ok(self.lib.hipModuleLoadData(ctypes.byref(module), image), "hipModuleLoadData")
        fn = ctypes.c_void_p()
        self._ok(
            self.lib.hipModuleGetFunction(ctypes.byref(fn), module, name.encode()),
            f"hipModuleGetFunction({name})",
        )
        self.keep.append((image, module))
        return fn

    def launch(self, fn, grid: int, block: int, args: bytes, stream: int):
        buf = ctypes.create_string_buffer(args, len(args))
        size = ctypes.c_size_t(len(args))
        extra = (ctypes.c_void_p * 5)(
            1, ctypes.cast(buf, ctypes.c_void_p).value,
            2, ctypes.cast(ctypes.pointer(size), ctypes.c_void_p).value,
            3,
        )  # fmt: skip
        u = ctypes.c_uint
        self._ok(
            self.lib.hipModuleLaunchKernel(
                fn, u(grid), u(1), u(1), u(block), u(1), u(1), u(0),
                ctypes.c_void_p(stream), None, extra,
            ),
            "hipModuleLaunchKernel",
        )  # fmt: skip


_HIP: Hip | None = None


def hip() -> Hip:
    global _HIP
    if _HIP is None:
        _HIP = Hip()
    return _HIP


def pack_args(args) -> bytes:
    """The FlyDSL launcher's positional arguments as the rocKE kernel's
    kernarg block: 36 64-bit addresses, then rank and layer."""
    ptrs, (rank, layer) = args[:-2], args[-2:]
    if len(ptrs) != 36:
        raise ValueError(f"expected 36 pointer arguments, got {len(ptrs)}")
    return struct.pack("<36Q2i", *[int(p) & (2**64 - 1) for p in ptrs], int(rank), int(layer))


def kind_of(kernel) -> str:
    """This variant's layer kind (an index-sharing build is ``rope_shared``)."""
    return variant_kind(kernel.with_indexer)


def full_order(names) -> list[str]:
    """Every planned stage, in an order whose waits only point backwards."""
    levels = sorted((n for n in names if n.startswith("index_sel") and n != "index_select"),
                    key=lambda n: int(n[len("index_sel"):]))  # fmt: skip
    order = ["qkv_a", "q_norm", "q_b", "index_q", "cache", "uk", "index_score"]
    order += levels + ["index_select", "split", "uv", "o", "router", "ug", "down"]
    return [n for n in order if n in names]


def layer_spec(kernel, plan, kind: str, launches_per_step: int, stages=None) -> dict:
    """The rocKE ``Ml4LayerSpec`` (as a dict) of one ``Glm5MonoKernel``."""
    from .layout import layout

    # the indexer layout at this geometry, as the serving wrapper sizes it
    numa_hdr = layout(
        kernel.S, kernel.W.heads, kernel.npes, kernel.topk, True, kernel.index_max_seq,
        indexer_cp=kernel.indexer_cp, inter=kernel.W.config.inter,
        output_heads=kernel.output_heads, dcp_size=kernel.dcp_size,
        native_fp4_mfma=kernel.native_fp4_mfma,
    )[0]["_bytes"]  # fmt: skip
    return spec_of(
        kind=kind, rows=kernel.S, heads=kernel.W.heads, npes=kernel.npes,
        inter=kernel.W.config.inter, stage_tasks=kernel.stages, plan=plan,
        scr_layout=kernel.scr_layout, sym_layout=kernel.sym_layout,
        launches_per_step=launches_per_step, index_page_size=kernel.index_page_size,
        index_bt_stride=kernel.index_block_table_stride or 0,
        index_request_width=kernel.index_request_width or 0,
        index_capacity=kernel.index_max_seq, stages=stages, numa_hdr=numa_hdr,
    )  # fmt: skip


def spec_of(*, kind, rows, heads, npes, inter, stage_tasks, plan, scr_layout, sym_layout,
            launches_per_step, index_page_size, index_bt_stride, index_request_width,
            index_capacity, stages=None, numa_hdr: int = 0) -> dict:  # fmt: skip
    """The rocKE ``Ml4LayerSpec`` (as a dict) of one FlyDSL build: its stage
    task counts, the CTA placement ``plan`` it recorded and its layouts.
    ``numa_hdr``: the ``mb_numa`` header offset, past every layout that
    shares the scratch (the indexer layout covers the index-sharing one)."""
    stage_plan = [
        [name, int(n), int(plan["base"].get(name, 0))] for name, n in stage_tasks if n > 0
    ]
    names = [n for n, _, _ in stage_plan]
    stages = full_order(names) if stages is None else [s for s in stages if s in names]
    spec = dict(
        kind=kind,
        rows=int(rows),
        heads=int(heads),
        npes=int(npes),
        inter=int(inter),
        stages=stages,
        plan=stage_plan,
        scratch=[[n, int(o)] for n, o in scr_layout.items() if not n.startswith("_")],
        sym=[[n, int(o)] for n, o in sym_layout.items() if isinstance(o, int)],
        launches_per_step=int(launches_per_step),
        index_page_size=int(index_page_size),
        index_bt_stride=int(index_bt_stride),
        index_request_width=int(index_request_width),
        index_capacity=int(index_capacity),
        name=f"ml4rk_{kind}_r{rows}_h{heads}_tp{npes}",
    )
    # schedule overrides for A/B runs, e.g. '{"ug_prefetch": false}'
    spec.update(json.loads(os.environ.get(SPEC_ENV) or "{}"))
    if spec.get("mb_numa") and not numa_on(spec):
        spec["mb_numa"] = False
    if spec.get("mb_numa"):
        if numa_hdr <= 0:
            raise ValueError("mb_numa needs the caller's numa_hdr")
        spec.setdefault("numa_hdr", -(-int(numa_hdr) // NUMA_HDR_BYTES) * NUMA_HDR_BYTES)
    return spec


def numa_on(spec: dict) -> bool:
    """``mb_numa`` for this spec's kind (``numa_kinds`` names the kinds it
    applies to; empty: all)."""
    kinds = spec.get("numa_kinds") or ""
    return bool(spec.get("mb_numa")) and (not kinds or spec["kind"] in kinds.split("+"))


def numa_area(ptr: int, numa_hdr: int) -> tuple[int, int]:
    """(byte offset from ``ptr``, bytes) of the ``mb_numa`` dual-copy area."""
    start = (ptr + numa_hdr + NUMA_HDR_BYTES + NUMA_PAGE - 1) & ~(NUMA_PAGE - 1)
    return start - ptr, 2 * (-(-numa_hdr // NUMA_CHUNK) * NUMA_CHUNK)


def numa_scratch_bytes(numa_hdr: int) -> int:
    """Scratch bytes an ``mb_numa`` build needs, for any base alignment."""
    return numa_hdr + NUMA_HDR_BYTES + NUMA_PAGE + numa_area(0, numa_hdr)[1]


def numa_logical(scratch: torch.Tensor, numa_hdr: int, off: int, n: int, copy: int = 0) -> torch.Tensor:
    """Logical scratch bytes ``[off, off + n)`` of a tagged region, read from
    copy ``copy`` of the dual-copy area."""
    area_off, _ = numa_area(scratch.data_ptr(), numa_hdr)
    x = torch.arange(off, off + n, device=scratch.device, dtype=torch.int64)
    return scratch[area_off + x + (x & ~(NUMA_CHUNK - 1)) + copy * NUMA_CHUNK]


def numa_copy_bytes(scratch: torch.Tensor, numa_hdr: int, off: int, n: int) -> torch.Tensor:
    """Indices (into ``scratch``) of both copies of logical bytes ``[off, off + n)``."""
    area_off, _ = numa_area(scratch.data_ptr(), numa_hdr)
    x = torch.arange(off, off + n, device=scratch.device, dtype=torch.int64)
    p = area_off + x + (x & ~(NUMA_CHUNK - 1))
    return torch.cat([p, p + NUMA_CHUNK])


_PINGPONG = """
import json, sys
from rocke.examples.gfx950.persistent_mailbox import kernels as K
from rocke.helpers.compile import compile_kernel
art = compile_kernel(K.build_pingpong(K.PingPongSpec()), arch="gfx950", backend="python")
open(sys.argv[1], "wb").write(art.hsaco)
print(json.dumps({"kernel": art.kernel_name, "cols": K.PP_COLS}))
"""


def _pingpong(root: str) -> tuple[bytes, str, int]:
    """(code object, kernel name, out columns) of the platform's mailbox
    ping-pong, compiled once per cache directory."""
    cache = pathlib.Path(os.environ.get(CACHE_ENV, "~/.cache/aiter_ml4_rocke")).expanduser()
    final = cache / ("pingpong_" + _source_digest(root))
    built = final / "built.json"
    if not built.exists():
        out = final / f"tmp_{os.getpid()}"
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, ROCKE_BACKEND="python")
        env.pop("ROCKE_LLVM_FLAVOR", None)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(root, "platform", "python")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        hsaco = out / "pingpong.hsaco"
        r = subprocess.run([sys.executable, "-c", _PINGPONG, str(hsaco)], env=env, capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not lines:
            raise RuntimeError(f"ping-pong build failed:\n{r.stdout}\n{r.stderr}")
        info = json.loads(lines[-1])
        os.replace(hsaco, final / "pingpong.hsaco")
        info["hsaco"] = str(final / "pingpong.hsaco")
        marker = out / "built.json"
        marker.write_text(json.dumps(info))
        os.replace(marker, built)
    info = json.loads(built.read_text())
    return pathlib.Path(info["hsaco"]).read_bytes(), info["kernel"], int(info["cols"])


def numa_calibrate(scratch: torch.Tensor, numa_hdr: int, root: str, iters: int = 1500) -> int | None:
    """The copy mask of ``scratch``'s dual-copy area: bit p set when, on area
    page p, the 8 KiB block at page offset 0 (copy 0) is homed away from the
    IOD of XCD 0. A same-XCD ping-pong (CTAs 0 and 8) at page offsets 0 and
    8 KiB: the home is the faster one (the platform's ``calibrate_iod_home``).
    ``None`` when the probe does not land both CTAs on one XCD."""
    code, name, cols = _pingpong(root)
    fn = hip().function(code, name)
    area_off, area = numa_area(scratch.data_ptr(), numa_hdr)
    grid = 9
    out = torch.zeros(grid * cols, dtype=torch.int64, device=scratch.device)
    stream = torch.cuda.current_stream().cuda_stream
    base, mask = 1 << 24, 0
    for page in range(-(-area // NUMA_PAGE)):
        rtt = []
        for off in (0, NUMA_CHUNK):
            box = scratch.data_ptr() + area_off + page * NUMA_PAGE + off
            ticks = []
            for _ in range(3):
                out.zero_()
                hip().launch(fn, grid, 64, struct.pack("<QQiiii", box, out.data_ptr(), 0, 8, base, iters), stream)
                torch.cuda.synchronize()
                o = out.view(grid, cols).cpu()
                if int(o[0, 3]) != iters + 1000 or int(o[8, 3]) != iters or int(o[0, 4]) != int(o[8, 4]):
                    return None
                ticks.append((int(o[0, 1]) - int(o[0, 0])) / iters)
                base += iters + 1
            rtt.append(sorted(ticks)[1])
        mask |= (0 if rtt[0] <= rtt[1] else 1) << page
    return mask


def numa_prepare(scratch: torch.Tensor, numa_hdr: int, root: str) -> int:
    """The copy mask (calibrated, or ``AITER_ML4_ROCKE_NUMA_MASK``) into the
    header at ``numa_hdr`` and a zeroed dual-copy area; returns the mask.
    Idempotent: a header already written (another kind sharing the scratch)
    keeps its mask. Outside graph capture."""
    need = numa_scratch_bytes(numa_hdr)
    if scratch.numel() < need:
        raise ValueError(f"mb_numa needs {need} scratch bytes, got {scratch.numel()}")
    area_off, area = numa_area(scratch.data_ptr(), numa_hdr)
    if area > 32 * NUMA_PAGE:
        raise ValueError(f"mb_numa: a {area}-byte area is past the 32-page copy mask")
    hdr = scratch[numa_hdr : numa_hdr + 8].view(torch.int32)
    log = logging.getLogger("aiter")
    if int(hdr[1]) == NUMA_MAGIC:
        mask = int(hdr[0]) & 0xFFFFFFFF
    elif os.environ.get(NUMA_MASK_ENV):
        mask = int(os.environ[NUMA_MASK_ENV], 0)
    else:
        got = numa_calibrate(scratch, numa_hdr, root)
        if got is None:
            log.warning("ML4 mono layer: NUMA calibration did not land on one XCD; copy mask 0")
        mask = got or 0
    hdr[0], hdr[1] = mask - (1 << 32) if mask >> 31 else mask, NUMA_MAGIC
    scratch[area_off : area_off + area].zero_()
    torch.cuda.synchronize()
    log.info("ML4 mono layer: NUMA copy mask %#x over %d pages", mask, -(-area // NUMA_PAGE))
    return mask


def variant_kind(with_indexer: bool) -> str:
    """This variant's layer kind (an index-sharing build is ``rope_shared``)."""
    if not with_indexer:
        return "rope_shared"
    return ("nope" if NOPE_LAYER else "rope") + ("_kvs" if KV_SHARE else "")


def _source_digest(root: str) -> str:
    h = hashlib.sha256()
    pkg = pathlib.Path(root).joinpath(*LAYER_PKG)
    for p in sorted(pkg.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    for rel in ("platform/python/rocke/core/ir.py", "platform/python/rocke/core/lower_llvm.py"):
        h.update(pathlib.Path(root, rel).read_bytes())
    return h.hexdigest()[:16]


def build(spec: dict, root: str, flavor: str = "") -> tuple[bytes, str, dict, bool]:
    """(code object, kernel name, register / LDS usage, whether the build sums
    the MoE expert-major) of ``spec``, compiled once per cache key.
    Concurrent builders (one per TP rank) each compile into their own
    directory and publish with atomic renames, the marker last."""
    cache = pathlib.Path(os.environ.get(CACHE_ENV, "~/.cache/aiter_ml4_rocke")).expanduser()
    text = json.dumps(spec, sort_keys=True)
    dump = os.environ.get(SPEC_DUMP_ENV)
    if dump and not spec.get("timeline"):
        pathlib.Path(dump).mkdir(parents=True, exist_ok=True)
        pathlib.Path(dump, f"{spec['name']}.json").write_text(text)
    key = hashlib.sha256((text + _source_digest(root) + flavor).encode()).hexdigest()[:20]
    final = cache / key
    built = final / "built.json"
    if not built.exists():
        out = final / f"tmp_{os.getpid()}"
        out.mkdir(parents=True, exist_ok=True)
        spec_path = out / "spec.json"
        spec_path.write_text(text)
        env = dict(os.environ, ROCKE_BACKEND="python")
        env.pop("ROCKE_LLVM_FLAVOR", None)
        if flavor:
            env["ROCKE_LLVM_FLAVOR"] = flavor
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.join(root, "platform", "python"), os.path.join(root, "library")]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        r = subprocess.run(
            [sys.executable, "-m", "kernels.gfx950.ml4_mono.layer.build", str(spec_path), "--out", str(out)],
            env=env, capture_output=True, text=True,
        )  # fmt: skip
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not lines:
            raise RuntimeError(f"rocKE layer build failed:\n{r.stdout}\n{r.stderr}")
        info = json.loads(lines[-1])
        hsaco = final / pathlib.Path(info["hsaco"]).name
        os.replace(info["hsaco"], hsaco)
        info["hsaco"] = str(hsaco)
        marker = out / "built.json"
        marker.write_text(json.dumps(info))
        os.replace(marker, built)
    info = json.loads(built.read_text())
    return (pathlib.Path(info["hsaco"]).read_bytes(), info["kernel"], info.get("resources", {}),
            bool(info.get("moe_em")))  # fmt: skip


class RockeLauncher:
    """Drop-in for the FlyDSL ``launch``: same positional arguments and
    ``stream`` keyword."""

    def __init__(self, spec: dict, root: str):
        self.spec, self.root = spec, root
        # moe_em: x_out matches the FlyDSL layer to rounding, not bit for bit
        code, name, self.resources, self.moe_em = build(spec, root)
        self.fn = hip().function(code, name)
        logging.getLogger("aiter").info("ML4 mono layer: rocKE kernel %s %s", name, self.resources)
        self.numa_hdr = int(spec["numa_hdr"]) if spec.get("mb_numa") else 0
        # scratch bytes this build reads past its layout (0: the layout's own)
        self.scratch_bytes = numa_scratch_bytes(self.numa_hdr) if self.numa_hdr else 0
        self.prepared = not self.numa_hdr

    def prepare_scratch(self, scratch: torch.Tensor) -> None:
        """Before graph capture: ``numa_prepare`` for ``mb_numa`` builds."""
        if self.numa_hdr:
            numa_prepare(scratch, self.numa_hdr, self.root)
            self.prepared = True

    def __call__(self, *args, stream=None):
        if not self.prepared:
            # its dual copies sit past the layout: an unprepared scratch is
            # one the caller sized for the layout alone
            raise RuntimeError("mb_numa build launched before prepare_scratch (needs the runner that sizes the scratch)")
        s = stream if stream is not None else torch.cuda.current_stream()
        hip().launch(self.fn, GRID, THREADS, pack_args(args), s.cuda_stream)
