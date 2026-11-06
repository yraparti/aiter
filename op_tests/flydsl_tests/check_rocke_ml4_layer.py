#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check rocKE ML4 layer stages bit for bit against the FlyDSL ML4 mono layer.

One FlyDSL launch per kind on fresh state gives the golden scratch: every
stage's mailboxes, carrying this launch's tag. Each rocKE build (a group of
stages) then runs with the same arguments on a copy of that scratch whose
group outputs are cleared. Those regions must come back bit-identical, and no
other scratch byte may change.

The rocKE kernels compile in a subprocess (FlyDSL embeds its own LLVM) from
JSON specs written here. ``--rocke`` names the rocKE tree; its
``platform/python`` and ``library`` go on that process's PYTHONPATH.

    python check_rocke_ml4_layer.py --rocke $ROCKE --groups q_norm
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import struct
import subprocess
import sys

import torch

import bench_mono_ml4_layer as B

INDEX_Q_BYTES = lambda S: S * B.INDEX_HEADS * B.INDEX_DIM * 4  # noqa: E731


def _nope(kind):
    return 128 if B.KINDS[kind][0] else 64


# mailbox regions (and their byte sizes) each stage publishes; a q_b task of
# an indexer kind also writes index-query tiles, so check it with index_q
STAGE_OUT = {
    "qkv_a": lambda S, ix, kind: [("kv_a", S * B.QK * 8)]
    + ([("index_k", S * B.INDEX_DIM * 8), ("index_w", S * B.INDEX_HEADS * 8)] if ix else []),
    "q_norm": lambda S, ix, kind: [("q_an", S * B.HIDDEN * 4)]
    + ([("iq_an", S * B.HIDDEN * 4)] if ix else []),
    "q_b": lambda S, ix, kind: [
        ("q_nope", S * B.NH * _nope(kind) * 4),
        ("q_pe", S * B.NH * B.PE * 4),
    ]
    + ([("index_q", INDEX_Q_BYTES(S))] if ix else []),
    "index_q": lambda S, ix, kind: [("index_q", INDEX_Q_BYTES(S))],
    "cache": lambda S, ix, kind: [
        ("kvnew", S * B.KVL * 8),
        ("penew", S * B.PE * 8),
    ]
    + ([("index_k_new", S * B.INDEX_DIM * 4), ("index_ready", S * 8)] if ix else []),
    "uk": lambda S, ix, kind: [("q_lat", S * B.NH * B.KVL * 4)],
    "split": lambda S, ix, kind: [
        ("sp_acc", S * _n_split(S) * B.NH * B.KVL * 4),
        ("sp_m", S * _n_split(S) * B.NH * 8),
        ("sp_l", S * _n_split(S) * B.NH * 8),
    ],
    "uv": lambda S, ix, kind: [("o", S * B.NH * B.VD * 4)],
    "o": lambda S, ix, kind: [("a", S * B.HIDDEN * 4)],
    "router": lambda S, ix, kind: [
        ("scores", S * B.E * 8),
        ("xq", S * B.HIDDEN * 2),
        ("xqs", S * (B.HIDDEN // 32) * 8),
        ("xqd", S * B.HIDDEN * 4),
    ],
    "ug": lambda S, ix, kind: [
        ("mid", S * MOE_SLOTS * INTER * 8),
        ("sel", S * MOE_SLOTS * 8),
        ("prob", S * MOE_SLOTS * 8),
    ],
    "down": lambda S, ix, kind: [],
    "index_score": lambda S, ix, kind: [("index_scores", S * CAPACITY * 8)],
    "index_select": lambda S, ix, kind: [("indices", S * B.TOPK * 4), ("indices_ready", S * 8)],
}
MOE_SLOTS, INTER, CAPACITY = 5, 1536, 32768
SEG = 16384


def _tree_lists(cap, level):
    seg = min(cap, SEG)
    fan = seg // B.TOPK
    n = (cap + seg - 1) // seg
    for _ in range(level):
        n = (n + fan - 1) // fan
    return n


def full_order(stage_plan):
    """Every planned stage, in an order whose waits only point backwards."""
    names = [n for n, _, _ in stage_plan]
    levels = sorted((n for n in names if n.startswith("index_sel") and n != "index_select"),
                    key=lambda n: int(n[len("index_sel"):]))  # fmt: skip
    order = ["qkv_a", "q_norm", "q_b", "index_q", "cache", "uk", "index_score"]
    order += levels + ["index_select", "split", "uv", "o", "router", "ug", "down"]
    return [n for n in order if n in names]


def stage_regions(stage, S, ix, kind):
    if stage.startswith("index_sel") and stage != "index_select":
        level = int(stage[len("index_sel"):])
        # the level's score plane, then its identity plane
        return [(f"isel{level}", S * _tree_lists(CAPACITY, level) * B.TOPK * 2 * 8)]
    return STAGE_OUT[stage](S, ix, kind)


def _n_split(S):
    keys = 32 if B.NH > 16 or (B.NH == 8 and S > 4) else 64
    return B.TOPK // keys
# launch outputs outside scratch each stage writes
STAGE_GLOBALS = {"cache": ("kv_cache", "index_cache"), "down": ("x_out",)}


def _loaded_hip() -> str:
    """Path of the HIP runtime torch already mapped: a bare soname could
    load a second runtime next to it."""
    torch.cuda.init()
    with open("/proc/self/maps") as f:
        for line in f:
            if "libamdhip64.so" in line:
                return line.split()[-1]
    raise RuntimeError("libamdhip64 is not loaded in this process")


class Hip:
    """hipModule loading and launching through the process's HIP runtime."""

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


def pack_args(args) -> bytes:
    """The FlyDSL launcher's arguments as the rocKE kernel's kernarg block:
    36 64-bit addresses, then rank and layer."""
    ptrs, (rank, layer) = args[:-2], args[-2:]
    if len(ptrs) != 36:
        raise ValueError(f"expected 36 pointer arguments, got {len(ptrs)}")
    return struct.pack("<36Q2i", *[int(p) & (2**64 - 1) for p in ptrs], int(rank), int(layer))


def capture_launch(k, state, x_in, x_out):
    """Run one FlyDSL launch and return the arguments it was given."""
    seen = {}
    real = k.launch

    def spy(*args, **kw):
        seen["args"] = args
        return real(*args, **kw)

    k.launch = spy
    try:
        B.launch(k, state, x_in, x_out, 0)
    finally:
        k.launch = real
    return seen["args"]


def compile_specs(rocke, specs, out, flavor=""):
    """Build the specs in a fresh process. Unless ``flavor`` pins one, rocKE
    takes the IR flavor from the COMGR it compiles with (a pin inherited for
    linking into another compiler's module would not match it)."""
    paths = []
    for spec in specs:
        path = os.path.join(out, spec["name"] + ".json")
        with open(path, "w") as f:
            json.dump(spec, f, indent=1)
        paths.append(path)
    env = dict(os.environ, ROCKE_BACKEND="python")
    env.pop("ROCKE_LLVM_FLAVOR", None)
    if flavor:
        env["ROCKE_LLVM_FLAVOR"] = flavor
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(rocke, "platform", "python"), os.path.join(rocke, "library")]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    r = subprocess.run(
        [sys.executable, "-m", "kernels.gfx950.ml4_mono.layer.build", *paths, "--out", out],
        env=env, capture_output=True, text=True,
    )  # fmt: skip
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"rocKE build failed (rc {r.returncode})")
    return [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]


def check_tagged_indices(scr, golden, k, S, notes):
    """``sel_tagged``: rocKE also leaves each single-list row's selected
    positions, as ``(index, tag)`` pairs, in ``isel0``'s identity plane (the
    reference leaves those words alone). Every such pair must carry this
    launch's tag and the row's ``indices`` value, and they must form a prefix;
    the words are then restored to the reference's so the region compare
    covers the rest. Returns False on a mismatch."""
    if "isel0" not in k.scr_layout:
        return True
    slots = _tree_lists(CAPACITY, 0)
    tag = int(golden[k.scr_layout["indices_ready"]:][:8].view(torch.int32)[1])
    pairs = scr[k.scr_layout["isel0"]:].view(torch.int32)
    ref_pairs = golden[k.scr_layout["isel0"]:].view(torch.int32)
    idx = scr[k.scr_layout["indices"]:][: S * B.TOPK * 4].view(torch.int32).view(S, B.TOPK)
    ok = True
    for s in range(S):
        first = (S * slots * B.TOPK + s * slots * B.TOPK) * 2
        val = pairs[first : first + 2 * B.TOPK : 2]
        tg = pairs[first + 1 : first + 2 * B.TOPK : 2]
        written = tg == tag
        n = int(written.sum())
        if n and (not bool(written[:n].all()) or not torch.equal(val[:n], idx[s, :n])):
            ok = False
            notes.append(f"isel0 tagged indices of row {s}: {n} written, not the indices prefix")
        pairs[first : first + 2 * n].copy_(ref_pairs[first : first + 2 * n])
    return ok


def diff_words(got, want, limit=6):
    g, w = got.view(torch.int32), want.view(torch.int32)
    bad = (g != w).nonzero().flatten()
    lines = []
    for i in bad[:limit].tolist():
        gv, wv = int(g[i]), int(w[i])
        kind = "tag" if i % 2 else "payload"
        lines.append(f"word {i} ({kind}) got {gv & 0xFFFFFFFF:#010x} want {wv & 0xFFFFFFFF:#010x}")
    return int(bad.numel()), lines


def build_kernel(kind, S, args, rank=0, npes=1, group=None):
    """One FlyDSL ML4 layer kernel of this rank's shard, and its placement."""
    nope, kvs, with_indexer = B.KINDS[kind]
    v, W = B.build_weights(kind, rank)
    kmod = v.op._kernel_module
    k = v.op.Glm5MonoKernel(
        W, S, rank=rank, npes=npes, group=group, topk=B.TOPK, launches_per_step=71,
        with_indexer=with_indexer, index_max_seq=args.capacity, index_request_width=S,
        index_page_size=args.page, index_block_table_stride=args.capacity // args.page,
        index_cache_dtype="fp8", attention_weight="fp8_block128", kv_cache_layout="atom",
        kv_cache_dtype="fp8", native_fp4_mfma=True, rope_dtype="fp32",
        shared_indices=kind == "rope_shared", separate_shared=True,
    )  # fmt: skip
    return k, kmod.LAST_PLAN


def check_kind(kind, groups, args, hip, rank=0, npes=1, group=None):
    nope, kvs, with_indexer = B.KINDS[kind]
    S = args.rows
    k, plan = build_kernel(kind, S, args, rank, npes, group)
    overrides = json.loads(os.environ.get("AITER_ML4_ROCKE_SPEC") or "{}")
    rl = importlib.import_module(type(k).__module__.rsplit(".", 1)[0] + ".rocke_layer")
    numa_hdr = 0
    if rl.numa_on(dict(overrides, kind=kind)):
        # the launcher's geometry; the golden FlyDSL launch then runs on the
        # grown scratch, header and zeroed dual-copy area included
        numa_hdr = rl.layer_spec(k, plan, kind, 71)["numa_hdr"]
        need = rl.numa_scratch_bytes(numa_hdr)
        if k.scratch.numel() < need:
            k.scratch = torch.zeros(need, dtype=torch.uint8, device=k.scratch.device)
        rl.numa_prepare(k.scratch, numa_hdr, args.rocke)
    state = B.make_state(S, args.page, args.capacity, args.context)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    x_in = torch.randn(S, B.HIDDEN, generator=gen, device="cuda").bfloat16()
    x_out = torch.zeros(S, B.HIDDEN, dtype=torch.bfloat16, device="cuda")
    outputs = {"kv_cache": state["kv"], "index_cache": state["index_cache"], "x_out": x_out}
    before = {n: t.clone() for n, t in outputs.items()}
    k.advance_step()
    launch_args = capture_launch(k, state, x_in, x_out)
    torch.cuda.synchronize()
    golden = k.scratch.clone()
    after = {n: t.clone() for n, t in outputs.items()}
    stage_plan = [
        [name, int(n), int(plan["base"].get(name, 0))] for name, n in k.stages if n > 0
    ]
    scratch = [[n, int(o)] for n, o in k.scr_layout.items() if not n.startswith("_")]
    specs = []
    planned = {name for name, _, _ in stage_plan}
    sym = [[n, int(o)] for n, o in k.sym_layout.items() if isinstance(o, int)]
    for stage_group in groups:
        if numa_hdr and stage_group != "full":
            # a partial build would poll copies the FlyDSL launch never wrote
            print(f"[rk-check] rank {rank} {kind} {stage_group}: skipped (mb_numa checks whole layers)", flush=True)
            continue
        names = full_order(stage_plan) if stage_group == "full" else stage_group.split("+")
        stages = [st for st in names if st in planned]
        if not stages:
            continue
        specs.append(
            dict(
                kind=kind, rows=S, heads=k.W.heads, npes=npes, stages=stages,
                inter=int(k.W.config.inter), plan=stage_plan, scratch=scratch,
                sym=sym, launches_per_step=71,
                index_page_size=int(k.index_page_size),
                index_bt_stride=int(k.index_block_table_stride),
                index_request_width=int(k.index_request_width),
                index_capacity=int(k.index_max_seq),
                name=f"ml4rk_{kind}_r{S}_tp{npes}_" + "_".join(stages),
            )
        )  # fmt: skip
        # the launcher's schedule overrides apply here too
        specs[-1].update(overrides)
        if numa_hdr:
            specs[-1]["numa_hdr"] = numa_hdr
    out = os.path.join(args.out, f"{kind}_rank{rank}")
    os.makedirs(out, exist_ok=True)
    built_rk = compile_specs(args.rocke, specs, out, args.llvm_flavor)
    packed = pack_args(launch_args)
    results = {}
    for spec, art in zip(specs, built_rk):
        # moe_em (as built): up/gate also writes per-expert down partials
        # (FlyDSL has no such region: cleared, kept out of the stray check,
        # not compared), and x_out is re-associated: within 1 bf16 ulp of
        # FlyDSL's, not bitwise
        em = bool(art.get("moe_em"))
        if em and "down" in spec["stages"] and "ug" not in spec["stages"]:
            # its combine would wait on partials only the em up/gate writes
            print(f"[rk-check] rank {rank} {kind} {'+'.join(spec['stages'])}: skipped (moe_em: down needs ug)",
                  flush=True)  # fmt: skip
            continue
        with open(art["hsaco"], "rb") as f:
            fn = hip.function(f.read(), art["kernel"])
        regions = []
        for st in spec["stages"]:
            regions += [r for r in stage_regions(st, S, with_indexer, kind) if r not in regions]
        unchecked = set()
        if em and "ug" in spec["stages"]:
            regions.append(("dn_part", S * MOE_SLOTS * B.HIDDEN * 8))
            unchecked.add("dn_part")
        written = {g for st in spec["stages"] for g in STAGE_GLOBALS.get(st, ())}
        for name, t in outputs.items():
            t.copy_(before[name] if name in written else after[name])
        scr = k.scratch
        scr.copy_(golden)
        for name, nbytes in regions:
            off = k.scr_layout[name]
            scr[off : off + nbytes].zero_()
        cleared = scr.clone()
        if npes > 1:
            # peers must see this replay's pushes, not the FlyDSL launch's
            k.sym_storage.zero_()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=group)
        hip.launch(fn, 256, 512, packed, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        if npes > 1:
            torch.distributed.barrier(group=group)
        verdict, notes = "bitwise", []
        if spec.get("sel_tagged") and with_indexer and not check_tagged_indices(scr, golden, k, S, notes):
            verdict = "DIFFERS"
        for name, t in outputs.items():
            if torch.equal(t, after[name]):
                continue
            if em and name == "x_out":
                worst, n_diff = B.bf16_ulps(t, after[name])
                notes.append(f"x_out: {n_diff} of {t.numel()} values differ, "
                             f"at most {worst:.2f} bf16 ulp (tolerance 1)")  # fmt: skip
                if worst <= 1.0:
                    verdict = "tolerance" if verdict == "bitwise" else verdict
                    continue
            verdict = "DIFFERS"
            n_bad = int((t.view(torch.uint8) != after[name].view(torch.uint8)).sum())
            notes.append(f"{name}: {n_bad} bytes differ from the FlyDSL launch")
        outside = torch.ones_like(scr, dtype=torch.bool)
        for name, nbytes in regions:
            off = k.scr_layout[name]
            outside[off : off + nbytes] = False
            only = overrides.get("numa_regions") or ""
            dual = numa_hdr and name not in rl.NUMA_PLAIN and (not only or name in only.split("+"))
            if dual:
                # both copies' bytes are this region's
                outside[rl.numa_copy_bytes(scr, numa_hdr, off, nbytes)] = False
            if name in unchecked:
                # no golden: every pair must carry this launch's tag (the
                # golden mid's), i.e. the build wrote all of it
                tag = int(golden[k.scr_layout["mid"] + 4 : k.scr_layout["mid"] + 8].view(torch.int32).item())
                mine = rl.numa_logical(scr, numa_hdr, off, nbytes, 0) if dual else scr[off : off + nbytes]
                untagged = int((mine.view(torch.int32)[1::2] != tag).sum())
                notes.append(f"{name}: {nbytes // 8 - untagged} of {nbytes // 8} pairs written (not compared)")
                if untagged:
                    verdict = "DIFFERS"
                continue
            if dual:
                # copy 0 against the golden region, copy 1 against copy 0
                got = rl.numa_logical(scr, numa_hdr, off, nbytes, 0)
                other = rl.numa_logical(scr, numa_hdr, off, nbytes, 1)
                if not torch.equal(got, other):
                    verdict = "DIFFERS"
                    notes.append(f"{name}: the NUMA copies differ in {int((got != other).sum())} bytes")
            else:
                got = scr[off : off + nbytes]
            n_bad, lines = diff_words(got, golden[off : off + nbytes])
            if n_bad:
                verdict = "DIFFERS"
                notes.append(f"{name}: {n_bad} of {nbytes // 4} words differ")
                notes += ["  " + x for x in lines]
        stray = int(((scr != cleared) & outside).sum())
        if stray:
            verdict = "DIFFERS"
            notes.append(f"{stray} scratch bytes outside the outputs changed")
        results[spec["name"]] = {"verdict": verdict, "notes": notes, "compile_ms": art["ms"]}
        print(f"[rk-check] rank {rank} {kind} {'+'.join(spec['stages'])}: {verdict} "
              f"(rocKE compile {art['ms']:.0f} ms, {art.get('resources')})", flush=True)  # fmt: skip
        for x in notes:
            print(f"[rk-check]   {x}", flush=True)
    k.close()
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kinds", default="rope,rope_shared,nope,nope_kvs")
    p.add_argument("--groups", default="q_norm", help="comma list of stage groups (a+b)")
    p.add_argument("--rocke", default=os.environ.get("ROCKE_ROOT", ""))
    p.add_argument("--llvm-flavor", default="", help="pin rocKE's IR flavor")
    p.add_argument("--out", default="rk_ml4_out")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--page", type=int, default=16)
    p.add_argument("--capacity", type=int, default=32768)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--report", default="")
    p.add_argument("--npes", type=int, default=1, help="TP ranks, one GPU each")
    args = p.parse_args(argv)
    if not args.rocke:
        p.error("--rocke (or ROCKE_ROOT) is required")
    if args.npes == 1:
        return run_rank(0, args)
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, args)) for r in range(args.npes)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    return max(pr.exitcode or 0 for pr in procs)


def _rank_main(rank, args):
    sys.exit(run_rank(rank, args))


def run_rank(rank, args):
    torch.set_grad_enabled(False)
    group = None
    if args.npes > 1:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29571", world_size=args.npes, rank=rank
        )
        group = dist.group.WORLD
    hip = Hip()
    report = {}
    for kind in args.kinds.split(","):
        report[kind] = check_kind(kind, args.groups.split(","), args, hip, rank, args.npes, group)
        torch.cuda.empty_cache()
    if args.report:
        path = args.report if args.npes == 1 else args.report.replace(".json", f"_rank{rank}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=1)
    bad = [n for r in report.values() for n, e in r.items() if e["verdict"] not in ("bitwise", "tolerance")]
    print(f"[rk-check] rank {rank} {'PASS' if not bad else 'FAIL: ' + ', '.join(bad)}", flush=True)
    if group is not None:
        torch.distributed.destroy_process_group()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
