#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the rocKE ML4 draft-layer kind against a PyTorch model of vLLM's
DSpark draft layer.

Each rank runs its shard of a TP draft layer (16 heads and 6144 MLP rows a
rank, the TP4 shard) with ``--npes`` ranks, one GPU each, the peer reduces
live. The weights are random bf16 rounded through block FP8 (the kernel's
format; the reference multiplies by the dequantized values), the paged bf16
latent cache holds ``--context`` random rows before the draft block's 4, and
attention covers the last ``--window`` positions. Every stage's published
outputs are compared with the reference applied to the kernel's own inputs
of that stage (``o`` and ``mlp_down`` with every rank's inputs), so a stage's
error is its own; ``x_out`` also against the reference chain from ``h_in``
alone, and with TP across the ranks bit for bit.

    python check_rocke_ml4_draft.py --rocke $ROCKE --out DIR [--npes 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

import check_rocke_ml4_layer as C
from aiter.ops.flydsl.kernels.mono.mistral_large4.gfx950 import layout as L
from aiter.ops.flydsl.mono.runtime.peer_memory import SymmetricPeerBuffer
from aiter.ops.flydsl.mono.weights.packing import pack_fp8

S, H, NOPE, PE, KVL, VD, HID, INTER = 4, 16, 64, 64, 512, 128, 6144, 6144
QK = KVL + PE
TOPK = 2048
EPS = 1e-5
SCALE = 128**-0.5
FP8 = torch.float8_e4m3fn
# fp32 parameter table: SSMax s and b by head (s = 0, b = 1: the identity)
P_SS_S = 2 * HID
P_SS_B = P_SS_S + 64
ORDER = ["qkv_a", "q_norm", "q_b", "cache", "uk", "split", "uv", "o", "mlp_ug", "mlp_down"]
# (stage, tasks, first CTA): the attention chain spread over the grid, the
# MLP on every CTA
PLAN = [
    ("qkv_a", QK // 16, 0), ("q_norm", S, 36), ("cache", 1, 40), ("q_b", H * 128 // 16, 41),
    ("uk", H * KVL // 128, 169), ("split", S * TOPK // 64, 0), ("uv", S * H * VD // 64, 128),
    ("o", HID // 32, 0), ("mlp_ug", 2 * INTER // 48, 0), ("mlp_down", HID // 32, 64),
]  # fmt: skip
# relative L2 error bounds: a stage against the reference on its own inputs,
# and x_out against the reference chain
STAGE_TOL, E2E_TOL = 1e-2, 2e-2


def q128(w: torch.Tensor, bc: int = 128):
    """bf16 [r, c] -> (E4M3 [r, c], fp32 scales per 128 x bc block, the
    dequantized fp32 [r, c])."""
    r, c = w.shape
    wp = F.pad(w.float(), (0, (-c) % bc, 0, (-r) % 128))
    blk = wp.view(wp.shape[0] // 128, 128, wp.shape[1] // bc, bc)
    s = blk.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (blk / s[:, None, :, None]).clamp(-448, 448).to(FP8)
    dq = (q.float() * s[:, None, :, None]).view_as(wp)[:r, :c]
    return q.view(wp.shape)[:r, :c].contiguous(), s.contiguous(), dq.contiguous()


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.bfloat16().float()


def rms(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return bf(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g.float())


def rope_rows(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, per_product: bool) -> torch.Tensor:
    """GPT-J pairs of bf16 ``x`` [..., 64]; ``per_product``: each product
    rounded to bf16 first (``MistralRotaryEmbedding.rotate_``)."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    r = bf if per_product else (lambda v: v)
    out = torch.empty_like(x)
    out[..., 0::2] = bf(r(x0 * cos) - r(x1 * sin))
    out[..., 1::2] = bf(r(x1 * cos) + r(x0 * sin))
    return out


def rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-30))


def make_layer(gen, npes: int):
    """The whole TP layer's bf16 weights and norms (every rank draws the same)."""

    def w(n, k):
        return (torch.randn(n, k, generator=gen, device="cuda") * k**-0.5).bfloat16()

    def g(n):
        return (1.0 + 0.1 * torch.randn(n, generator=gen, device="cuda")).bfloat16()

    full = {
        "q_b": w(npes * H * (NOPE + PE), HID), "qkv_a": w(QK, HID), "kv_b": w(npes * H * (NOPE + VD), KVL),
        "o": w(HID, npes * H * VD), "ug": w(2 * npes * INTER, HID), "dn": w(HID, npes * INTER),
    }  # fmt: skip
    return full, {"g_in": g(HID), "g_kv": g(KVL), "g_post": g(HID)}


def shard(full, r: int, npes: int):
    """Rank ``r``'s bf16 weights in the kernel's layouts: its heads' query rows
    and W_UK^T / W_UV, its W_o columns, its [gate; up] rows, its down columns."""
    kvb = full["kv_b"].float().view(npes * H, NOPE + VD, KVL)[r * H : (r + 1) * H]
    up0 = npes * INTER
    return {
        "q_b": full["q_b"][r * H * (NOPE + PE) : (r + 1) * H * (NOPE + PE)],
        "qkv_a": full["qkv_a"],
        "uk": kvb[:, :NOPE, :].transpose(1, 2).reshape(H * KVL, NOPE).bfloat16(),
        "uv": kvb[:, NOPE:, :].reshape(H * VD, KVL).bfloat16(),
        "o": full["o"][:, r * H * VD : (r + 1) * H * VD].contiguous(),
        "ug": torch.cat([full["ug"][r * INTER : (r + 1) * INTER], full["ug"][up0 + r * INTER : up0 + (r + 1) * INTER]]),
        "dn": full["dn"][:, r * INTER : (r + 1) * INTER].contiguous(),
    }


def interleave(ug: torch.Tensor) -> torch.Tensor:
    """[gate; up] rows -> 16-row tiles of 8 gate rows and the same 8 up rows
    (``mlp_act``)."""
    n = ug.shape[0] // 2
    return torch.stack([ug[:n].view(n // 8, 8, -1), ug[n:].view(n // 8, 8, -1)], 1).reshape(2 * n, -1)


def deinterleave(w: torch.Tensor) -> torch.Tensor:
    n = w.shape[0] // 2
    t = w.view(n // 8, 2, 8, -1)
    return torch.cat([t[:, 0].reshape(n, -1), t[:, 1].reshape(n, -1)])


def act_of(gu: torch.Tensor) -> torch.Tensor:
    """bf16(SiLU(gate) * up) of the bf16 [gate | up] GEMM outputs."""
    g, u = gu[:, :INTER], gu[:, INTER:]
    return bf(g / (1 + torch.exp(-g)) * u)


def tp_sum(parts, npes: int) -> torch.Tensor:
    """``peer_reduce``'s sum: one rank's own fp32 partial; with peers the
    bf16-rounded partials in rank order."""
    if npes == 1:
        return parts[0]
    tot = torch.zeros_like(parts[0])
    for p in parts:
        tot = tot + bf(p)
    return tot


def pairs(scratch, off, n):
    """(payload int32 [n], tag int32 [n]) of the n tagged pairs at ``off``."""
    w = scratch[off : off + 8 * n].view(torch.int32).view(n, 2)
    return w[:, 0].contiguous(), w[:, 1].contiguous()


def gather(t: torch.Tensor, npes: int, group):
    if npes == 1:
        return [t]
    import torch.distributed as dist

    out = [torch.empty_like(t.cpu()) for _ in range(npes)]
    dist.all_gather(out, t.cpu().contiguous(), group=group)
    return [x.to(t.device) for x in out]


def run_rank(rank: int, args) -> int:
    torch.set_grad_enabled(False)
    npes, group = args.npes, None
    if npes > 1:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{args.port}", world_size=npes, rank=rank)
        group = dist.group.WORLD
    dev = "cuda"
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)

    # ---------------------------------------------------------------- state
    overrides = json.loads(os.environ.get("AITER_ML4_ROCKE_SPEC") or "{}")
    act = bool(overrides.get("mlp_act"))
    full, gam = make_layer(gen, npes)
    dq_all, mine = [], None
    for r in range(npes):
        sh = shard(full, r, npes)
        if act:
            sh["ug"] = interleave(sh["ug"])
        qsd = {n: q128(t, 64 if n == "uk" else 128) for n, t in sh.items()}
        dq = {n: v[2] for n, v in qsd.items()}
        if act:
            dq["ug"] = deinterleave(dq["ug"])
        dq_all.append(dq)
        if r == rank:
            mine = qsd
    del full
    q = {n: v[0] for n, v in mine.items()}
    s = {n: v[1] for n, v in mine.items()}
    dq = dq_all[rank]
    page = args.page
    n_pos = args.context + S
    pages = (n_pos + page - 1) // page + 2
    bt = torch.randperm(pages, generator=torch.Generator().manual_seed(args.seed))[: (n_pos + page - 1) // page]
    bt_stride = bt.numel()
    block_table = bt.to(torch.int32).to(dev).view(1, bt_stride)
    kv_cache = torch.randn(pages * page, QK, generator=gen, device=dev).bfloat16()
    positions = torch.arange(args.context, n_pos, device=dev, dtype=torch.int64)
    slot_of = lambda pos: block_table[0, pos // page].long() * page + pos % page  # noqa: E731
    slot_mapping = slot_of(positions)
    n_keys = min(args.window, n_pos)
    start = n_pos - n_keys
    indices = torch.zeros(S, TOPK, dtype=torch.int32, device=dev)
    indices[:, :n_keys] = torch.arange(start, n_pos, dtype=torch.int32, device=dev)
    indptr = torch.arange(S + 1, dtype=torch.int32, device=dev) * n_keys
    inv_freq = 10000.0 ** (-torch.arange(0, PE, 2, dtype=torch.float32, device=dev) / PE)
    ang = torch.arange(n_pos + 8, device=dev, dtype=torch.float32)[:, None] * inv_freq[None]
    rope_cos, rope_sin = bf(torch.cos(ang)).contiguous(), bf(torch.sin(ang)).contiguous()
    g_q = torch.zeros(P_SS_B + 64, dtype=torch.float32, device=dev)
    g_q[P_SS_B : P_SS_B + H] = 1.0
    h_in = torch.randn(S, HID, generator=gen, device=dev).bfloat16()
    x_out = torch.zeros(S, HID, dtype=torch.bfloat16, device=dev)
    scr, sym = L.layout(S, H, npes, TOPK, with_indexer=False, inter=INTER)
    scratch = torch.zeros(scr["_bytes"], dtype=torch.uint8, device=dev)
    peer = SymmetricPeerBuffer(max(sym["_bytes"], 64), rank=rank, npes=npes, group=group)
    table = torch.zeros(16, dtype=torch.int64, device=dev)
    # timeline builds: TL_COLS int64 stamps a task, rows in plan order
    timeline = torch.zeros(sum(n for _, n, _ in PLAN) * 8, dtype=torch.int64, device=dev)
    table[7] = timeline.data_ptr()
    step = torch.ones(1, dtype=torch.int32, device=dev)
    dummy = torch.zeros(1024, dtype=torch.float32, device=dev)
    packed = {name: pack_fp8(q[name]) for name in q}
    cache0 = kv_cache.clone()

    # ---------------------------------------------------------------- build
    spec = dict(
        kind="draft", rows=S, heads=H, npes=npes, inter=INTER, stages=ORDER,
        plan=[list(p) for p in PLAN],
        scratch=[[n, int(o)] for n, o in scr.items() if not n.startswith("_")],
        sym=[[n, int(o)] for n, o in sym.items() if isinstance(o, int)],
        launches_per_step=3, index_page_size=page, index_bt_stride=bt_stride,
        index_request_width=S, name=f"ml4rk_draft_r{S}_tp{npes}", timeline=args.timeline,
    )  # fmt: skip
    spec.update(overrides)
    out_dir = os.path.join(args.out, f"rank{rank}")
    os.makedirs(out_dir, exist_ok=True)
    art = C.compile_specs(args.rocke, [spec], out_dir, args.llvm_flavor)[0]
    if rank == 0:
        print(f"[draft-check] built {art['kernel']} in {art['ms']:.0f} ms, {art.get('resources')}", flush=True)
    hip = C.Hip()
    with open(art["hsaco"], "rb") as f:
        fn = hip.function(f.read(), art["kernel"])
    ptr = lambda t: t.data_ptr()  # noqa: E731
    launch = [
        ptr(h_in), ptr(x_out), ptr(block_table), ptr(positions), ptr(slot_mapping), ptr(indptr),
        ptr(kv_cache), ptr(kv_cache), ptr(indices), ptr(rope_cos), ptr(rope_sin),
        ptr(gam["g_in"]), ptr(g_q), ptr(gam["g_kv"]), ptr(gam["g_post"]),
        ptr(packed["qkv_a"]), ptr(s["qkv_a"]), ptr(packed["q_b"]), ptr(s["q_b"]),
        ptr(packed["uk"]), ptr(s["uk"]), ptr(packed["uv"]), ptr(s["uv"]),
        ptr(packed["o"]), ptr(s["o"]), ptr(dummy), ptr(dummy),
        ptr(packed["ug"]), ptr(s["ug"]), ptr(packed["dn"]), ptr(s["dn"]),
        ptr(scratch), peer.local_address, ptr(peer.addresses), ptr(table), ptr(step), rank, 0,
    ]  # fmt: skip
    if group is not None:
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
    hip.launch(fn, 256, 512, C.pack_args(launch), torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # ---------------------------------------------------------------- outputs
    notes, errs = [], {}
    tag = None

    def region(name, n):
        nonlocal tag
        val, tg = pairs(scratch, scr[name], n)
        if tag is None:
            # the scratch starts zeroed: a zero tag is no launch's
            tag = int(tg[0])
            if tag == 0:
                notes.append(f"{name}: no pair carries a launch tag")
        missing = int((tg != tag).sum())
        if missing:
            notes.append(f"{name}: {missing} of {n} pairs not written this launch")
        return val

    def f32(name, n):
        return region(name, n).view(torch.float32)

    def bf16s(name, n):
        return region(name, n // 2).view(torch.bfloat16).float()

    q_an = bf16s("q_an", S * HID).view(S, HID)
    kv_a = f32("kv_a", S * QK).view(S, QK)
    q_nope = bf16s("q_nope", S * H * NOPE).view(S, H, NOPE)
    q_pe = bf16s("q_pe", S * H * PE).view(S, H, PE)
    kvnew = f32("kvnew", S * KVL).view(S, KVL)
    penew = f32("penew", S * PE).view(S, PE)
    q_lat = bf16s("q_lat", S * H * KVL).view(S, H, KVL)
    o = bf16s("o", S * H * VD).view(S, H * VD)
    a = bf16s("a", S * HID).view(S, HID)
    # mlp_act: the activations as bf16 pairs, else the gate and up values
    mid = bf16s("mid", S * INTER).view(S, INTER) if act else f32("mid", S * 2 * INTER).view(S, 2 * INTER)
    xo = x_out.float()
    o_all, mid_all = gather(o, npes, group), gather(mid, npes, group)

    # ---------------------------------------------------------------- reference
    h = h_in.float()
    cos, sin = rope_cos[positions], rope_sin[positions]

    def attention(w, ql, qp, kvn, pen):
        """One rank's o rows [S, H * VD] from its latent queries and the
        block's rows; the window's older keys from the cache as the launch
        found it."""
        keys = torch.arange(start, n_pos, device=dev)
        rows = cache0[slot_of(keys)].float()
        rows[-S:, :KVL], rows[-S:, KVL:] = kvn, pen
        sc = (torch.einsum("shl,kl->shk", ql, rows[:, :KVL]) + torch.einsum("shp,kp->shk", qp, rows[:, KVL:])) * SCALE
        lat = bf(torch.einsum("shk,kl->shl", torch.softmax(sc, -1), rows[:, :KVL]))
        return bf(torch.einsum("shl,hvl->shv", lat, w["uv"].view(H, VD, KVL))).reshape(S, H * VD)

    def q_of(w, xn):
        qq = (xn @ w["q_b"].T).view(S, H, NOPE + PE)
        return bf(qq[..., :NOPE]), rope_rows(bf(qq[..., NOPE:]), cos[:, None], sin[:, None], False)

    def cache_of(w, xn_kva):
        kva = bf(xn_kva)
        return rms(kva[:, :KVL], gam["g_kv"]), rope_rows(kva[:, KVL:], cos, sin, True)

    def lat_of(w, nope):
        return bf(torch.einsum("shd,hld->shl", nope, w["uk"].view(H, KVL, NOPE)))

    def resid_o(os_):
        return bf(h + tp_sum([oo @ w["o"].T for oo, w in zip(os_, dq_all)], npes))

    def mid_of(w, a_):
        gu = bf(rms(a_, gam["g_post"]) @ w["ug"].T)
        return act_of(gu) if act else gu

    def resid_dn(a_, mids):
        acts = mids if act else [act_of(m) for m in mids]
        return bf(a_ + tp_sum([x @ w["dn"].T for x, w in zip(acts, dq_all)], npes))

    # each stage on the kernel's inputs of that stage
    ref_nope, ref_pe = q_of(dq, q_an)
    ref_kvn, ref_pen = cache_of(dq, kv_a)
    stage = {
        "q_norm": (q_an, rms(h, gam["g_in"])),
        "qkv_a": (kv_a, q_an @ dq["qkv_a"].T),
        "q_b nope": (q_nope, ref_nope),
        "q_b rope": (q_pe, ref_pe),
        "cache kv": (kvnew, ref_kvn),
        "cache pe": (penew, ref_pen),
        "cache rows": (kv_cache[slot_mapping].float(), torch.cat([kvnew, penew], -1)),
        "uk": (q_lat, lat_of(dq, q_nope)),
        "split+uv": (o, attention(dq, q_lat, q_pe, kvnew, penew)),
        "o": (a, resid_o(o_all)),
        "mlp_ug": (mid, mid_of(dq, a)),
        "mlp_down": (xo, resid_dn(a, mid_all)),
    }
    for name, (got, want) in stage.items():
        errs[name] = rel(got, want)
    # the chain from h_in alone, over every rank's shard
    xn = rms(h, gam["g_in"])
    kvn_r, pen_r = cache_of(dq, xn @ dq["qkv_a"].T)
    o_r = []
    for w in dq_all:
        nope_r, pe_r = q_of(w, xn)
        o_r.append(attention(w, lat_of(w, nope_r), pe_r, kvn_r, pen_r))
    a_r = resid_o(o_r)
    x_r = resid_dn(a_r, [mid_of(w, a_r) for w in dq_all])
    errs["x_out end to end"] = rel(xo, x_r)
    untouched = torch.ones(kv_cache.shape[0], dtype=torch.bool, device=dev)
    untouched[slot_mapping] = False
    if not torch.equal(kv_cache[untouched], cache0[untouched]):
        notes.append("cache rows outside the block's slots changed")
    if npes > 1:
        xs = gather(x_out, npes, group)
        if not all(torch.equal(x.view(torch.int16), xs[0].view(torch.int16)) for x in xs):
            notes.append("x_out differs across the ranks")

    ok = not notes
    for name, e in errs.items():
        tol = E2E_TOL if name.endswith("end to end") else STAGE_TOL
        ok &= e < tol
        if rank == 0:
            print(f"[draft-check] {name:18s} relative error {e:.2e}  ({'ok' if e < tol else 'FAIL'}, < {tol:.0e})",
                  flush=True)  # fmt: skip
    for n in notes:
        print(f"[draft-check]   rank {rank}: {n}", flush=True)
    verdict = "PASS" if ok else "FAIL"
    print(f"[draft-check] rank {rank} {verdict}  (tag {tag:#x}, window {n_keys} keys from {start})", flush=True)
    timing = bench(hip, fn, launch, step, args.bench, npes, group) if args.bench and ok else None
    if args.timeline and rank == 0:
        # the last launch's stamps (the check's, without --bench)
        stage_times(timeline)
    if args.report:
        path = args.report if npes == 1 else args.report.replace(".json", f"_rank{rank}.json")
        with open(path, "w") as f:
            json.dump({"verdict": verdict, "errors": errs, "notes": notes, "resources": art.get("resources"),
                       "compile_ms": art["ms"], "us": timing}, f, indent=1)  # fmt: skip
    peer.close()
    if group is not None:
        torch.distributed.destroy_process_group()
    return 0 if ok else 1


def stage_times(timeline: torch.Tensor) -> None:
    """Per stage of a timeline launch: its first task start and last task end
    since the launch's first stamp, and its median task time (s_memrealtime,
    100 MHz)."""
    tl = timeline.view(-1, 8).cpu()
    t0 = int(tl[:, 0][tl[:, 0] > 0].min())
    first = 0
    print("[draft-check] stage       tasks   start     end  median task  (us)", flush=True)
    for name, n, _ in PLAN:
        rows = tl[first : first + n]
        first += n
        st, en = rows[:, 0] - t0, rows[:, 4] - t0
        dur = (rows[:, 4] - rows[:, 0]).float().median()
        line = (f"[draft-check] {name:10s} {n:6d} {float(st.min()) / 100:7.1f} {float(en.max()) / 100:7.1f} "
                f"{float(dur) / 100:8.1f}")  # fmt: skip
        inner = [c for c in (1, 2, 3) if bool((rows[:, c] > 0).all())]
        if inner:
            # the stage's inner stamps (tl_inner): each phase's median, and
            # when its last task reached the stamp
            cols = [0] + inner + [4]
            phases = [float((rows[:, b] - rows[:, a]).float().median()) / 100 for a, b in zip(cols, cols[1:])]
            reached = [float((rows[:, c] - t0).max()) / 100 for c in inner]
            line += ("   phases " + " / ".join(f"{p:.1f}" for p in phases)
                     + "   last at " + " / ".join(f"{r:.1f}" for r in reached))  # fmt: skip
        print(line, flush=True)


def bench(hip, fn, launch, step, n, npes, group):
    """GPU time of ``n`` launches, each with a fresh tag (the step advanced on
    the device) and a cold last-level cache: serving streams every other
    layer's weights between two launches of one draft layer, so none of its
    weights stay cached. With peers the launches pace each other through
    their reduces; the slowest rank's median is the layer's time."""
    stream = torch.cuda.current_stream()
    flush = torch.empty(1 << 29, dtype=torch.uint8, device="cuda")
    args = C.pack_args(launch)
    if group is not None:
        torch.cuda.synchronize()
        torch.distributed.barrier(group=group)
    events = []
    for i in range(n + 3):
        step.add_(1)
        flush.fill_(i & 0xFF)
        if npes == 1:
            # the GPU is busy while the CPU enqueues the events and the launch
            torch.cuda._sleep(2_000_000)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(stream)
        hip.launch(fn, 256, 512, args, stream.cuda_stream)
        e1.record(stream)
        events.append((e0, e1))
        if npes == 1:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    times = sorted(e0.elapsed_time(e1) * 1e3 for e0, e1 in events[3:])
    med = times[len(times) // 2]
    meds = [torch.tensor([med])] * npes if npes == 1 else None
    if npes > 1:
        import torch.distributed as dist

        meds = [torch.zeros(1) for _ in range(npes)]
        dist.all_gather(meds, torch.tensor([med]), group=group)
    worst = max(float(m) for m in meds)
    if torch.cuda.current_device() == 0:
        print(f"[draft-check] {n} cold-cache launches: median {med:.1f} us (slowest rank {worst:.1f}), "
              f"min {times[0]:.1f} us, max {times[-1]:.1f} us", flush=True)  # fmt: skip
    return {"median": med, "slowest_rank_median": worst, "min": times[0], "max": times[-1]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rocke", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--npes", type=int, default=1, help="TP ranks, one GPU each")
    p.add_argument("--context", type=int, default=3000, help="cached positions before the block")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--page", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--report", default="")
    p.add_argument("--llvm-flavor", default="")
    p.add_argument("--bench", type=int, default=0, help="time this many launches after the check")
    p.add_argument("--timeline", action="store_true", help="a timeline build: per-stage times of the last launch")
    args = p.parse_args(argv)
    if args.npes == 1:
        return run_rank(0, args)
    import socket

    import torch.multiprocessing as mp

    # a free port: jobs on one node must not share the rendezvous
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        args.port = sock.getsockname()[1]
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_rank_main, args=(r, args)) for r in range(args.npes)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    return max(pr.exitcode or 0 for pr in procs)


def _rank_main(rank, args):
    sys.exit(run_rank(rank, args))


if __name__ == "__main__":
    raise SystemExit(main())
