#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time one TP4 rank's ML4 mono layer on one GPU and break it into stages.

Each kind runs as one TP4 rank's shard with ``npes=1`` (no peer reduce), with
random weights in the production layouts, page-16 caches sized to the
serving index capacity and one DSpark verify block of ``--rows`` rows.

- ``graph``: ``--launches`` back-to-back launches captured in one HIP graph,
  cycling ``--copies`` independent weight sets and eight inputs so neither L2
  nor MALL holds a launch's weights; reported per launch.
- ``timeline``: a ``timeline=True`` build (``s_memrealtime`` stamps per task:
  start, inputs staged, compute done, end) gives every stage's span, input
  wait and compute, relative to the first task start of the launch.
- ``bytes``: weight, cache and expert bytes one launch must read (the routed
  experts from the union the router picked), and the bandwidth that implies.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time

import torch

FP8 = torch.float8_e4m3fn
HIDDEN, NH, KVL, PE, VD, E = 6144, 16, 512, 64, 128, 128
QK = KVL + PE
TOPK = 2048
INDEX_HEADS, INDEX_DIM = 32, 128
TL_COLS = 8
TICK_US = 0.01  # s_memrealtime runs at 100 MHz
# (nope, kv_share, with_indexer), as vLLM runtime.weights.KINDS
KINDS = {
    "rope": (False, False, True),
    "rope_shared": (False, False, False),
    "nope": (True, False, True),
    "nope_kvs": (True, True, True),
}
# layer launches per step in the served model
MODEL_MIX = {"rope": 25, "rope_shared": 28, "nope": 9, "nope_kvs": 9}


def _randn(g, *shape, scale=1.0):
    return torch.randn(*shape, generator=g, device="cuda") * scale


def _q128(w, block_cols=128):
    import torch.nn.functional as F

    r, c = w.shape
    w = w.float()
    wp = F.pad(w, (0, (-c) % block_cols, 0, (-r) % 128))
    b = wp.view(wp.shape[0] // 128, 128, wp.shape[1] // block_cols, block_cols)
    s = b.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (b / s[:, None, :, None]).clamp(-448, 448).view_as(wp)[:r, :c].to(FP8)
    return q.contiguous(), s.contiguous()


def _shuffle_mxfp4(values):
    from aiter.ops.shuffle import shuffle_weight

    return shuffle_weight(values, (16, 16))


def build_weights(kind, seed):
    """(variant, LayerWeights) of one random layer in the served layouts."""
    from aiter.ops.flydsl.mono.variants import load_variant
    from aiter.ops.flydsl.mono.weights.config import Mxfp4ScaleLayout, Mxfp4WeightLayout
    from aiter.ops.flydsl.mono.weights.layer import LayerWeights
    from aiter.utility.fp4_utils import e8m0_shuffle

    nope, kvs, _ = KINDS[kind]
    g = torch.Generator(device="cuda")
    g.manual_seed(1000 + seed)
    v = load_variant(nope, kvs)
    C = v.geometry_config.glm5_tp_config(4)
    bf, f32, u8 = torch.bfloat16, torch.float32, torch.uint8
    t = {}
    t["w_q_b"], t["s_q_b"] = _q128(_randn(g, NH * 128, HIDDEN, scale=HIDDEN**-0.5))
    wkva = _randn(g, KVL + PE, HIDDEN, scale=HIDDEN**-0.5)
    if nope:
        wkva[KVL:] = 0
    t["w_qkv_a"], t["s_qkv_a"] = _q128(wkva * (not kvs))
    t["g_kv"] = (torch.rand(KVL, generator=g, device="cuda") + 0.5).to(bf)
    wkvb = _randn(g, NH, C.nope_dim + VD, KVL, scale=KVL**-0.5)
    w_uk = wkvb[:, : C.nope_dim].transpose(1, 2).reshape(NH * KVL, C.nope_dim)
    t["w_uk"], t["s_uk"] = _q128(w_uk, 64)
    t["w_uv"], t["s_uv"] = _q128(wkvb[:, C.nope_dim :].reshape(NH * VD, KVL))
    t["w_o"], t["s_o"] = _q128(_randn(g, HIDDEN, NH * VD, scale=(NH * VD) ** -0.5))
    t["g_in"] = t["g_post"] = torch.ones(HIDDEN, dtype=bf, device="cuda")
    t["w_r"] = _randn(g, E, HIDDEN, scale=HIDDEN**-0.5).to(bf)
    t["bias"] = torch.zeros(E, dtype=f32, device="cuda")
    gc = v.geometry_config
    prm = torch.zeros(gc.ML4_P_WORDS, dtype=f32, device="cuda")
    prm[gc.ML4_P_SS_S : gc.ML4_P_SS_S + NH] = 1.0
    prm[gc.ML4_P_IQ_W : gc.ML4_P_IQ_W + HIDDEN] = 1.0
    prm[gc.ML4_P_ISS_S : gc.ML4_P_ISS_S + INDEX_HEADS] = 1.0
    t["g_q"] = prm
    t["w_index_q"], t["s_index_q"] = _q128(
        _randn(g, INDEX_HEADS * INDEX_DIM, HIDDEN, scale=HIDDEN**-0.5)
    )
    kw = _randn(g, INDEX_DIM + INDEX_HEADS, HIDDEN, scale=HIDDEN**-0.5)
    t["w_index_k"], t["s_index_k"] = _q128(kw[:INDEX_DIM])
    t["w_index_w"] = kw[INDEX_DIM:].to(bf).contiguous()
    t["g_index_k"] = torch.ones(INDEX_DIM, dtype=f32, device="cuda")
    t["b_index_k"] = torch.zeros(INDEX_DIM, dtype=f32, device="cuda")
    # Routed experts: random MXFP4 codes with small E8M0 scales, so expert
    # outputs stay finite; their values do not matter for timing.
    for w, s, rows, cols in (
        ("w_ug", "s_ug", 2 * C.inter, HIDDEN),
        ("w_dn", "s_dn", HIDDEN, C.inter),
    ):
        values = torch.randint(
            0, 256, (E, rows, cols // 2), dtype=u8, device="cuda", generator=g
        )
        scales = torch.randint(
            110, 114, (E * rows, cols // 32), dtype=u8, device="cuda", generator=g
        )
        t[s] = e8m0_shuffle(scales).view(u8)
        t[w] = _shuffle_mxfp4(values)
        del values
    t["w_ug"].is_shuffled = t["w_dn"].is_shuffled = True
    t["w_ug_sh"] = t["w_ug"][0].clone()
    t["s_ug_sh"] = t["s_ug"][: 2 * C.inter].clone()
    t["w_dn_sh"] = t["w_dn"][0].clone()
    t["s_dn_sh"] = t["s_dn"][:HIDDEN].clone()
    W = LayerWeights(
        NH,
        t,
        C,
        0,
        1,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
        physical_experts=E,
    )
    return v, W


def make_state(rows, page, capacity, context, seed=1):
    """Paged fp8 latent and index caches, one request's block table and the
    step tensors for ``rows`` consecutive verify rows ending the context."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    blocks = capacity // page
    kv = (_randn(g, blocks, page, QK, scale=0.5)).to(FP8)
    keys = _randn(g, blocks, page * 128, scale=0.5).to(FP8).view(torch.uint8)
    scales = torch.ones(blocks, page, device="cuda").view(torch.uint8)
    index_cache = torch.cat([keys, scales], 1).view(blocks, page, 132).contiguous()
    pages = torch.randperm(blocks, device="cuda", dtype=torch.int32, generator=g)
    positions = torch.arange(context, context + rows, device="cuda")
    slots = pages[positions // page].long() * page + positions % page
    nkeys = torch.clamp(positions + 1, max=TOPK)
    indptr = torch.cat([nkeys.new_zeros(1), nkeys.cumsum(0)]).int()
    freqs = 10000 ** (-torch.arange(32, device="cuda") / 32)
    ang = torch.arange(capacity, device="cuda")[:, None] * freqs[None]
    # rope_shared: the logical top-k a preceding indexer layer published
    indices = torch.stack(
        [
            torch.randperm(int(p) + 1, device="cuda", generator=g)[:TOPK]
            .sort()
            .values.int()
            for p in positions.tolist()
        ]
    ).contiguous()
    return dict(
        kv=kv,
        index_cache=index_cache,
        block_tables=pages[None].contiguous(),
        positions=positions,
        slots=slots,
        indptr=indptr,
        cur_pos=torch.tensor([context], dtype=torch.int32, device="cuda"),
        cos=torch.cos(ang).contiguous(),
        sin=torch.sin(ang).contiguous(),
        indices=indices,
    )


def build_kernels(kind, rows, page, capacity, launches, weights, timeline=True):
    """One kernel object per weight copy sharing one runtime and one compiled
    launcher, plus the ``timeline=True`` launcher of the same build (None
    without ``timeline``)."""
    nope, kvs, with_indexer = KINDS[kind]
    v = weights[0][0]
    kw = dict(
        rank=0,
        npes=1,
        group=None,
        topk=TOPK,
        launches_per_step=launches,
        with_indexer=with_indexer,
        index_max_seq=capacity,
        index_request_width=rows,
        index_page_size=page,
        index_block_table_stride=capacity // page,
        index_cache_dtype="fp8",
        attention_weight="fp8_block128",
        kv_cache_layout="atom",
        kv_cache_dtype="fp8",
        native_fp4_mfma=True,
        rope_dtype="fp32",
        shared_indices=kind == "rope_shared",
        separate_shared=True,
    )
    captured = {}
    build = v.op.build_glm5_monokernel

    def spy(*a, **k):
        captured.setdefault("call", (a, k))
        return build(*a, **k)

    v.op.build_glm5_monokernel = spy
    try:
        k0 = v.op.Glm5MonoKernel(weights[0][1], rows, **kw)
        ks = [k0] + [
            v.op.Glm5MonoKernel(W, rows, runtime=k0, **kw) for _, W in weights[1:]
        ]
    finally:
        v.op.build_glm5_monokernel = build
    for k in ks[1:]:
        k.launch = k0.launch
    if not timeline:
        return v, ks, None
    spec = getattr(k0.launch, "spec", None)
    if spec is not None:  # the rocKE layer kernel (AITER_ML4_ROCKE)
        rl = sys.modules[type(k0.launch).__module__]
        tl = rl.RockeLauncher(dict(spec, timeline=True), rl.rocke_root())
        # the same dual-copy area as the layer build's (mb_numa)
        tl.prepare_scratch(k0.scratch)
        return v, ks, tl
    a, k = captured["call"]
    tl_launch = build(*a, **dict(k, timeline=True))
    return v, ks, tl_launch


def launch(kernel, state, x_in, x_out, layer):
    with_indexer = kernel.with_indexer
    kernel.forward(
        x_in,
        state["cur_pos"],
        state["kv"],
        state["kv"],
        None if with_indexer else state["indices"],
        state["cos"],
        state["sin"],
        x_out=x_out,
        layer=layer,
        advance=False,
        index_cache=state["index_cache"] if with_indexer else None,
        block_tables=state["block_tables"],
        positions=state["positions"],
        slot_mapping=state["slots"],
        sparse_kv_indptr=state["indptr"],
    )


def _sha(t):
    import hashlib

    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def bf16_ulps(got, want):
    """(largest |got - want| in bf16 ulps at the larger magnitude, number of
    differing values); NaN on a NaN in either."""
    g, w = got.float(), want.float()
    mag = torch.maximum(g.abs(), w.abs()).clamp_min(2.0**-126)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
    return float(((g - w).abs() / ulp).max()), int((g != w).sum())


def _rocke_reassociates(k):
    """The rocKE build under test sums the MoE in another order (``moe_em``):
    its x_out is held to 1 bf16 ulp of FlyDSL's, not to the digest."""
    return bool(getattr(k.launch, "moe_em", False))


def check_launch(ks, state, ins, outs):
    """Digests of everything one launch on fresh state writes: the layer
    output, both caches and (indexer kinds) the published top-k."""
    k = ks[0]
    k.advance_step()
    launch(k, state, ins[0], outs[0], 0)
    torch.cuda.synchronize()
    got = {
        "x_out": _sha(outs[0]),
        "kv_cache": _sha(state["kv"]),
        "index_cache": _sha(state["index_cache"]),
    }
    if k.with_indexer:
        off = k.scr_layout["indices"]
        n = k.S * TOPK * 4
        got["indices"] = _sha(k.scratch[off : off + n])
    got["x_out_head"] = outs[0].float().flatten()[:8].tolist()
    return got


def graph_us(ks, state, ins, outs, launches, reps):
    def run_all():
        for i in range(launches):
            launch(ks[i % len(ks)], state, ins[i % len(ins)], outs[i % 2], i)
        ks[0].advance_step()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run_all()
        run_all()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=side):
        run_all()
    torch.cuda.synchronize()
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        gr.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / launches)
    del gr
    return {
        "median": statistics.median(ts),
        "min": min(ts),
        "max": max(ts),
        "reps": reps,
    }


def expert_union(ks, state, ins, outs, n):
    """Distinct routed experts per launch (the router's ``sel`` mailbox)."""
    from aiter.ops.flydsl.kernels.mono.mistral_large4.gfx950.geometry_config import (
        MOE_SLOTS,
        SHARED_EXPERT,
    )

    def picks(k):
        # a rocKE mb_numa build publishes sel in the dual-copy area only
        hdr = getattr(k.launch, "numa_hdr", 0)
        only = (getattr(k.launch, "spec", None) or {}).get("numa_regions") or ""
        if hdr and (not only or "sel" in only.split("+")):
            rl = importlib.import_module(type(k).__module__.rsplit(".", 1)[0] + ".rocke_layer")
            raw = rl.numa_logical(k.scratch, hdr, k.scr_layout["sel"], k.S * MOE_SLOTS * 8, 0)
            return raw.view(torch.int32)[0::2]
        return k.debug("sel", (k.S, MOE_SLOTS), torch.int32)

    sizes = []
    for i in range(n):
        k = ks[i % len(ks)]
        k.advance_step()
        launch(k, state, ins[i % len(ins)], outs[0], 0)
        torch.cuda.synchronize()
        sel = picks(k)
        routed = {int(e) for e in sel.flatten().tolist() if int(e) != SHARED_EXPERT}
        sizes.append(len(routed))
    return sizes


def tl_rows(kernel):
    """Timeline rows: the stage tasks, then (S > 1) one per (sample, up/gate
    task) and one per CTA of MoE checkpoints."""
    stages = kernel.stages
    base = sum(c for _, c in stages)
    ug = dict(stages)["ug"]
    return base, base + kernel.S * ug + 256


def timeline_runs(ks, tl_launch, state, ins, outs, n):
    stages = ks[0].stages
    _, rows_total = tl_rows(ks[0])
    tl = torch.zeros(rows_total, TL_COLS, dtype=torch.int64, device="cuda")
    saved = []
    for k in ks:
        saved.append((k.launch, int(k.index_params[7])))
        k.index_params[7] = tl.data_ptr()
        k.launch = tl_launch
    runs = []
    try:
        for i in range(n):
            k = ks[i % len(ks)]
            tl.zero_()
            k.advance_step()
            launch(k, state, ins[i % len(ins)], outs[0], 0)
            torch.cuda.synchronize()
            runs.append(tl.cpu().clone())
    finally:
        for k, (fn, slot7) in zip(ks, saved):
            k.launch = fn
            k.index_params[7] = slot7
    return stages, runs


def launch_gaps(ks, tl_launch, state, ins, outs, launches):
    """Clock time from one launch's last stamp to the next launch's first in a
    graph replay: kernel tail + launch gap + next prologue. Each weight copy
    stamps its own buffer, so the last ``len(ks)`` launches survive."""
    base_rows, rows_total = tl_rows(ks[0])
    tls = [torch.zeros(rows_total, TL_COLS, dtype=torch.int64, device="cuda") for _ in ks]
    saved = []
    for k, tl in zip(ks, tls):
        saved.append((k.launch, int(k.index_params[7])))
        k.index_params[7] = tl.data_ptr()
        k.launch = tl_launch

    def run_all():
        for i in range(launches):
            launch(ks[i % len(ks)], state, ins[i % len(ins)], outs[i % 2], i)
        ks[0].advance_step()

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_all()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=side):
            run_all()
        torch.cuda.synchronize()
        gr.replay()
        torch.cuda.synchronize()
        for tl in tls:
            tl.zero_()
        gr.replay()
        torch.cuda.synchronize()
        spans = []
        for tl in tls:
            r = tl.cpu()[:base_rows]
            st, en = r[:, 0][r[:, 0] > 0], r[:, 4][r[:, 4] > 0]
            spans.append((int(st.min()), int(en.max())))
        del gr
    finally:
        for k, (fn, slot7) in zip(ks, saved):
            k.launch = fn
            k.index_params[7] = slot7
    spans.sort()
    return {
        "gaps_us": [(spans[j + 1][0] - spans[j][1]) * TICK_US for j in range(len(spans) - 1)],
        "spans_us": [(e - s) * TICK_US for s, e in spans],
    }


def moe_stats(kernel, runs):
    """The S>1 MoE from the extension rows: per-CTA checkpoints and every
    up/gate (round, sample) iteration, medians over runs, us since the
    launch's first task start."""
    S = kernel.S
    base, _ = tl_rows(kernel)
    ug = dict(kernel.stages)["ug"]
    if S == 1:
        return None
    per_run = []
    for run in runs:
        starts = run[:base, 0][run[:base, 0] > 0]
        if starts.numel() == 0:
            continue
        t0 = int(starts.min())
        cta = run[base + S * ug : base + S * ug + 256]
        rec = {}
        for c in range(5):
            col = cta[:, c]
            col = col[col > 0]
            if col.numel():
                rec[f"cta{c}_med"] = (float(col.double().median()) - t0) * TICK_US
                rec[f"cta{c}_max"] = (int(col.max()) - t0) * TICK_US
        for r in range(ug // 256):
            for s in range(S):
                rows = run[base + s * ug + r * 256 : base + s * ug + (r + 1) * 256]
                ok = (rows[:, 0] > 0) & (rows[:, 4] > 0)
                if not bool(ok.any()):
                    continue
                x = rows[ok]
                key = f"r{r}s{s}"
                rec[key + "_start"] = (float(x[:, 0].double().median()) - t0) * TICK_US
                rec[key + "_end"] = (float(x[:, 4].double().median()) - t0) * TICK_US
                for a, b, nm in ((0, 1, "issue"), (1, 2, "mfma"), (2, 3, "reduce"), (3, 4, "emit")):
                    rec[f"{key}_{nm}"] = float((x[:, b] - x[:, a]).double().median()) * TICK_US
        per_run.append(rec)
    if not per_run:
        return None
    keys = sorted({k for r in per_run for k in r})
    return {k: statistics.median([r[k] for r in per_run if k in r]) for k in keys}


def fmt_moe(m):
    if not m:
        return ""
    lines = [
        "MoE per CTA (median / max us): "
        + ", ".join(
            f"{nm} {m.get(f'cta{c}_med', float('nan')):.1f}/{m.get(f'cta{c}_max', float('nan')):.1f}"
            for c, nm in enumerate(("route", "routed", "xq staged", "shared done", "ug done"))
        ),
        f"{'iter':6s} {'start':>7s} {'end':>7s} {'issue':>6s} {'mfma':>6s} {'reduce':>6s} {'emit':>6s}",
    ]
    r = 0
    while f"r{r}s0_start" in m:
        s = 0
        while f"r{r}s{s}_start" in m:
            k = f"r{r}s{s}"
            lines.append(
                f"{k:6s} {m[k + '_start']:7.2f} {m[k + '_end']:7.2f} {m[k + '_issue']:6.2f} "
                f"{m[k + '_mfma']:6.2f} {m[k + '_reduce']:6.2f} {m[k + '_emit']:6.2f}"
            )
            s += 1
        r += 1
    return "\n".join(lines)


def stage_stats(stages, run):
    """Per stage, in microseconds since the launch's first task start."""
    starts = run[:, 0][run[:, 0] > 0]
    if starts.numel() == 0:
        return None, {}
    t0 = int(starts.min())
    out, off = {}, 0
    for name, count in stages:
        r = run[off : off + count]
        off += count
        st, inp, done, end = r[:, 0], r[:, 2], r[:, 3], r[:, 4]
        ran = (st > 0) & (end > 0)
        if not bool(ran.any()):
            out[name] = {"tasks": count, "ran": 0}
            continue
        st_r, end_r = st[ran], end[ran]
        has_in = ran & (inp > 0)
        entry = {
            "tasks": count,
            "ran": int(ran.sum()),
            "start_min": (int(st_r.min()) - t0) * TICK_US,
            "start_max": (int(st_r.max()) - t0) * TICK_US,
            "end_min": (int(end_r.min()) - t0) * TICK_US,
            "end_max": (int(end_r.max()) - t0) * TICK_US,
            "task_med": float((end_r - st_r).double().median()) * TICK_US,
            "task_max": float((end_r - st_r).max()) * TICK_US,
        }
        if bool(has_in.any()):
            entry["wait_med"] = float((inp[has_in] - st[has_in]).double().median()) * TICK_US
            entry["wait_max"] = float((inp[has_in] - st[has_in]).max()) * TICK_US
            entry["busy_med"] = float((end[has_in] - inp[has_in]).double().median()) * TICK_US
            entry["busy_max"] = float((end[has_in] - inp[has_in]).max()) * TICK_US
            has_done = has_in & (done > 0)
            if bool(has_done.any()):
                entry["comp_med"] = float((done[has_done] - inp[has_done]).double().median()) * TICK_US
        out[name] = entry
    ends = [e["end_max"] for e in out.values() if e.get("ran")]
    return (max(ends) if ends else None), out


def median_stats(stages, runs):
    spans, per = [], []
    for run in runs:
        span, st = stage_stats(stages, run)
        if span is not None:
            spans.append(span)
            per.append(st)
    agg = {}
    for name, _ in stages:
        vals = [p[name] for p in per if p.get(name, {}).get("ran")]
        if not vals:
            agg[name] = {"ran": 0}
            continue
        keys = {k for v in vals for k in v}
        agg[name] = {
            k: statistics.median([v[k] for v in vals if k in v]) for k in sorted(keys)
        }
    return (statistics.median(spans) if spans else None), agg


def layer_bytes(kernel, kind, rows, context, union):
    """Bytes one launch has to read from HBM (weights, caches, experts)."""
    t = dict(kernel.W.t, **kernel.packed)
    attn = (
        "w_qkv_a", "s_qkv_a", "w_q_b", "s_q_b", "w_uk", "s_uk", "w_uv", "s_uv",
        "w_o", "s_o", "w_r",
    )  # fmt: skip
    index = ("w_index_q", "s_index_q", "w_index_k", "s_index_k", "w_index_w")
    nb = lambda x: x.numel() * x.element_size()  # noqa: E731
    attn_b = sum(nb(t[n]) for n in attn if n in t)
    index_b = sum(nb(t[n]) for n in index if n in t) if kernel.with_indexer else 0
    # the routed experts are read in place from the unpacked storage
    raw = kernel.W.t
    expert_b = sum(nb(raw[n]) // E for n in ("w_ug", "w_dn", "s_ug", "s_dn"))
    keys = min(context + 1, TOPK)
    kv_b = rows * keys * QK  # fp8 latent rows each row attends to
    scan_b = (context + rows) * 132 if kernel.with_indexer else 0
    return {
        "attention": attn_b,
        "indexer_weights": index_b,
        "expert_each": expert_b,
        "experts": (union + 1) * expert_b,
        "kv": kv_b,
        "index_scan": scan_b,
        "total": attn_b + index_b + (union + 1) * expert_b + kv_b + scan_b,
    }


def fmt_table(agg, span, us):
    lines = [
        f"{'stage':18s} {'tasks':>6s} {'ran':>5s} {'start':>7s} {'end':>7s} "
        f"{'dend':>7s} {'task':>7s} {'wait':>7s} {'busy':>7s} {'comp':>7s}"
    ]
    prev = 0.0
    for name, e in agg.items():
        if not e.get("ran"):
            continue
        d = e["end_max"] - prev
        prev = max(prev, e["end_max"])
        lines.append(
            f"{name:18s} {int(e['tasks']):6d} {int(e['ran']):5d} "
            f"{e['start_min']:7.2f} {e['end_max']:7.2f} {d:7.2f} "
            f"{e['task_med']:7.2f} {e.get('wait_med', float('nan')):7.2f} "
            f"{e.get('busy_med', float('nan')):7.2f} {e.get('comp_med', float('nan')):7.2f}"
        )
    lines.append(
        f"stamped span {span:.2f} us; graph {us:.2f} us/launch "
        f"(outside stamps {us - span:.2f} us)"
    )
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kinds", default="rope,rope_shared,nope,nope_kvs")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--page", type=int, default=16)
    p.add_argument("--capacity", type=int, default=32768)
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--copies", type=int, default=4)
    # distinct layer inputs cycled through the launches; 1 routes every launch
    # to the same experts (with --copies 1, a warm-cache bound)
    p.add_argument("--inputs", type=int, default=8)
    # every row of an input a copy of its first row: the rows route alike, so a
    # launch reads ~5 experts instead of ~16 (a working set that fits the MALL)
    p.add_argument("--same-rows", action="store_true")
    p.add_argument("--launches", type=int, default=71)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--timeline-runs", type=int, default=8)
    p.add_argument("--no-timeline", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--dump", default="", help="write per-kind output digests here")
    p.add_argument("--compare", default="", help="digests to match bit for bit")
    args = p.parse_args(argv)
    golden = {}
    if args.compare:
        with open(args.compare) as f:
            golden = json.load(f)
    torch.set_grad_enabled(False)
    dev = torch.cuda.get_device_properties(0)
    print(f"[bench] {dev.name} {getattr(dev, 'gcnArchName', '')} CUs={dev.multi_processor_count}")
    state = make_state(args.rows, args.page, args.capacity, args.context)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(7)
    ins = [
        torch.randn(1 if args.same_rows else args.rows, HIDDEN, generator=gen, device="cuda")
        .bfloat16()
        .expand(args.rows, HIDDEN)
        .contiguous()
        for _ in range(args.inputs)
    ]
    outs = [torch.empty(args.rows, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    report = {"args": vars(args), "kinds": {}}
    x_vals = {}
    for kind in args.kinds.split(","):
        t0 = time.time()
        weights = [build_weights(kind, c) for c in range(args.copies)]
        ks_v, ks, tl_launch = build_kernels(
            kind, args.rows, args.page, args.capacity, args.launches, weights
        )
        res = getattr(ks[0].launch, "resources", None)
        if res is not None:
            print(f"[bench] {kind} rocKE layer kernel {res}", flush=True)
        # the bitwise check runs first, on fresh caches and scratch
        fresh = make_state(args.rows, args.page, args.capacity, args.context)
        digest = check_launch(ks, fresh, ins, outs)
        x_vals[kind] = outs[0].float().flatten().tolist()
        del fresh
        verdict = None
        if kind in golden:
            want = golden[kind]
            diff = [n for n in want if not n.endswith(("_head", "_vals")) and want[n] != digest.get(n)]
            verdict = "bitwise" if not diff else "DIFFERS: " + ",".join(diff)
            if diff == ["x_out"] and _rocke_reassociates(ks[0]) and "x_out_vals" in want:
                worst, n_diff = bf16_ulps(outs[0].float().flatten(), torch.tensor(want["x_out_vals"], device="cuda"))
                if worst <= 1.0:
                    verdict = f"tolerance (x_out: {n_diff} values differ, at most {worst:.2f} bf16 ulp)"
                else:
                    verdict += f" (x_out: {n_diff} values differ, at most {worst:.2f} bf16 ulp)"
            print(f"[bench] {kind} check vs {args.compare}: {verdict}", flush=True)
            if diff:
                print(f"[bench]   x_out head want {want.get('x_out_head')}", flush=True)
                print(f"[bench]   x_out head got  {digest.get('x_out_head')}", flush=True)
        t1 = time.time()
        g = graph_us(ks, state, ins, outs, args.launches, args.reps)
        t2 = time.time()
        union = expert_union(ks, state, ins, outs, 8)
        mean_union = statistics.mean(union)
        nb = layer_bytes(ks[0], kind, args.rows, args.context, mean_union)
        entry = {
            "digest": digest,
            "check": verdict,
            "graph_us": g,
            "expert_union": union,
            "bytes": nb,
            "gbps": nb["total"] / (g["median"] * 1e3),
            "build_s": t1 - t0,
            "first_graph_s": t2 - t1,
        }
        print(
            f"[bench] {kind} r{args.rows}: {g['median']:.2f} us/launch "
            f"[{g['min']:.2f}, {g['max']:.2f}], experts/launch {mean_union:.1f}, "
            f"{nb['total'] / 2**20:.1f} MiB -> {entry['gbps']:.0f} GB/s "
            f"(build {t1 - t0:.0f}s, compile+capture {t2 - t1:.0f}s)",
            flush=True,
        )
        if not args.no_timeline:
            stages, runs = timeline_runs(
                ks, tl_launch, state, ins, outs, args.timeline_runs
            )
            base_rows, _ = tl_rows(ks[0])
            span, agg = median_stats(stages, [r[:base_rows] for r in runs])
            moe = moe_stats(ks[0], runs)
            entry["timeline"] = {"span_us": span, "stages": agg, "moe": moe}
            if span is not None:
                print(fmt_table(agg, span, g["median"]), flush=True)
            if moe:
                print(fmt_moe(moe), flush=True)
            gaps = launch_gaps(ks, tl_launch, state, ins, outs, args.launches)
            entry["launch_gaps"] = gaps
            print(
                "[bench] in-graph launch spans "
                + ", ".join(f"{x:.1f}" for x in gaps["spans_us"])
                + " us; gaps between them "
                + ", ".join(f"{x:.1f}" for x in gaps["gaps_us"])
                + " us",
                flush=True,
            )
        report["kinds"][kind] = entry
        for k in ks:
            k.close()
        del ks, weights, tl_launch
        torch.cuda.empty_cache()
    mix = {k: n for k, n in MODEL_MIX.items() if k in report["kinds"]}
    if len(mix) == len(MODEL_MIX):
        step = sum(n * report["kinds"][k]["graph_us"]["median"] for k, n in mix.items())
        report["step_layers_us"] = step
        print(f"[bench] 71-layer mix at npes=1: {step / 1e3:.3f} ms per verify step")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"[bench] wrote {args.out}")
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({k: dict(e["digest"], x_out_vals=x_vals[k]) for k, e in report["kinds"].items()}, f)
        print(f"[bench] wrote digests {args.dump}")
    bad = [k for k, e in report["kinds"].items()
           if e["check"] not in (None, "bitwise") and not e["check"].startswith("tolerance")]  # fmt: skip
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
