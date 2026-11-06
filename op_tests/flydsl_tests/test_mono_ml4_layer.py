# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU tests of the aiter-hosted ML4 mono layer on an fp8 latent KV cache.

Ported from vLLM ``tests/.../test_fp8_kv_cache.py``. One TP4 rank's shard of
one MoE layer per kind runs on a single GPU through ``Glm5MonoKernel``
(``npes=1``), with random weights, a randomly paged cache and one DSpark
verify step of 4 rows whose last row is padding.

Uses aiter's MXFP4 shuffle helpers and a torch reference for the fp8 MLA
cache write instead of vLLM's ``concat_and_cache_mla`` /
``rocm_aiter_ops.shuffle_weights``.
"""

from __future__ import annotations

import functools

import pytest
import torch

pytest.importorskip("aiter")

FP8 = torch.float8_e4m3fn
HIDDEN, NH, KVL, PE, VD, E = 6144, 16, 512, 64, 128, 128
QK = KVL + PE
PAGE, PAGES, S, POS0, TOPK = 64, 64, 4, 3000, 2048


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return False
    # MI355X / gfx950 report as gfx950 in HIP builds of PyTorch.
    props = torch.cuda.get_device_properties(0)
    gcn = getattr(props, "gcnArchName", "") or getattr(props, "name", "")
    return "gfx950" in str(gcn).lower() or "mi355" in name or "mi350" in name


pytestmark = pytest.mark.skipif(
    not _on_gfx950(), reason="the ML4 mono kernels need gfx950 (MI355X)"
)


def _randn(*shape, scale=1.0):
    return torch.randn(*shape, device="cuda") * scale


def _q128(w: torch.Tensor, block_cols: int = 128):
    """bf16/fp32 [r, c] -> (fp8, fp32 scale per 128 x block_cols), amax / 448.

    Pads like vLLM ``runtime.weights._q128`` so shapes such as 576 x 6144 work.
    """
    import torch.nn.functional as F

    r, c = w.shape
    w = w.float()
    wp = F.pad(w, (0, (-c) % block_cols, 0, (-r) % 128))
    b = wp.view(wp.shape[0] // 128, 128, wp.shape[1] // block_cols, block_cols)
    s = b.abs().amax((1, 3)).clamp_min(1e-12) / 448.0
    q = (b / s[:, None, :, None]).clamp(-448, 448).view_as(wp)[:r, :c].to(FP8)
    return q.contiguous(), s.contiguous()


def _shuffle_mxfp4(values: torch.Tensor) -> torch.Tensor:
    """Best-effort aiter shuffle; fall back to identity if unavailable."""
    try:
        from aiter.ops.shuffle import shuffle_weight

        return shuffle_weight(values, (16, 16))
    except Exception:
        try:
            from aiter import shuffle_weight

            return shuffle_weight(values, (16, 16))
        except Exception:
            out = values.contiguous()
            out.is_shuffled = True  # type: ignore[attr-defined]
            return out


@functools.cache
def build_layer(kind: str):
    """(kernel variant, LayerWeights) of one random layer."""
    from aiter.ops.flydsl.mono.variants import load_variant
    from aiter.ops.flydsl.mono.weights.config import Mxfp4ScaleLayout, Mxfp4WeightLayout
    from aiter.ops.flydsl.mono.weights.layer import LayerWeights
    from aiter.utility.fp4_utils import e8m0_shuffle

    # Layer-kind flags mirror vLLM runtime.weights.KINDS
    kinds = {
        "rope": (False, False, True),
        "nope": (True, False, True),
        "nope_kvs": (True, True, False),
    }
    nope_layer, kv_share, _ = kinds[kind]
    torch.manual_seed(0)
    v = load_variant(nope_layer, kv_share)
    C = v.geometry_config.glm5_tp_config(4)
    bf, f32 = torch.bfloat16, torch.float32
    t = {}
    t["w_q_b"], t["s_q_b"] = _q128(_randn(NH * 128, HIDDEN, scale=HIDDEN**-0.5))
    wkva = _randn(KVL + PE, HIDDEN, scale=HIDDEN**-0.5)
    if nope_layer:
        wkva[KVL:] = 0
    t["w_qkv_a"], t["s_qkv_a"] = _q128(wkva * (not kv_share))
    t["g_kv"] = (torch.rand(KVL, device="cuda") + 0.5).to(bf)
    wkvb = _randn(NH, C.nope_dim + VD, KVL, scale=KVL**-0.5)
    w_uk = wkvb[:, : C.nope_dim].transpose(1, 2).reshape(NH * KVL, C.nope_dim)
    t["w_uk"], t["s_uk"] = _q128(w_uk, 64)
    t["w_uv"], t["s_uv"] = _q128(wkvb[:, C.nope_dim :].reshape(NH * VD, KVL))
    t["w_o"], t["s_o"] = _q128(_randn(HIDDEN, NH * VD, scale=(NH * VD) ** -0.5))
    t["g_in"] = t["g_post"] = torch.ones(HIDDEN, dtype=bf, device="cuda")
    t["w_r"] = _randn(E, HIDDEN, scale=HIDDEN**-0.5).to(bf)
    t["bias"] = torch.zeros(E, dtype=f32, device="cuda")
    gc, ih, idim = v.geometry_config, 32, 128
    prm = torch.zeros(gc.ML4_P_WORDS, dtype=f32, device="cuda")
    prm[gc.ML4_P_SS_S : gc.ML4_P_SS_S + NH] = 1.0
    prm[gc.ML4_P_IQ_W : gc.ML4_P_IQ_W + HIDDEN] = 1.0
    prm[gc.ML4_P_ISS_S : gc.ML4_P_ISS_S + ih] = 1.0
    t["g_q"] = prm
    t["w_index_q"], t["s_index_q"] = _q128(
        _randn(ih * idim, HIDDEN, scale=HIDDEN**-0.5)
    )
    kw = _randn(idim + ih, HIDDEN, scale=HIDDEN**-0.5)
    t["w_index_k"], t["s_index_k"] = _q128(kw[:idim])
    t["w_index_w"] = kw[idim:].to(bf).contiguous()
    t["g_index_k"] = torch.ones(idim, dtype=f32, device="cuda")
    t["b_index_k"] = torch.zeros(idim, dtype=f32, device="cuda")
    u8 = torch.uint8
    for w, s, rows, cols in (
        ("w_ug", "s_ug", 2 * C.inter, HIDDEN),
        ("w_dn", "s_dn", HIDDEN, C.inter),
    ):
        values = torch.randint(0, 256, (E, rows, cols // 2), dtype=u8, device="cuda")
        scales = torch.randint(
            118, 122, (E * rows, cols // 32), dtype=u8, device="cuda"
        )
        t[s] = e8m0_shuffle(scales).view(u8)
        t[w] = values
    t["w_ug"] = _shuffle_mxfp4(t["w_ug"])
    t["w_dn"] = _shuffle_mxfp4(t["w_dn"])
    t["w_ug"].is_shuffled = t["w_dn"].is_shuffled = True  # type: ignore[attr-defined]

    # Shared expert tensors use the op.py SHARED_EXPERT_NAMES keys (w_*_sh).
    # Content is irrelevant for the fp8-cache gate; reuse routed expert 0.
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


def fp8_mla_rows(rows: torch.Tensor) -> torch.Tensor:
    """[n, 576] latent rows as an fp8 cache writer stores them (scale 1)."""
    rows = rows.to(torch.bfloat16)
    # Saturating cast to e4m3 with scale 1, matching vLLM's scale=1 path.
    return rows.to(FP8)


def make_step():
    torch.manual_seed(1)
    base = fp8_mla_rows(_randn(PAGES * PAGE, QK, scale=0.5))
    keys = _randn(PAGES, PAGE * 128, scale=0.5).to(FP8).view(torch.uint8)
    scales = torch.ones(PAGES, PAGE, device="cuda").view(torch.uint8)
    index_cache = torch.cat([keys, scales], 1).view(PAGES, PAGE, 132).contiguous()
    pages = torch.randperm(PAGES, device="cuda", dtype=torch.int32)
    positions = torch.arange(POS0, POS0 + S, device="cuda")
    slots = pages[positions // PAGE].long() * PAGE + positions % PAGE
    slots[-1] = -1
    nkeys = torch.where(slots >= 0, torch.clamp(positions + 1, max=TOPK), 0)
    indptr = torch.cat([nkeys.new_zeros(1), nkeys.cumsum(0)]).int()
    freqs = 10000 ** (-torch.arange(32, device="cuda") / 32)
    ang = torch.arange(PAGES * PAGE, device="cuda")[:, None] * freqs[None]
    step = dict(
        h=_randn(S, HIDDEN).to(torch.bfloat16),
        cur_pos=torch.tensor([POS0], dtype=torch.int32, device="cuda"),
        cos=torch.cos(ang).contiguous(),
        sin=torch.sin(ang).contiguous(),
        block_tables=pages[None].contiguous(),
        positions=positions,
        slots=slots,
        indptr=indptr,
    )
    return base, index_cache, step


def run_layer(kind, kv_cache_dtype, kv, index_cache, step, W=None):
    v, W0 = build_layer(kind)
    kernel = v.op.Glm5MonoKernel(
        W or W0,
        S,
        rank=0,
        npes=1,
        group=None,
        topk=TOPK,
        launches_per_step=1,
        with_indexer=True,
        index_max_seq=PAGES * PAGE,
        index_request_width=S,
        index_page_size=PAGE,
        index_block_table_stride=PAGES,
        index_cache_dtype="fp8",
        attention_weight="fp8_block128",
        kv_cache_layout="atom",
        kv_cache_dtype=kv_cache_dtype,
        native_fp4_mfma=True,
        rope_dtype="fp32",
        separate_shared=True,
    )
    try:
        out = kernel.forward(
            step["h"],
            step["cur_pos"],
            kv,
            kv,
            None,
            step["cos"],
            step["sin"],
            index_cache=index_cache.clone(),
            block_tables=step["block_tables"],
            positions=step["positions"],
            slot_mapping=step["slots"],
            sparse_kv_indptr=step["indptr"],
        )[: S - 1]
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        return out.clone()
    finally:
        kernel.close()


def run_both(kind, history_scale=1, W=None):
    base, index_cache, step = make_step()
    base = (base.float() * history_scale).to(FP8)
    kv_bf, kv_f8 = base.to(torch.bfloat16), base.clone()
    out_bf = run_layer(kind, "bf16", kv_bf, index_cache, step, W)
    out_f8 = run_layer(kind, "fp8", kv_f8, index_cache, step, W)
    return out_bf, out_f8, kv_bf, kv_f8, base, step["slots"][:-1]


@pytest.mark.parametrize("kind", ("rope", "nope", "nope_kvs"))
@pytest.mark.parametrize("history_scale", (1, 4))
@torch.inference_mode()
def test_fp8_cache_matches_bf16(kind, history_scale):
    out_bf, out_f8, kv_bf, kv_f8, base, slots = run_both(kind, history_scale)
    written = torch.zeros(PAGES * PAGE, dtype=torch.bool, device="cuda")
    written[slots] = kind != "nope_kvs"
    u8 = torch.uint8
    assert torch.equal(kv_f8[~written].view(u8), base[~written].view(u8))
    if kind != "nope_kvs":
        new = fp8_mla_rows(kv_bf[written])
        assert torch.equal(kv_f8[written].view(u8), new.view(u8))
    if kind == "nope_kvs" or history_scale > 1:
        assert torch.equal(out_f8, out_bf)
    else:
        rel = (out_f8.float() - out_bf.float()).norm() / out_bf.float().norm()
        assert rel < 0.1, rel


@pytest.mark.parametrize("kind", ("rope", "nope"))
@torch.inference_mode()
def test_fp8_cache_saturates(kind):
    from dataclasses import replace

    _, W = build_layer(kind)
    t = dict(W.t, g_kv=W.t["g_kv"] * 1000, s_qkv_a=W.t["s_qkv_a"] * 1000)
    _, _, kv_bf, kv_f8, _, slots = run_both(kind, W=replace(W, t=t))
    new_bf, new_f8 = kv_bf[slots], kv_f8[slots]
    assert (new_bf[:, :KVL].float().abs() > 448).any()
    assert (new_bf[:, KVL:].float().abs() > 448).any() == (kind == "rope")
    assert not torch.isnan(new_f8.float()).any()
    assert torch.equal(
        new_f8.view(torch.uint8), fp8_mla_rows(new_bf).view(torch.uint8)
    )
