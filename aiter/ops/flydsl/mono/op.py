# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Host wrapper for the ML4 decode monokernel (default RoPE variant).

Prefer ``aiter.ops.flydsl.mono.variants.load_variant`` when NoPE / KV-share
geometry differs; this module re-exports the default (RoPE, no KV-share) build.
"""

from aiter.ops.flydsl.mono.variants import load_variant as _load_variant

_v = _load_variant(False, False)
Glm5MonoKernel = _v.op.Glm5MonoKernel
prepare_glm5_weights = _v.op.prepare_glm5_weights
param_table = _v.op.param_table
fault_word_offset = _v.op.fault_word_offset
build_glm5_monokernel = _v.op.build_glm5_monokernel

__all__ = [
    "Glm5MonoKernel",
    "prepare_glm5_weights",
    "param_table",
    "fault_word_offset",
    "build_glm5_monokernel",
]
