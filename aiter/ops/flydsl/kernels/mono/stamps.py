# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Optional per-CTA stage stamps for ML4 mono timing.

The existing ML4 launcher also has a global ``timeline`` buffer.  This module
provides the LDS/scratch layout used by the schedule tooling without changing
that ABI.  It is intentionally off by default; callers that need the
per-CTA view can allocate :data:`REGION_BYTES` after the normal mailbox
scratch and use the helpers while tracing a kernel.
"""

from __future__ import annotations

import os

ENV = "AITER_MONO_STAMPS"
GRID = 256
WAVES = 8
LAYER_SLOTS = 128
MAX_STAGES = 11
PER_STAGE = 2 + WAVES
POINTS = 1 + MAX_STAGES * PER_STAGE
LAUNCH_BYTES = GRID * POINTS * 8
REGION_BYTES = LAYER_SLOTS * LAUNCH_BYTES


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def scratch_extra() -> int:
    return REGION_BYTES if enabled() else 0


def enter_ix(stage: int) -> int:
    return 1 + stage * PER_STAGE


def exit_ix(stage: int) -> int:
    return 2 + stage * PER_STAGE


def ready_ix(stage: int, wave: int = 0) -> int:
    return 3 + stage * PER_STAGE + wave


def offset(base_scratch_bytes: int) -> int:
    return base_scratch_bytes


def memrealtime():
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import T

    return llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])


def reset():
    global CURRENT
    CURRENT = None


CURRENT = None
