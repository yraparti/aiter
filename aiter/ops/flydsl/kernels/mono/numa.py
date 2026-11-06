# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""NUMA dual-copy mailbox views for the ML4 schedule lever.

Only rank-local tagged regions use :class:`Box`.  Peer reductions, the fault
word, and the plain ``indices`` hand-off remain ordinary addresses.  The
layout is intentionally independent of a particular launch width: the
reserved NUMA area is sized for the largest ML4 local mailbox layout and the
logical offsets of a narrower build are used inside it.
"""

from __future__ import annotations

import os

CHUNK = 8192
PAGE = 1 << 21
MAGIC = 0x4E554D41
CM_DEV = 16


def _mailbox_bytes() -> int:
    from .mistral_large4.gfx950.layout import layout

    # ML4's largest local mailbox build is the 12-row TP4 geometry.  Indexer
    # and peer regions are deliberately excluded: they are system-scope or
    # plain-store hand-offs, not NUMA mailboxes.
    scratch, _ = layout(12, 16, 4, 2048, with_indexer=False)
    return scratch["_bytes"]


def header_offset() -> int:
    return _mailbox_bytes()


def area_bytes() -> int:
    return 2 * _mailbox_bytes()


def scratch_bytes_needed() -> int:
    """Header + slack to a 2 MiB boundary + two mailbox copies."""
    return header_offset() + PAGE + area_bytes()


def area_base(scratch_ptr: int) -> int:
    start = scratch_ptr + header_offset()
    return (start + PAGE - 1) & ~(PAGE - 1)


def phys(x: int) -> int:
    """Copy-0 offset for logical mailbox byte ``x``."""
    return x + (x & ~(CHUNK - 1))


class Box:
    """A bitwise-addressable NUMA view of one logical mailbox region."""

    def __init__(self, scratch, off: int, state):
        self.scratch = scratch
        self.off = int(off)
        self.state = state

    def __add__(self, offset: int):
        if not isinstance(offset, int):
            raise TypeError("numa.Box only accepts static integer offsets")
        return Box(self.scratch, self.off + offset, self.state)

    def __sub__(self, offset: int):
        return self + (-int(offset))

    def base(self):
        import flydsl.expr as fx

        address = fx.Int64(self.scratch) + fx.Int64(header_offset() + PAGE - 1)
        return address & fx.Int64(~(PAGE - 1))

    def logical(self):
        import flydsl.expr as fx

        return fx.Int64(self.scratch) + fx.Int64(self.off)


def _x(box: Box, byte_offset):
    import flydsl.expr as fx

    value = fx.Int32(byte_offset) + box.off
    return value + (value & -CHUNK)


def store(box: Box, byte_offset, data, cm=CM_DEV):
    """Store a payload to both IOD copies."""
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from aiter.ops.flydsl.kernels.mono.common.device.ops import rsrc

    offset = _x(box, byte_offset)
    resource = rsrc(box.base())
    bo.buffer_store(data, resource, offset >> 2, cache_modifier=cm)
    bo.buffer_store(data, resource, (offset + CHUNK) >> 2, cache_modifier=cm)


def load(box: Box, byte_offset, words: int, cm=CM_DEV):
    """Load the copy selected for this CTA's IOD."""
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from aiter.ops.flydsl.kernels.mono.common.device.ops import rsrc
    from flydsl.expr.typing import T

    offset = _x(box, byte_offset)
    if box.state.mode == 2:
        selected = offset
    else:
        page_mask = (box.state.sel_mask >> (offset >> 21)) & 1
        selected = offset + (page_mask << 13)
    return bo.buffer_load(
        rsrc(box.base()),
        selected >> 2,
        vec_width=words,
        dtype=T.i32,
        cache_modifier=cm,
    )


def _xcc_id():
    import flydsl.expr as fx
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import T

    value = llvm.InlineAsmOp(
        T.i32,
        [],
        "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
        "=s",
        has_side_effects=False,
    )
    return fx.Int32(value.result)


def _iod_id():
    return (_xcc_id() >> 2) & 1


def entry_state(scratch, mode=1):
    """Load the calibrated page mask once per CTA."""
    import flydsl.expr as fx
    from aiter.ops.flydsl.kernels import buffer_ops as bo
    from aiter.ops.flydsl.kernels.mono.common.device.ops import rsrc
    from flydsl.expr.typing import T

    from types import SimpleNamespace

    if mode == 2:
        return SimpleNamespace(sel_mask=None, mode=2)
    header = rsrc(fx.Int64(scratch) + fx.Int64(header_offset()))
    mask = fx.Int32(bo.buffer_load(header, 0, vec_width=1, dtype=T.i32, is_scalar=True))
    flip = _iod_id() if mode == 1 else _iod_id() ^ 1
    return SimpleNamespace(sel_mask=mask ^ (fx.Int32(0) - flip), mode=mode)


_PREPARED = set()


def calibrate(scratch, iters: int = 48):
    """Write a valid default calibration header.

    The host hook is intentionally isolated here so a runner can replace the
    conservative mask-zero fallback with its XCD ping-pong probe before graph
    capture.  A zero mask is correct for the protocol (it merely selects the
    remote copy on one IOD); it is never an uninitialized header.
    """
    import torch

    if scratch.numel() * scratch.element_size() < scratch_bytes_needed():
        raise ValueError("scratch is too small for the ML4 NUMA mailbox area")
    words = scratch.view(torch.int32)
    header = header_offset() // 4
    words[header] = 0
    words[header + 1] = MAGIC
    torch.cuda.synchronize()
    return 0, 0.0, {"remote_over_local_rtt": None, "calibrated": False, "iters": iters}


def wanted(key: str = "numa") -> bool:
    from . import sched

    spec = os.environ.get(sched.ENV, "")
    if not spec:
        return False
    if "best" in spec:
        return True
    for group in spec.split(";"):
        _selector, _, body = group.rpartition(":")
        body = body if _ else group
        for name, value in sched.parse_levers(body):
            if value and (name == key or name.startswith(key + ".")):
                return True
    return False


def prepare(scratch):
    """Calibrate one scratch allocation once, before graph capture."""
    if not wanted():
        return None
    pointer = int(scratch.data_ptr())
    if pointer in _PREPARED:
        return None
    result = calibrate(scratch)
    _PREPARED.add(pointer)
    import sys

    print(
        "[AITER_MONO_SCHED numa] scratch calibrated: "
        f"page mask {result[0]:#x}, agreement {result[1]:.4f}",
        file=sys.stderr,
        flush=True,
    )
    return result
