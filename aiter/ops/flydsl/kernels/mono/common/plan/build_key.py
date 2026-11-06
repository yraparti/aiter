# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A kernel build's parameters, all of them in FlyDSL's JIT cache key.

FlyDSL keys a compiled kernel by the scalar values (int, float, bool, str, enum,
tuple) its launcher's closures capture; an object in a closure is not keyed, and
a parameter used only by build-time Python (a placement table, a layout) is in no
closure at all. Two builds differing only there would share one binary -- every
TP rank ran rank 3's kernel once that way. So a build's parameters are one frozen
dataclass of scalars, ``key_tuple`` turns it into one (name, value) tuple, and the
launcher references that tuple: every field is keyed, whatever the body uses.

The tuple also carries the model's kernel sources (``source_digest``): FlyDSL
keys a helper's source only when its walk reaches it, and it does not reach a
helper called inside a traced function's dynamic loop or branch (the rewriter
moves those bodies into nested functions it does not scan) nor one outside the
kernel's directory -- an edit there kept the old binary.
"""

from __future__ import annotations

import enum
import functools
import hashlib
from dataclasses import fields
from pathlib import Path

_SCALARS = (int, float, bool, str, enum.Enum)

# ``kernels/mono`` package root (common/plan -> ../..)
_MONO_KERNELS = Path(__file__).resolve().parents[2]
# ``ops/flydsl`` (host mono + aiter flydsl helpers the kernel imports)
_FLYDSL = Path(__file__).resolve().parents[4]


@functools.cache
def source_digest(*paths: Path | str) -> str:
    """sha256 of every ``.py`` file under ``paths`` (absolute files or dirs)."""
    h = hashlib.sha256()
    for raw in paths:
        root = Path(raw)
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        if not files:
            raise FileNotFoundError(f"no kernel sources at {root}")
        for f in files:
            try:
                rel = str(f.relative_to(_FLYDSL))
            except ValueError:
                rel = str(f)
            h.update(rel.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


@functools.cache
def mono_source_digest() -> str:
    """Digest of the ML4 mono sources and the aiter FlyDSL helpers they import."""
    return source_digest(
        _MONO_KERNELS,
        _FLYDSL / "mono",
        _FLYDSL / "kernels" / "buffer_ops.py",
        _FLYDSL / "kernels" / "dpp_utils.py",
        _FLYDSL / "kernels" / "kernels_common.py",
    )


def key_tuple(key, sources: str) -> tuple:
    """``key``'s (field name, value) pairs, every value a scalar, then
    ("sources", ``sources``): the model's ``source_digest``."""
    out = []
    for f in fields(key):
        value = getattr(key, f.name)
        if not isinstance(value, _SCALARS):
            raise TypeError(
                f"{type(key).__name__}.{f.name} = {value!r}: a build parameter must be"
                " a scalar to reach the JIT cache key"
            )
        out.append((f.name, value))
    out.append(("sources", sources))
    return tuple(out)


def symbol_params(key) -> dict:
    """The fields that name the kernel symbol (``metadata={"sym": short name}``),
    short name -> value, in field order."""
    return {
        f.metadata["sym"]: getattr(key, f.name)
        for f in fields(key)
        if "sym" in f.metadata
    }
