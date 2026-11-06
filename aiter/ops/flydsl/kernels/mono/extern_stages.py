# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""rocKE stage linking for the ML4 mono kernel.

The resolver deliberately lives outside the FlyDSL builder.  A linked file is
part of the binary even when the Python builder's source has not changed, so
its digest is returned separately and must be included in the builder key.
With the environment unset this module is entirely inert.

``AITER_MONO_EXTERN_STAGES`` is a comma-separated list such as::

    router=/tmp/ml4/{kind}/router_{variant}.bc,ug=/tmp/ml4/{kind}

The directory form looks for ``<stage>_<variant>.bc`` and then ``.ll``.
``AITER_MONO_EXTERN_MISSING=mono`` keeps the FlyDSL body when a requested
variant is absent; the default policy is an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass

ENV = "AITER_MONO_EXTERN_STAGES"
MISSING_ENV = "AITER_MONO_EXTERN_MISSING"
SYMBOL_PREFIX = "rocke_ml4_"

# The ABI is the actual ML4 kernel parameter list, not the host wrapper's
# parameter table.  Keep this tuple local so CPU tests can validate sidecars
# without importing FlyDSL or compiling a kernel.
KERNEL_ARGS = (
    "h_in",
    "x_out",
    "cur_pos",
    "positions",
    "slot_mapping",
    "sparse_kv_indptr",
    "kv_cache",
    "pe_cache",
    "indices",
    "rope_cos",
    "rope_sin",
    "g_in",
    "g_q",
    "g_kv",
    "g_post",
    "w_qkv_a",
    "s_qkv_a",
    "w_q_b",
    "s_q_b",
    "w_uk",
    "s_uk",
    "w_uv",
    "s_uv",
    "w_o",
    "s_o",
    "w_r",
    "bias",
    "w_ug",
    "s_ug",
    "w_dn",
    "s_dn",
    "scratch",
    "sym",
    "peers",
    "timeline_buf",
    "step",
    "rank",
    "layer",
)

KNOWN_STAGES = ("router", "ug", "down")
LAYER_ARGS = KERNEL_ARGS


@dataclass(frozen=True)
class ExternStage:
    stage: str
    path: str
    symbol: str
    ffi_types: tuple[str, ...]
    digest: str
    mailboxes: str = "plain"


def variant(rows: int, seq: bool = False, group: int = 0) -> str:
    """Return the rocKE variant spelling used by ML4 generators."""
    # ``group`` is accepted for parity with the ATOM resolver.  ML4's current
    # launcher has no grouped layout, but accepting it keeps the resolver
    # useful to DSpark's verify launcher.
    return f"r{rows}" + ("s" if seq else "") + (f"g{group}" if group else "")


def parse(value: str | None) -> dict[str, str]:
    """Parse ``<stage>=<path>,...``; an empty value disables externs."""
    result: dict[str, str] = {}
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        stage, sep, path = item.partition("=")
        stage, path = stage.strip(), path.strip()
        if not sep or not stage or not path:
            raise ValueError(f"{ENV}: expected <stage>=<path>, got {item!r}")
        if stage in result:
            raise ValueError(f"{ENV}: stage {stage!r} listed twice")
        result[stage] = path
    return result


def requested() -> dict[str, str]:
    return parse(os.environ.get(ENV))


def expected_abi(layer_args=KERNEL_ARGS) -> list[list[str]]:
    """Return the sidecar ABI, including the three ML4 stage scalars.

    ``abi`` uses LLVM spellings (``i32``/``i64``), while ``ffi_types`` uses
    FlyDSL's public names (``int32``/``int64``).
    """
    args = [["tag", "i32"], ["task", "i32"], ["peer_slot", "i32"]]
    args += [
        [name, "i32" if name in ("rank", "layer") else "i64"]
        for name in layer_args
    ]
    args.append(["lds_base", "i32"])
    return args


def ffi_types(layer_args=KERNEL_ARGS) -> list[str]:
    return [
        "int32" if typ == "i32" else "int64"
        for _name, typ in expected_abi(layer_args)
    ]


def kind_of(stages, kind: str | None = None) -> str:
    """Return the ML4 layer family used by ``{kind}`` path templates."""
    if kind is not None:
        return kind
    # The stage sequence itself does not distinguish NoPE and RoPE.  ``ml4``
    # is intentionally stable for callers that do not provide the build's
    # geometry kind.
    return "ml4"


def _file(stage: str, path: str, var: str) -> str | None:
    if "{variant}" in path:
        candidate = path.format(variant=var)
    elif os.path.isdir(path):
        stem = os.path.join(path, f"{stage}_{var}")
        candidate = next(
            (stem + suffix for suffix in (".bc", ".ll") if os.path.isfile(stem + suffix)),
            stem + ".ll",
        )
    else:
        candidate = path
    if os.path.isfile(candidate):
        return candidate
    policy = os.environ.get(MISSING_ENV, "error")
    if policy == "mono":
        print(
            f"[{ENV}] no {stage} function for build variant {var} "
            f"({candidate}): using the FlyDSL body",
            file=sys.stderr,
        )
        return None
    if policy != "error":
        raise ValueError(f"{MISSING_ENV} must be error or mono, got {policy!r}")
    raise FileNotFoundError(f"{ENV}: no {stage} function for build variant {var} ({candidate})")


def _load(stage, path, var, layer_args):
    file_path = _file(stage, path, var)
    if file_path is None:
        return None
    sidecar = os.path.splitext(file_path)[0] + ".json"
    try:
        with open(sidecar, encoding="utf-8") as stream:
            meta = json.load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{ENV}: missing sidecar {sidecar}") from error

    symbol = f"{SYMBOL_PREFIX}{stage}_{var}"
    if meta.get("symbol") != symbol:
        raise ValueError(f"{sidecar}: symbol {meta.get('symbol')!r}, expected {symbol!r}")
    if meta.get("abi") != expected_abi(layer_args):
        raise ValueError(f"{sidecar}: ABI differs from the ML4 layer kernel")
    types = tuple(meta.get("ffi_types", ()))
    if len(types) != len(expected_abi(layer_args)):
        raise ValueError(f"{sidecar}: ffi_types length differs from the ML4 layer kernel")
    with open(file_path, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()[:16]
    _check_numa(sidecar, meta)
    return ExternStage(
        stage=stage,
        path=os.path.abspath(file_path),
        symbol=symbol,
        ffi_types=types,
        digest=digest,
        mailboxes=meta.get("mailboxes", "plain"),
    ), meta, sidecar


def resolve(
    stages,
    rows: int,
    seq: bool = False,
    group: int = 0,
    layer_args=KERNEL_ARGS,
    known=KNOWN_STAGES,
    spec=None,
    kind: str | None = None,
) -> tuple[ExternStage, ...]:
    """Resolve requested functions for one ML4 build variant."""
    spec = requested() if spec is None else dict(spec)
    unknown = set(spec) - set(known)
    if unknown:
        raise ValueError(f"{ENV}: unknown stage(s) {sorted(unknown)}")
    var = variant(rows, seq, group)
    family = kind_of(stages, kind)
    spec = {stage: path.replace("{kind}", family) for stage, path in spec.items()}
    result: list[ExternStage] = []
    metadata = {}
    for stage in stages:
        if stage not in spec or stage in metadata:
            continue
        loaded = _load(stage, spec[stage], var, layer_args)
        if loaded is None:
            continue
        ext, meta, sidecar = loaded
        result.append(ext)
        metadata[stage] = (sidecar, meta)
        print(
            f"[{ENV}] {stage} {var}: {ext.symbol} from {ext.path} ({ext.digest})",
            file=sys.stderr,
        )

    linked = {item.stage for item in result}
    for stage, (sidecar, meta) in metadata.items():
        for required, pair in (meta.get("requires") or {}).items():
            if required not in linked:
                raise ValueError(
                    f"{sidecar}: {stage} {var} needs {required} extern in the same build"
                )
            if pair is not None and metadata[required][1].get("pair") != pair:
                raise ValueError(
                    f"{sidecar}: {stage} {var} needs {required} of its own generation "
                    f"(pair {pair!r})"
                )
    return tuple(result)


def _check_numa(sidecar, meta) -> None:
    mailbox_layout = meta.get("mailboxes", "plain")
    if mailbox_layout not in ("plain", "numa1"):
        raise ValueError(
            f"{sidecar}: mailboxes {mailbox_layout!r}, expected 'plain' or 'numa1'"
        )
    header = meta.get("numa_hdr")
    if header is None:
        if mailbox_layout == "numa1":
            raise ValueError(f"{sidecar}: mailboxes 'numa1' without numa_hdr")
        return
    from . import numa

    if header != numa.header_offset():
        raise ValueError(
            f"{sidecar}: NUMA header {header}, expected {numa.header_offset()}"
        )
    if not numa.wanted():
        raise ValueError(
            f"{sidecar}: NUMA mailbox layout requires AITER_MONO_SCHED=numa=1"
        )


def numa_kept(externs, levers) -> bool:
    """Whether linked stages and the selected mailbox layout agree."""
    if not externs:
        return True
    numa_stages = [item for item in externs if item.mailboxes == "numa1"]
    if not numa_stages:
        return False
    if len(numa_stages) != len(externs) or levers.get("numa") != 1:
        raise ValueError(
            f"{ENV}: numa=1 mailbox layout disagrees with selected levers "
            f"({[(item.stage, item.mailboxes) for item in externs]})"
        )
    return True


def key(externs) -> tuple:
    return tuple((item.stage, item.digest) for item in externs)


def suffix(externs) -> str:
    if not externs:
        return ""
    return "_x" + hashlib.sha256(repr(key(externs)).encode()).hexdigest()[:8]


_FNS = {}


def _function(ext: ExternStage):
    from flydsl.compiler.extern_link import link_extern
    from flydsl.expr.extern import ffi

    fn = _FNS.get(ext)
    if fn is None:
        fn = _FNS[ext] = link_extern(
            ffi(ext.symbol, list(ext.ffi_types), "void"),
            bitcode_path=ext.path,
        )
    return fn


def call(ext, tag, task, peer_slot, raw_args, lds_base):
    """Emit one ML4 extern call at trace time.

    ``raw_args`` is the 38 kernel arguments in declaration order.  Keeping
    this assembly here makes the ABI boundary auditable and prevents the
    host-side ``param_table`` from accidentally being passed in its place.
    """
    if len(raw_args) != len(KERNEL_ARGS):
        raise ValueError(f"{ENV}: expected {len(KERNEL_ARGS)} kernel args, got {len(raw_args)}")
    _function(ext)(tag, task, peer_slot, *raw_args, lds_base)
