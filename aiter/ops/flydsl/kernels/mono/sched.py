# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ML4 mono scheduling levers.

The parser and selector are host-side and intentionally have no FlyDSL
dependency, which makes them usable by CPU tests and launch configuration
tools.  Device helpers are imported lazily only when a knob-enabled kernel is
being traced.  With ``AITER_MONO_SCHED`` unset, :func:`resolve` returns an
empty tuple and the kernel has no schedule-specific code.
"""

from __future__ import annotations

import hashlib
import os
import re
from types import SimpleNamespace

from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import range_constexpr, rocdl

ENV = "AITER_MONO_SCHED"
ENV_DELAY = "AITER_MONO_DELAY"
MAX_ROWS = 12
GRID = 256
HEAD_SLEEP = 64

LEVER_KEYS = (
    "pause",
    "prio",
    "head_sleep",
    "hold",
    "place",
    "numa",
    "pipe",
    "b3pf",
    "mutate",
    "dyn",
)

# Kept deliberately conservative until ML4 timing data is available.  The
# string is still useful for exercising selector precedence and DSpark's
# r<q>s builds.
BEST = "numa=1"


def parse_delay(spec: str | None):
    """Parse optional causal entry delays using ML4 stage names."""
    result = []
    for item in filter(None, (part.strip() for part in (spec or "").split(";"))):
        match = re.fullmatch(
            r"([a-z][a-z0-9_]*)(?::(entry|post|exit))?(?:@(\d+)-(\d+))?=(\d+)",
            item,
        )
        if not match:
            raise ValueError(
                f"{ENV_DELAY}: bad item {item!r} (stage[:mode][@t0-t1]=units)"
            )
        stage, mode, begin, end, units = match.groups()
        begin, end = (
            (int(begin), int(end)) if begin is not None else (0, GRID)
        )
        if end <= begin:
            raise ValueError(f"{ENV_DELAY}: empty task range in {item!r}")
        if int(units):
            result.append((stage, mode or "entry", begin, end, int(units)))
    return tuple(sorted(result))


def variant_name(rows: int, seq: bool = False, group: int = 0) -> str:
    return f"r{rows}" + ("s" if seq else "") + (f"g{group}" if group else "")


def kind_of(stages) -> str:
    # ML4's layer geometry is selected by the imported variant module, not by
    # the stage names.  ``ml4`` is the stable schedule selector namespace.
    return "ml4"


def select(spec: str | None, kind: str, variant: str) -> str:
    """Select comma-separated lever groups for ``kind`` and ``variant``."""
    groups = []
    for group in (spec or "").split(";"):
        groups.extend(BEST.split(";") if group.strip() == "best" else [group])
    selected = []
    for group in groups:
        group = group.strip()
        if not group:
            continue
        selector, sep, body = group.rpartition(":")
        selector = selector.strip() if sep else "*"
        body = body.strip() if sep else group
        selected_kind, dot, selected_variant = selector.partition(".")
        if selected_kind not in ("*", kind):
            continue
        if dot and selected_variant not in ("*", variant):
            continue
        selected.append(body)
    return ",".join(selected)


def parse_levers(spec: str | None) -> tuple[tuple[str, int], ...]:
    """Parse ``key=N`` and stage selectors such as ``hold.ug=32``."""
    result = {}
    for item in filter(None, (part.strip() for part in (spec or "").split(","))):
        match = re.fullmatch(r"([a-z_][a-z0-9_]*?)(?:\.([a-z][a-z0-9_]*))?=(\d+)", item)
        if not match or match.group(1) not in LEVER_KEYS:
            raise ValueError(f"{ENV}: bad item {item!r} (keys {LEVER_KEYS})")
        key, stage, value = match.group(1), match.group(2), int(match.group(3))
        if key == "prio" and (stage is None or not 0 <= value <= 3):
            raise ValueError(f"{ENV}: prio.<stage>=0..3, got {item!r}")
        if key in ("hold", "place", "mutate", "dyn") and stage is None:
            raise ValueError(f"{ENV}: {key}.<stage>=N, got {item!r}")
        if key in ("head_sleep", "numa", "b3pf") and stage is not None:
            raise ValueError(f"{ENV}: {key} takes no stage, got {item!r}")
        result[key + (f".{stage}" if stage else "")] = value
    return tuple(sorted(result.items()))


def resolve(
    stamped: bool,
    rows: int,
    seq: bool = False,
    group: int = 0,
    stages=(),
    kind: str | None = None,
) -> tuple[tuple[str, object], ...]:
    selected = select(
        os.environ.get(ENV, ""),
        kind or kind_of(stages),
        variant_name(rows, seq, group),
    )
    levers = parse_levers(selected)
    if rows > MAX_ROWS:
        levers = tuple(
            item
            for item in levers
            if not item[0].startswith("place.") and item[0] != "numa"
        )
    delay = parse_delay(os.environ.get(ENV_DELAY, ""))
    result = list(("lever", item) for item in levers)
    if delay:
        result.append(("delay", delay))
    return tuple(result)


def from_knobs(knobs) -> SimpleNamespace:
    levers, delay = {}, ()
    for kind, value in knobs:
        if kind == "lever":
            levers[value[0]] = value[1]
        elif kind == "delay":
            delay = value
    return SimpleNamespace(levers=levers, delay=delay)


def suffix(knobs) -> str:
    return "" if not knobs else "_k" + hashlib.sha256(repr(knobs).encode()).hexdigest()[:8]


def without_numa(knobs, stages=(), rows=1, seq=False, group=0):
    """Drop NUMA when a plain-layout extern function is linked."""
    state = from_knobs(knobs)
    if not state.levers.get("numa"):
        return knobs
    print(
        f"[{ENV}] numa off for {kind_of(stages)} {variant_name(rows, seq, group)}: "
        "extern stages address the plain mailboxes",
        file=__import__("sys").stderr,
        flush=True,
    )
    return tuple(
        item
        for item in knobs
        if item != ("lever", ("numa", state.levers["numa"]))
    )


_T = None


def reset() -> None:
    global _T
    _T = None


def active() -> bool:
    return _T is not None


def begin(knobs, scratch=None, stages=()) -> None:
    """Enter device trace state for one knob-enabled build."""
    global _T
    _T = None
    if not knobs:
        return
    _T = SimpleNamespace(
        levers=from_knobs(knobs).levers,
        scratch=scratch,
        stages=tuple(stages),
        numa=None,
    )
    if _T.levers.get("numa"):
        from . import numa

        _T.numa = numa.entry_state(scratch, _T.levers["numa"])


def end() -> None:
    reset()


def numa_state():
    return None if _T is None else _T.numa


def hold(name: str, task, count: int) -> None:
    """Sleep a stage's task CTAs at entry, before their weight loads."""
    if _T is None:
        return
    units = _T.levers.get(f"hold.{name}", 0)
    if not units:
        return
    _hold(task, count, units)


def _hold_body(task, count, units):
    if task < count:
        for _ in range_constexpr(units // 127):
            rocdl.s_sleep(127)
        if units % 127:
            rocdl.s_sleep(units % 127)


_hold = ASTRewriter.transform(_hold_body)


def stage_enter(name: str, task, count: int) -> None:
    hold(name, task, count)


def pause_fn(default):
    if _T is None:
        return default
    value = _T.levers.get(f"pause.{_T.stage}", _T.levers.get("pause", 0))
    if not value:
        return default
    return lambda: rocdl.s_sleep(value)


def apply_layout(context, knobs) -> None:
    """Apply the optional static CTA placement lever when a build uses it."""
    values = {
        key[7:]: value
        for key, value in from_knobs(knobs).levers.items()
        if key.startswith("place.")
    }
    if not values:
        return
    base = dict(context.base)
    for stage, value in values.items():
        if stage in base:
            base[stage] = value % GRID
    context.base = base


def log_build(stages, rows, seq, group, knobs, externs=()) -> None:
    if not os.environ.get(ENV):
        return
    values = from_knobs(knobs).levers
    ext = " ".join(f"{item.stage}:{item.mailboxes}" for item in externs)
    text = " ".join(f"{key}={value}" for key, value in sorted(values.items())) or "none"
    print(
        f"[{ENV}] build {kind_of(stages)} {variant_name(rows, seq, group)}: {text}"
        + (f" | extern {ext}" if ext else ""),
        file=__import__("sys").stderr,
        flush=True,
    )
