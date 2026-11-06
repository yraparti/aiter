# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only contract tests for ML4 extern stages and schedule levers."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from aiter.ops.flydsl.kernels.mono import extern_stages as extern
from aiter.ops.flydsl.kernels.mono import numa, sched


def _write(directory, stage, variant, *, body="; fn\n", **metadata):
    (directory / f"{stage}_{variant}.ll").write_text(body)
    sidecar = {
        "symbol": f"{extern.SYMBOL_PREFIX}{stage}_{variant}",
        "abi": extern.expected_abi(),
        "ffi_types": extern.ffi_types(),
        **metadata,
    }
    (directory / f"{stage}_{variant}.json").write_text(json.dumps(sidecar))


def test_extern_is_off_and_abi_has_38_kernel_arguments(monkeypatch):
    monkeypatch.delenv(extern.ENV, raising=False)
    assert extern.requested() == {}
    assert extern.resolve(("router", "ug", "down"), 1) == ()
    abi = extern.expected_abi()
    assert len(extern.KERNEL_ARGS) == 38
    assert len(abi) == 42
    assert abi[:3] == [["tag", "i32"], ["task", "i32"], ["peer_slot", "i32"]]
    assert abi[-1] == ["lds_base", "i32"]


def test_variant_names_include_dspark_sequence():
    assert extern.variant(1) == "r1"
    assert extern.variant(2) == "r2"
    assert extern.variant(4) == "r4"
    assert extern.variant(8, True) == "r8s"


@pytest.mark.parametrize("form", ["directory", "template"])
def test_resolve_and_digest(tmp_path, form):
    _write(tmp_path, "router", "r4")
    path = (
        str(tmp_path)
        if form == "directory"
        else str(tmp_path / "router_{variant}.ll")
    )
    (resolved,) = extern.resolve(
        ("router", "ug", "down"),
        4,
        spec={"router": path},
    )
    assert resolved.symbol == "rocke_ml4_router_r4"
    assert resolved.path.endswith("router_r4.ll")
    before = extern.key((resolved,))
    _write(tmp_path, "router", "r4", body="; changed\n")
    (changed,) = extern.resolve(
        ("router",),
        4,
        spec={"router": str(tmp_path)},
    )
    assert extern.key((changed,)) != before
    assert extern.suffix((changed,)).startswith("_x")


def test_missing_variants_and_pairing(tmp_path, monkeypatch):
    _write(tmp_path, "ug", "r1")
    with pytest.raises(FileNotFoundError):
        extern.resolve(("ug",), 2, spec={"ug": str(tmp_path)})
    monkeypatch.setenv(extern.MISSING_ENV, "mono")
    assert extern.resolve(("ug",), 2, spec={"ug": str(tmp_path)}) == ()

    _write(tmp_path, "router", "r4", pair="g1", requires={"ug": "g1"})
    _write(tmp_path, "ug", "r4", pair="g1")
    linked = extern.resolve(
        ("router", "ug"),
        4,
        spec={"router": str(tmp_path), "ug": str(tmp_path)},
    )
    assert [item.stage for item in linked] == ["router", "ug"]


def test_schedule_grammar_and_selectors(monkeypatch):
    assert sched.parse_levers("hold.ug=32,pause=2,prio.down=3") == (
        ("hold.ug", 32),
        ("pause", 2),
        ("prio.down", 3),
    )
    assert sched.select(
        "*:hold.ug=8;ml4.r4s:hold.down=16",
        "ml4",
        "r4s",
    ) == "hold.ug=8,hold.down=16"
    monkeypatch.setenv(sched.ENV, "hold.ug=32")
    assert sched.from_knobs(sched.resolve(False, 4, True)).levers == {"hold.ug": 32}
    assert sched.resolve(False, 13) == ()
    with pytest.raises(ValueError):
        sched.parse_levers("hold=2")


def test_numa_box_views_are_bitwise_and_layout_is_bounded():
    values = [0, 8, CHUNK := 8192, 16384 + 24, numa._mailbox_bytes() - 8]
    physical = [numa.phys(value) for value in values]
    assert physical[:2] == [0, 8]
    assert physical[2] == 16384
    assert physical[3] == 32768 + 24
    assert physical[-1] + CHUNK <= numa.area_bytes()
    for pointer in (0, 1 << 20, (5 << 21) + 4096):
        base = numa.area_base(pointer)
        assert base % numa.PAGE == 0
        assert base + numa.area_bytes() <= pointer + numa.scratch_bytes_needed()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
