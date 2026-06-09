"""Tests for the collection driver's pure-Python parts.

The interposition itself is exercised by the shim's own end-to-end test
(see shim/README.md); here we cover env construction, trace loading/sorting,
and error handling without requiring a built shim.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fragmetrics import collect
from fragmetrics.trace import Event, Op, write_jsonl


def test_preload_env_orders_libraries() -> None:
    libs = [Path("/x/libfragtrace.so"), Path("/y/libmimalloc.so")]
    env = collect.preload_env(libs, base_env={})
    key = "DYLD_INSERT_LIBRARIES" if collect.is_macos() else "LD_PRELOAD"
    assert key in env
    # tracer first so it forwards to the production allocator
    assert env[key].split(os.pathsep)[0].endswith("libfragtrace.so")


def test_preload_env_preserves_base() -> None:
    env = collect.preload_env([Path("/x/lib.so")], base_env={"FOO": "bar"})
    assert env["FOO"] == "bar"


def test_collect_missing_tracer_raises(tmp_path: Path) -> None:
    with pytest.raises(collect.ShimNotBuiltError):
        collect.collect(
            ["true"],
            tracer=tmp_path / "does-not-exist.so",
            out=tmp_path / "t.jsonl",
        )


def test_load_trace_sorts_by_ts(tmp_path: Path) -> None:
    # write events deliberately out of ts order
    events = [
        Event(ts=2, op=Op.FREE, id=1),
        Event(ts=0, op=Op.ALLOC, id=1, size=64),
        Event(ts=1, op=Op.ALLOC, id=2, size=32),
    ]
    path = tmp_path / "unordered.jsonl"
    write_jsonl(events, path)
    loaded = collect.load_trace(path)
    assert [e.ts for e in loaded] == [0, 1, 2]


def test_load_trace_handles_large_pointer_ids(tmp_path: Path) -> None:
    # real malloc addresses are large; the schema must accept them as ids/addrs
    big = 0x7F_FFFF_FFFF
    events = [Event(ts=0, op=Op.ALLOC, id=big, size=16, addr=big)]
    path = tmp_path / "big.jsonl"
    write_jsonl(events, path)
    loaded = collect.load_trace(path)
    assert loaded[0].id == big
    assert loaded[0].addr == big
