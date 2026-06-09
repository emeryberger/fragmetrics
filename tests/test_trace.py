"""Trace schema validation and file round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from fragmetrics import workload as w
from fragmetrics.trace import Event, Op, read_jsonl, write_jsonl


def test_alloc_requires_size() -> None:
    with pytest.raises(ValidationError):
        Event(ts=0, op=Op.ALLOC, id=1)


def test_free_rejects_size() -> None:
    with pytest.raises(ValidationError):
        Event(ts=0, op=Op.FREE, id=1, size=64)


def test_alloc_rejects_nonpositive_size() -> None:
    with pytest.raises(ValidationError):
        Event(ts=0, op=Op.ALLOC, id=1, size=0)


def test_unknown_op_rejected() -> None:
    with pytest.raises(ValidationError):
        Event.model_validate({"ts": 0, "op": "realloc", "id": 1, "size": 8})


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Event.model_validate({"ts": 0, "op": "free", "id": 1, "bogus": 5})


def test_jsonl_round_trip(tmp_path: Path) -> None:
    original = w.checkerboard(block=64, n=8)
    path = tmp_path / "trace.jsonl"
    write_jsonl(original, path)
    restored = list(read_jsonl(path))
    assert restored == original


def test_jsonl_round_trip_random(tmp_path: Path) -> None:
    original = w.generate(w.WorkloadConfig(n_events=500, seed=4))
    path = tmp_path / "trace.jsonl"
    write_jsonl(original, path)
    restored = list(read_jsonl(path))
    assert restored == original
