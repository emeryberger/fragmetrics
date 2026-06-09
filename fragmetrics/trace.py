"""Trace schema and readers.

A trace is a time-ordered stream of allocation events. The canonical on-disk
format is JSONL, one object per line::

    {"ts": 0, "op": "alloc", "id": 1, "size": 64}
    {"ts": 1, "op": "alloc", "id": 2, "size": 64, "addr": 64}
    {"ts": 2, "op": "free",  "id": 1}

Fields
------
ts   : monotonic event index (or wall-clock); only ordering matters.
op   : "alloc" or "free".
id   : caller-chosen object identifier, unique among live objects. A "free"
       references the id returned by a prior "alloc".
size : object size in bytes. Required (and positive) on "alloc", forbidden on "free".
addr : OPTIONAL. If present, the trace already carries a placement (an
       address-resolved trace). If absent, a placement policy in ``heap.py``
       assigns addresses during replay. This is the key flexibility: external
       fragmentation needs addresses, but most real traces lack them, so we
       reconstruct them.

Validation happens at the boundary via pydantic, so the rest of the codebase
receives fully-typed, semantically-valid ``Event`` instances.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator


class Op(str, Enum):
    """Allocation event operations."""

    ALLOC = "alloc"
    FREE = "free"


class Event(BaseModel):
    """A single, validated allocation-trace event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ts: int
    op: Op
    id: int
    size: PositiveInt | None = None
    addr: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _check_op_fields(self) -> Event:
        if self.op is Op.ALLOC and self.size is None:
            raise ValueError(f"alloc event id={self.id} requires a positive size")
        if self.op is Op.FREE and self.size is not None:
            raise ValueError(f"free event id={self.id} must not carry a size")
        return self


def events_from_dicts(rows: Iterable[dict[str, Any]]) -> Iterator[Event]:
    """Normalize raw dict rows (from JSON/CSV) into validated ``Event``s.

    Missing/blank ``ts`` defaults to the row index so positional traces work.
    Blank optional fields (common in CSV) are dropped before validation.
    """
    for i, row in enumerate(rows):
        cleaned: dict[str, Any] = {k: v for k, v in row.items() if v not in (None, "")}
        cleaned.setdefault("ts", i)
        try:
            yield Event.model_validate(cleaned)
        except ValueError as exc:
            raise ValueError(f"malformed event at row {i}: {row!r} ({exc})") from exc


def read_jsonl(path: str | Path) -> Iterator[Event]:
    """Read a JSONL trace file."""
    with open(path, encoding="utf-8") as fh:
        rows = (json.loads(line) for line in fh if line.strip())
        yield from events_from_dicts(rows)


def read_csv(path: str | Path) -> Iterator[Event]:
    """Read a CSV trace file (header row required: ts,op,id,size,addr)."""
    with open(path, encoding="utf-8", newline="") as fh:
        yield from events_from_dicts(csv.DictReader(fh))


def read_trace(path: str | Path) -> Iterator[Event]:
    """Read a trace, dispatching on file extension (.jsonl/.json vs .csv)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".jsonl", ".json", ".ndjson"):
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"unsupported trace extension {suffix!r} for {path}")


def write_jsonl(events: Iterable[Event], path: str | Path) -> None:
    """Write events to a JSONL file (round-trips with ``read_jsonl``)."""
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(ev.model_dump_json(exclude_none=True) + "\n")
