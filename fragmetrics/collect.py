"""Collect allocation traces from real programs via library interposition.

This drives the ``fragtrace`` shim (see ``shim/``) -- or any alloc8-based tracing
allocator -- to capture a program's allocation request stream, optionally with a
production allocator (mimalloc / jemalloc / Hoard) loaded underneath.

Methodology note
----------------
Interposition observes the *request stream* (sizes + lifetimes) the program makes
of its allocator, not the allocator's internal free lists. Loading mimalloc vs
jemalloc underneath does not change the captured trace -- the program asks for
the same bytes. The value is capturing a *realistic workload* that we then replay
through fragmetrics' reference policies (the Johnstone-Wilson methodology) to
compare fragmentation behaviour on a common, workload-coupled scale.

To stack a production allocator *under* the shim, both are preloaded; the shim
forwards every call to the next allocator in the chain (the production one), so
the production allocator does the real work while the shim records the requests.
"""

from __future__ import annotations

import os
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .trace import Event, read_jsonl


class ShimNotBuiltError(RuntimeError):
    """Raised when the requested interposition library cannot be found."""


class CollectResult(BaseModel):
    """Outcome of a collection run."""

    model_config = ConfigDict(frozen=True)

    trace_path: Path
    n_events: int
    returncode: int
    command: list[str]


def is_macos() -> bool:
    return platform.system() == "Darwin"


def preload_env(libs: Sequence[Path], base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build an environment that preloads ``libs`` in order.

    The first library wins symbol resolution, so put the *tracer* first and any
    production allocator after it: the tracer forwards to the next allocator in
    the chain. Uses ``DYLD_INSERT_LIBRARIES`` on macOS, ``LD_PRELOAD`` on Linux.
    """
    env = dict(base_env if base_env is not None else os.environ)
    joined = os.pathsep.join(str(p) for p in libs)
    if is_macos():
        env["DYLD_INSERT_LIBRARIES"] = joined
        # required on macOS to interpose into system-protected binaries' children
        env.setdefault("DYLD_FORCE_FLAT_NAMESPACE", "1")
    else:
        env["LD_PRELOAD"] = joined
    return env


def collect(
    command: Sequence[str],
    *,
    tracer: Path,
    out: Path,
    allocator: Path | None = None,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> CollectResult:
    """Run ``command`` under the tracer (and optional production allocator),
    writing a JSONL trace to ``out``; return a summary.

    Parameters
    ----------
    tracer    : path to the fragtrace interposition library (.so/.dylib).
    allocator : optional production allocator library to load *under* the tracer
                (e.g. libmimalloc.so). Loaded second so the tracer forwards to it.
    out       : trace output path (passed to the shim via FRAGTRACE_OUT).
    """
    if not tracer.exists():
        raise ShimNotBuiltError(
            f"tracer library not found: {tracer}. Build it first (see shim/README.md)."
        )
    libs = [tracer] + ([allocator] if allocator else [])
    if allocator and not allocator.exists():
        raise ShimNotBuiltError(f"allocator library not found: {allocator}")

    env = preload_env(libs)
    env["FRAGTRACE_OUT"] = str(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(  # noqa: S603 -- command is user-supplied by design
        list(command),
        env=env,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=False,
    )
    n = sum(1 for _ in read_jsonl(out)) if out.exists() else 0
    return CollectResult(
        trace_path=out,
        n_events=n,
        returncode=proc.returncode,
        command=list(command),
    )


def load_trace(path: str | Path) -> list[Event]:
    """Read a collected trace and return events in timestamp order.

    The shim stamps a monotonic atomic ``ts`` per event, but multi-threaded
    programs may write lines out of order; sorting by ``ts`` restores the true
    program order before replay.
    """
    events = list(read_jsonl(path))
    events.sort(key=lambda e: e.ts)
    return events
