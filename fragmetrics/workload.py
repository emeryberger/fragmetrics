"""Synthetic workload generation.

Two layers:

1. **Canonical fixtures** -- tiny, deterministic, address-resolved traces that
   pin down metric behaviour in the golden tests (fully-coalesced,
   checkerboard, capacity-exhaustion). These carry explicit ``addr`` so they
   need no placement policy.

2. **Stochastic generator** -- a configurable workload (size distribution,
   object lifetimes, alloc/free mix, phase changes) that *induces* realistic
   fragmentation. It emits address-free traces; addresses come from a placement
   policy at replay time.

The generator is driven by an explicit ``numpy.random.Generator`` (seeded by the
caller) so traces are reproducible without touching global RNG state.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from .trace import Event, Op

IntArray = NDArray[np.int64]


class SizeDist(str, Enum):
    """Object-size distribution families."""

    UNIFORM = "uniform"
    ZIPF = "zipf"          # few huge, many tiny -- heavy tailed
    MIXTURE = "mixture"    # bimodal: small objects + occasional large


class WorkloadConfig(BaseModel):
    """Parameters for the stochastic generator."""

    model_config = ConfigDict(frozen=True)

    n_events: PositiveInt = 10_000
    seed: int = 0
    size_dist: SizeDist = SizeDist.MIXTURE
    min_size: PositiveInt = 8
    max_size: PositiveInt = 4096
    zipf_a: float = Field(default=1.5, gt=1.0)
    # mean object lifetime, in number of subsequent events before free becomes eligible
    mean_lifetime: PositiveInt = 200
    # probability the next event is a free (vs alloc), when frees are available
    free_bias: float = Field(default=0.5, ge=0.0, le=1.0)
    # number of phases; each phase resamples the size distribution scale
    n_phases: PositiveInt = 1


def _draw_sizes(cfg: WorkloadConfig, rng: np.random.Generator, n: int, scale: float) -> IntArray:
    lo, hi = cfg.min_size, int(cfg.max_size * scale)
    hi = max(hi, lo + 1)
    sizes: IntArray
    if cfg.size_dist is SizeDist.UNIFORM:
        sizes = rng.integers(lo, hi + 1, size=n)
    elif cfg.size_dist is SizeDist.ZIPF:
        raw = rng.zipf(cfg.zipf_a, size=n)
        sizes = lo + (raw % (hi - lo + 1))
    else:  # MIXTURE -- 90% small, 10% large
        is_large = rng.random(n) < 0.1
        small = rng.integers(lo, max(lo + 1, hi // 8) + 1, size=n)
        large = rng.integers(max(lo + 1, hi // 2), hi + 1, size=n)
        sizes = np.where(is_large, large, small)
    return sizes.astype(np.int64)


def generate(cfg: WorkloadConfig) -> list[Event]:
    """Generate a stochastic, address-free trace as a list of events.

    Uses a min-heap keyed on scheduled free-time so objects are freed roughly
    after ``mean_lifetime`` events, producing overlapping lifetimes (the source
    of external fragmentation).
    """
    rng = np.random.default_rng(cfg.seed)
    events: list[Event] = []
    # (scheduled_free_ts, obj_id)
    live: list[tuple[int, int]] = []
    next_id = 0
    phase_len = max(1, cfg.n_events // cfg.n_phases)

    for ts in range(cfg.n_events):
        phase = ts // phase_len
        # phase scale cycles through a few magnitudes to force layout churn
        scale = float(1.0 + (phase % cfg.n_phases) * 0.5)

        # free anything whose time has come, or with free_bias probability
        due = live and live[0][0] <= ts
        if due or (live and rng.random() < cfg.free_bias):
            _, obj_id = heapq.heappop(live)
            events.append(Event(ts=ts, op=Op.FREE, id=obj_id))
            continue

        size = int(_draw_sizes(cfg, rng, 1, scale)[0])
        obj_id = next_id
        next_id += 1
        events.append(Event(ts=ts, op=Op.ALLOC, id=obj_id, size=size))
        lifetime = int(rng.exponential(cfg.mean_lifetime)) + 1
        heapq.heappush(live, (ts + lifetime, obj_id))

    return events


# --- canonical fixtures (address-resolved, deterministic) -----------------


def coalesced(block: int = 64, n: int = 16) -> list[Event]:
    """Allocate ``n`` blocks contiguously, then free every other-from-the-end so
    the survivors leave one big contiguous free region (no fragmentation).

    Concretely: allocate n blocks, then free the *top half* -- their addresses
    are contiguous, so freeing them yields a single coalesced free run.
    """
    events: list[Event] = []
    ts = 0
    for i in range(n):
        events.append(Event(ts=ts, op=Op.ALLOC, id=i, size=block, addr=i * block))
        ts += 1
    for i in range(n // 2, n):  # free the contiguous top half
        events.append(Event(ts=ts, op=Op.FREE, id=i))
        ts += 1
    return events


def checkerboard(block: int = 64, n: int = 16) -> list[Event]:
    """The canonical external-fragmentation case.

    Allocate ``n`` adjacent blocks, then free every *other* one. Result: n/2
    free runs of length ``block``, each separated by a live block. Total free =
    half the heap, but the largest contiguous free run is just ``block`` -- so a
    request of ``2*block`` cannot be satisfied despite ample free bytes. The
    occupancy metric (in-use/allocated = 0.5) is blind to this; F(2*block)->1.
    """
    events: list[Event] = []
    ts = 0
    for i in range(n):
        events.append(Event(ts=ts, op=Op.ALLOC, id=i, size=block, addr=i * block))
        ts += 1
    for i in range(0, n, 2):  # free even-indexed blocks -> alternating holes
        events.append(Event(ts=ts, op=Op.FREE, id=i))
        ts += 1
    return events


def capacity_exhaustion(block: int = 64, n: int = 16) -> list[Event]:
    """Fill the heap and free only the *last* block: free space is one small run
    that is genuinely too small for a large request -- a capacity limit, not
    fragmentation. ``fragmentation_index`` should report ->0 here, not ->1.
    """
    events: list[Event] = []
    ts = 0
    for i in range(n):
        events.append(Event(ts=ts, op=Op.ALLOC, id=i, size=block, addr=i * block))
        ts += 1
    events.append(Event(ts=ts, op=Op.FREE, id=n - 1))
    return events


FIXTURES = {
    "coalesced": coalesced,
    "checkerboard": checkerboard,
    "capacity-exhaustion": capacity_exhaustion,
}


def fixture(name: str) -> list[Event]:
    """Look up a canonical fixture trace by name."""
    try:
        return FIXTURES[name]()
    except KeyError:
        raise ValueError(f"unknown fixture {name!r}; choose from {sorted(FIXTURES)}") from None


def iter_events(events: list[Event]) -> Iterator[Event]:
    """Trivial iterator helper (keeps call sites uniform with file readers)."""
    yield from events
