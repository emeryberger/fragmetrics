"""External-fragmentation metric family (M1-M6).

Every metric reads the same shared structure -- the coalesced free-run multiset
of a :class:`~fragmetrics.heap.Snapshot` -- so a snapshot is summarized once and
all metrics derive from it. The free-run lengths are lifted into a numpy array
(``runs``) up front; the heavy metrics (F(S), MWF) are vectorized over it.

Metrics
-------
M1  ``allocatable_count`` / ``fragmentation_at_size``   F(S), N(S)
M2  ``min_window_fill``                                 spatial-MMU MWF(W,S)
M3  ``window_fill_percentile`` + ``HeadroomSeries``     P95/P99 robust variants
M4  ``fragmentation_index``                             generalized Gorman Fidx(S)
M5  ``BlowupResult`` (blowup, ext_growth_rate)          workload-coupled
M6  ``checkerboard_index``, ``free_gini``, ``free_entropy``   cheap scalars

All public results are pydantic models or plain typed scalars; numpy arrays are
used internally and exposed as ``list[float]`` at the boundary so results are
JSON-serializable for dashboards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .heap import Heap, Snapshot
from .trace import Event, Op

FloatArray = NDArray[np.float64]


def _runs(snap: Snapshot) -> FloatArray:
    """Free-run lengths as a float array (the shared basis for all metrics)."""
    return np.array([r.length for r in snap.free_runs], dtype=np.float64)


# --- M1: allocatable count and fragmentation-at-size ----------------------


def allocatable_count(snap: Snapshot, size: int) -> int:
    """N(S): how many objects of ``size`` fit *now*, greedily, per-run."""
    if size <= 0:
        raise ValueError("size must be positive")
    runs = _runs(snap)
    if runs.size == 0:
        return 0
    return int(np.floor(runs / size).sum())


def ideal_allocatable_count(snap: Snapshot, size: int) -> int:
    """N*(S): count if all free space were perfectly coalesced."""
    if size <= 0:
        raise ValueError("size must be positive")
    return int(snap.total_free // size)


def fragmentation_at_size(snap: Snapshot, size: int) -> float:
    """F(S) = 1 - N(S)/N*(S) in [0, 1). 0 = no external fragmentation at S.

    Defined as 0 when even the ideal heap cannot hold one object (N*(S)=0):
    that is a capacity limit, not fragmentation (see ``fragmentation_index``).
    """
    ideal = ideal_allocatable_count(snap, size)
    if ideal == 0:
        return 0.0
    return 1.0 - allocatable_count(snap, size) / ideal


class FragmentationCurve(BaseModel):
    """F(S) sampled over a range of sizes, plus its scalar summary (AUC)."""

    model_config = ConfigDict(frozen=True)

    sizes: list[int]
    fragmentation: list[float]
    allocatable: list[int]

    @property
    def auc(self) -> float:
        """Area under F(S) on a log-size axis -- a single fragmentation scalar."""
        if len(self.sizes) < 2:
            return 0.0
        log_s = np.log(np.asarray(self.sizes, dtype=np.float64))
        f = np.asarray(self.fragmentation, dtype=np.float64)
        return float(np.trapezoid(f, log_s) / (log_s[-1] - log_s[0]))


def fragmentation_curve(snap: Snapshot, sizes: Sequence[int]) -> FragmentationCurve:
    """M1 headline object: F(S) and N(S) sampled over ``sizes``."""
    return FragmentationCurve(
        sizes=list(sizes),
        fragmentation=[fragmentation_at_size(snap, s) for s in sizes],
        allocatable=[allocatable_count(snap, s) for s in sizes],
    )


# --- M2/M3: spatial-MMU windowing and its robust percentile variants ------


def _window_fills(snap: Snapshot, window: int, size: int) -> FloatArray:
    """Per-window usable fraction u(win,S)/W for non-overlapping windows.

    Usable bytes in a window = sum over free runs intersecting the window of
    floor(intersection_len / size) * size. Windows tile [0, capacity); the
    final short window is scaled by its actual length.
    """
    if window <= 0 or size <= 0:
        raise ValueError("window and size must be positive")
    if snap.capacity == 0:
        return np.zeros(0, dtype=np.float64)
    n_windows = math.ceil(snap.capacity / window)
    usable = np.zeros(n_windows, dtype=np.float64)
    for run in snap.free_runs:
        # distribute this run's usable bytes across the windows it spans
        first = run.start // window
        last = (run.end - 1) // window
        for w in range(first, last + 1):
            lo = max(run.start, w * window)
            hi = min(run.end, (w + 1) * window)
            seg = hi - lo
            if seg > 0:
                usable[w] += (seg // size) * size
    # denominator: full window length, except the trailing partial window
    lengths = np.full(n_windows, float(window))
    tail = snap.capacity - (n_windows - 1) * window
    lengths[-1] = float(tail)
    fills: FloatArray = usable / lengths
    return fills


def min_window_fill(snap: Snapshot, window: int, size: int) -> float:
    """MWF(W,S): the minimum (worst) window's usable fraction -- the MMU analog."""
    fills = _window_fills(snap, window, size)
    return float(fills.min()) if fills.size else 0.0


def window_fill_percentile(snap: Snapshot, window: int, size: int, pct: float) -> float:
    """M3: low-tail percentile of per-window fill. pct=0 recovers MWF (the min).

    P95-latency-style robustness: ``pct=1`` is the P1 (near-worst) window,
    ``pct=5`` the P5, etc. Lower percentile = more pessimistic.
    """
    if not 0.0 <= pct <= 100.0:
        raise ValueError("pct must be in [0, 100]")
    fills = _window_fills(snap, window, size)
    if fills.size == 0:
        return 0.0
    return float(np.percentile(fills, pct))


# --- M4: generalized Gorman fragmentation index ---------------------------


def fragmentation_index(snap: Snapshot, size: int) -> float:
    """Fidx(S) in [-1, 1], generalizing Linux ``__fragmentation_index`` to any S.

    -1   : request would succeed (some free run is large enough)
    ->0  : failure due to insufficient *total* free memory (capacity problem)
    ->1  : failure despite enough total free memory (genuine external fragmentation)

    Closed form (Gorman): 1 - (requested / total_free + 1) / blocks, where
    ``blocks`` is the number of free runs and ``requested`` is one object. As the
    free space shatters into many runs, ``blocks`` grows and the index -> 1.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if snap.max_free >= size:
        return -1.0
    total_free = snap.total_free
    if total_free < size:
        return 0.0  # cannot satisfy even when fully coalesced -> capacity, not frag
    blocks = len(snap.free_runs)
    if blocks == 0:
        return 0.0
    return 1.0 - (size / total_free + 1.0) / blocks


# --- M5: blowup and operational external-fragmentation rate ---------------


class BlowupResult(BaseModel):
    """Workload-coupled fragmentation: blowup ratio + ext-frag failure rate."""

    model_config = ConfigDict(frozen=True)

    peak_heap: int
    peak_live: int
    blowup: float
    ext_growth_rate: float
    n_requests: int

    @property
    def overhead_fraction(self) -> float:
        """Fraction of peak heap that is *not* peak live data."""
        return 0.0 if self.peak_heap == 0 else 1.0 - self.peak_live / self.peak_heap


def replay_blowup(events: Sequence[Event], policy: str = "first-fit") -> BlowupResult:
    """M5: replay the whole trace, measuring blowup (Berger/Zorn/McKinley) vs the
    live high-water mark (Johnstone-Wilson methodology) and the operational
    external-fragmentation rate (requests that would fit only after compaction).
    """
    heap = Heap(policy)
    ext_frag_events = 0
    n_requests = 0
    for ev in events:
        if ev.op is Op.ALLOC:
            assert ev.size is not None
            n_requests += 1
            # would this allocation fail for *fragmentation* (not capacity)?
            snap = heap.snapshot(ev.ts)
            if snap.max_free < ev.size <= snap.total_free:
                ext_frag_events += 1
        heap.apply(ev)
    peak_live = heap.peak_live
    peak_heap = heap.peak_capacity
    blowup = float(peak_heap / peak_live) if peak_live else 1.0
    rate = ext_frag_events / n_requests if n_requests else 0.0
    return BlowupResult(
        peak_heap=peak_heap,
        peak_live=peak_live,
        blowup=blowup,
        ext_growth_rate=rate,
        n_requests=n_requests,
    )


# --- M6: cheap observability scalars --------------------------------------


def checkerboard_index(snap: Snapshot) -> float:
    """Fraction of address-space boundaries that flip used<->free.

    1.0 = maximally scattered (every adjacent pair alternates); 0.0 = fully
    segregated. Counts free<->used transitions implied by the free-run layout.
    """
    if snap.capacity == 0 or not snap.free_runs:
        return 0.0
    # transitions = 2 per interior free run, minus boundary runs touching an edge
    transitions = 0
    for run in snap.free_runs:
        if run.start > 0:
            transitions += 1  # used|free boundary on the left
        if run.end < snap.capacity:
            transitions += 1  # free|used boundary on the right
    # normalize by the number of free runs * 2 (max boundaries they could induce)
    return transitions / (2 * len(snap.free_runs))


def free_gini(snap: Snapshot) -> float:
    """Gini coefficient of free-run sizes in [0, 1]. 0 = all equal; ->1 = one dominates."""
    runs = np.sort(_runs(snap))
    n = runs.size
    if n == 0 or runs.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (index * runs).sum()) / (n * runs.sum()) - (n + 1.0) / n)


def free_entropy(snap: Snapshot) -> float:
    """Shannon entropy (bits) of the byte-share distribution across free runs.

    High entropy = free bytes spread evenly over many runs (scattered);
    low entropy = concentrated in few runs (good for large allocations).
    """
    runs = _runs(snap)
    total = runs.sum()
    if total == 0 or runs.size <= 1:
        return 0.0
    p = runs / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class SnapshotMetrics(BaseModel):
    """All scalar metrics for a single snapshot -- one dashboard row."""

    model_config = ConfigDict(frozen=True)

    ts: int
    capacity: int
    live_bytes: int
    total_free: int
    max_free: int
    occupancy: float            # legacy metric, for comparison
    max_free_ratio: float       # MaxFree / TotalFree
    checkerboard: float
    gini: float
    entropy: float


def summarize(snap: Snapshot) -> SnapshotMetrics:
    """Collapse a snapshot into the cheap scalar dashboard row (M6 + legacy)."""
    occupancy = snap.live_bytes / snap.capacity if snap.capacity else 0.0
    mf_ratio = snap.max_free / snap.total_free if snap.total_free else 0.0
    return SnapshotMetrics(
        ts=snap.ts,
        capacity=snap.capacity,
        live_bytes=snap.live_bytes,
        total_free=snap.total_free,
        max_free=snap.max_free,
        occupancy=occupancy,
        max_free_ratio=mf_ratio,
        checkerboard=checkerboard_index(snap),
        gini=free_gini(snap),
        entropy=free_entropy(snap),
    )
