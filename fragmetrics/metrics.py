"""External-fragmentation metric family (M1-M6).

Every metric reads the same shared structure -- the coalesced free-run multiset
of a :class:`~fragmetrics.heap.Snapshot` -- so a snapshot is summarized once and
all metrics derive from it. The free-run lengths are lifted into a numpy array
(``runs``) up front; the heavy metrics (F(S), the windowed CDFs) are vectorized
over it.

Metrics
-------
M1  ``allocatable_count`` / ``fragmentation_at_size``   F(S), N(S)
M1b ``usable_free_curve`` / ``pooled_usable_free_curve``
    (+ ``.unusable`` loss orientation)                  usable-free curve
M2  ``occupancy_distribution`` (+ ``pooled_occupancy``,
    ``occupancy_spectrum``)                             occupancy CDF O_W
M3  ``usability_distribution`` (+ ``pooled_usability``) usability CDF U_{W,S}
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

from .heap import Heap, Snapshot, replay
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


# --- M1b: usable-free curve over request size ------------------------------
#
# The byte-weighted companion to F(S): what fraction of the unallocated space
# is actually usable at request size S? Two variants bracket the answer.
# ``contiguous`` is the survival function of the byte-weighted free-extent
# length distribution (monotone non-increasing, a true 1-CDF over S);
# ``packed`` additionally charges the per-run packing loss of carving objects
# of exactly S (floor(b/S)*S), so packed <= contiguous, with a sawtooth within
# each octave. The gap between the two is pure packing loss; the cliff in
# ``contiguous`` is the characteristic free-run size. ``packed`` equals the
# byte-weighted mean of the M3 usability CDF at W >= capacity, and generalizes
# Gorman's UFSI complement from powers of two to arbitrary S. (Objects of size
# "<= S" would be degenerate -- tiny objects can always fill everything -- so
# both variants carve at exactly S.)


def packed_usable_fraction(snap: Snapshot, size: int) -> float:
    """Fraction of free bytes usable when carving size-S objects: N(S)*S/TotalFree."""
    if size <= 0:
        raise ValueError("size must be positive")
    runs = _runs(snap)
    total = runs.sum()
    if total == 0:
        return 0.0
    return float((runs // size * size).sum() / total)


def contiguous_free_fraction(snap: Snapshot, size: int) -> float:
    """Fraction of free bytes in runs >= S: the free-extent survival function."""
    if size <= 0:
        raise ValueError("size must be positive")
    runs = _runs(snap)
    total = runs.sum()
    if total == 0:
        return 0.0
    return float(runs[runs >= size].sum() / total)


class UsableFreeCurve(BaseModel):
    """packed/contiguous usable-free fractions sampled over request sizes."""

    model_config = ConfigDict(frozen=True)

    sizes: list[int]
    packed: list[float]
    contiguous: list[float]

    @property
    def unusable(self) -> list[float]:
        """Loss orientation (UFSI-style): 1 - packed. Up = bad, 0 = perfect.

        Preferred for presentation: it reads consistently with F(S) and the
        Gorman indices, keeps the interesting region (a few percent loss) off
        the axis ceiling, and admits a log scale.
        """
        return [1.0 - p for p in self.packed]

    @property
    def unusable_auc(self) -> float:
        """Area under the unusable curve on a log-size axis, in [0, 1] --
        the M1b scalar companion to ``FragmentationCurve.auc``."""
        if len(self.sizes) < 2:
            return 0.0
        log_s = np.log(np.asarray(self.sizes, dtype=np.float64))
        u = 1.0 - np.asarray(self.packed, dtype=np.float64)
        return float(np.trapezoid(u, log_s) / (log_s[-1] - log_s[0]))


def usable_free_curve(snap: Snapshot, sizes: Sequence[int]) -> UsableFreeCurve:
    """M1b: the usable-free curve y(S) over ``sizes`` (see module comment)."""
    return UsableFreeCurve(
        sizes=list(sizes),
        packed=[packed_usable_fraction(snap, s) for s in sizes],
        contiguous=[contiguous_free_fraction(snap, s) for s in sizes],
    )


def pooled_usable_free_curve(
    events: Sequence[Event], policy: str, sizes: Sequence[int], *, every: int = 1
) -> UsableFreeCurve:
    """Whole-trace M1b: fractions pooled over the replay, each snapshot
    weighted by dwell time x free bytes. ``packed[j]`` reads "over the run,
    this fraction of free byte-time was usable when carving size-S objects";
    ``contiguous`` likewise for "sat in runs >= S"."""
    snaps, dwell = _snapshot_dwell(events, policy, every)
    sz = np.asarray(sizes, dtype=np.float64)
    packed_num = np.zeros(len(sizes))
    contig_num = np.zeros(len(sizes))
    denom = 0.0
    for snap, dt in zip(snaps, dwell):
        runs = _runs(snap)
        total = runs.sum()
        weight = dt * total
        if weight <= 0:
            continue
        denom += weight
        cols = runs[:, None]
        packed_num += weight * (np.floor(cols / sz) * sz).sum(axis=0) / total
        contig_num += weight * np.where(cols >= sz, cols, 0.0).sum(axis=0) / total
    if denom == 0:
        zeros = [0.0] * len(sizes)
        return UsableFreeCurve(sizes=list(sizes), packed=zeros, contiguous=list(zeros))
    return UsableFreeCurve(
        sizes=list(sizes),
        packed=(packed_num / denom).tolist(),
        contiguous=(contig_num / denom).tolist(),
    )


# --- M2/M3: windowed distributions (occupancy and usability CDFs) ---------
#
# The old MWF(W,S) = min over windows of usable_free/W conflated two opposite
# conditions: a window with NO free space (healthy density) scored 0 exactly
# like a window whose free space is shredded below S (pathology) -- a perfectly
# compacted heap had MWF = 0. The redefinition splits the windowing idea into
# two clean distributions over tiled windows, reported as full (weighted)
# empirical CDFs rather than a single order statistic:
#
#   M2  occupancy_distribution(W)    value = occupied/W,      weight = W bytes
#   M3  usability_distribution(W,S)  value = usable/free,     weight = free bytes
#
# Identities: the weighted mean of M2 is exactly global occupancy at every W;
# the weighted mean of M3 at W >= capacity is exactly N(S)*S/TotalFree (the
# complement of Gorman's unusable-free-space index). With dyadic windows, M2 at
# scale 2W is the length-weighted mean of its two children, so its variance is
# non-increasing in W -- the decay of spread with scale is the fragmentation
# spectrum (see ``occupancy_spectrum``).


class WindowDistribution(BaseModel):
    """A byte-weighted empirical distribution over tiled address-space windows.

    ``values`` are sorted ascending with aligned ``weights`` (bytes). The CDF is
    F(u) = fraction of weight on windows with value <= u; ``quantile(0)`` is the
    worst window (the old MMU-style min, now a derived view).

    Caveat: free runs are clipped at window boundaries before ``floor(seg/S)``,
    so a run straddling a boundary can undercount usable bytes in both windows.
    The bias inflates the low tail as S approaches W; the mean identities above
    are exact only at W >= capacity. Prefer W >> S when reading tails.
    """

    model_config = ConfigDict(frozen=True)

    window: int
    size: int | None = None  # set for usability distributions
    values: list[float]
    weights: list[float]

    @property
    def total_weight(self) -> float:
        return float(sum(self.weights))

    @property
    def mean(self) -> float:
        """Byte-weighted mean (0.0 for an empty distribution)."""
        total = self.total_weight
        if total == 0:
            return 0.0
        v = np.asarray(self.values)
        w = np.asarray(self.weights)
        return float((v * w).sum() / total)

    @property
    def std(self) -> float:
        """Byte-weighted standard deviation (0.0 for an empty distribution)."""
        total = self.total_weight
        if total == 0:
            return 0.0
        v = np.asarray(self.values)
        w = np.asarray(self.weights)
        mu = (v * w).sum() / total
        return float(math.sqrt(((v - mu) ** 2 * w).sum() / total))

    def quantile(self, pct: float) -> float:
        """Weighted quantile in [0, 100]. pct=0 is the worst window (old MWF min)."""
        if not 0.0 <= pct <= 100.0:
            raise ValueError("pct must be in [0, 100]")
        if not self.values:
            return 0.0
        cw = np.cumsum(self.weights)
        idx = int(np.searchsorted(cw, pct / 100.0 * cw[-1], side="left"))
        return self.values[min(idx, len(self.values) - 1)]

    def cdf_at(self, u: float) -> float:
        """F(u): fraction of weight on windows with value <= u.

        E.g. ``occupancy_distribution(snap, PAGE).cdf_at(0.5)`` is the fraction
        of the heap sitting in pages at most half occupied -- the reclaim/Mesh
        tail mass.
        """
        if not self.values:
            return 0.0
        idx = int(np.searchsorted(np.asarray(self.values), u, side="right"))
        if idx == 0:
            return 0.0
        return float(np.cumsum(self.weights)[idx - 1] / self.total_weight)

    @classmethod
    def pool(
        cls,
        dists: Sequence["WindowDistribution"],
        time_weights: Sequence[float] | None = None,
    ) -> "WindowDistribution":
        """Pool per-snapshot distributions into one, scaling each snapshot's
        byte weights by its ``time_weights`` entry (e.g. event-clock dwell time).
        The result is a distribution over (window, instant) samples: byte-time.
        """
        if time_weights is not None and len(time_weights) != len(dists):
            raise ValueError("time_weights must align with dists")
        kept = [
            (d, 1.0 if time_weights is None else float(time_weights[i]))
            for i, d in enumerate(dists)
            if d.values and (time_weights is None or time_weights[i] > 0)
        ]
        if not kept:
            base = dists[0] if dists else None
            return cls(window=base.window if base else 0,
                       size=base.size if base else None, values=[], weights=[])
        window, size = kept[0][0].window, kept[0][0].size
        if any(d.window != window or d.size != size for d, _ in kept):
            raise ValueError("cannot pool distributions with differing window/size")
        v = np.concatenate([np.asarray(d.values) for d, _ in kept])
        w = np.concatenate([np.asarray(d.weights) * tw for d, tw in kept])
        order = np.argsort(v, kind="stable")
        return cls(window=window, size=size,
                   values=v[order].tolist(), weights=w[order].tolist())


def _window_profile(
    snap: Snapshot, window: int, size: int | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per tiled window: (length, free bytes, usable-for-size bytes) arrays.

    Windows tile [0, capacity); the trailing window may be short. ``usable`` is
    all-zero when ``size`` is None (occupancy only needs free bytes).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if size is not None and size <= 0:
        raise ValueError("size must be positive")
    if snap.capacity == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z.copy(), z.copy()
    n_windows = math.ceil(snap.capacity / window)
    free = np.zeros(n_windows, dtype=np.float64)
    usable = np.zeros(n_windows, dtype=np.float64)
    for run in snap.free_runs:
        first = run.start // window
        last = (run.end - 1) // window
        for w in range(first, last + 1):
            lo = max(run.start, w * window)
            hi = min(run.end, (w + 1) * window)
            seg = hi - lo
            if seg > 0:
                free[w] += seg
                if size is not None:
                    usable[w] += (seg // size) * size
    lengths = np.full(n_windows, float(window))
    lengths[-1] = float(snap.capacity - (n_windows - 1) * window)
    return lengths, free, usable


def _sorted_dist(
    window: int, size: int | None, values: FloatArray, weights: FloatArray
) -> WindowDistribution:
    order = np.argsort(values, kind="stable")
    return WindowDistribution(
        window=window, size=size,
        values=values[order].tolist(), weights=weights[order].tolist(),
    )


def occupancy_distribution(snap: Snapshot, window: int) -> WindowDistribution:
    """M2: the occupancy CDF O_W -- occupied fraction per window, weighted by
    window length. The spatial analog of MMU's mutator utilization.

    The low tail is the actionable part: nearly-empty windows are what decommit
    (W = page) or huge-page demotion (W = 2 MiB) can reclaim, and bimodality at
    page scale is the Mesh signal. The weighted mean equals global occupancy
    ``used_bytes/capacity`` at every W. "Occupied" counts placed footprints
    (buddy's rounding slack is occupied), matching the free-run structure.
    """
    lengths, free, _ = _window_profile(snap, window, None)
    if lengths.size == 0:
        return WindowDistribution(window=window, values=[], weights=[])
    occ: FloatArray = (lengths - free) / lengths
    return _sorted_dist(window, None, occ, lengths)


def usability_distribution(snap: Snapshot, window: int, size: int) -> WindowDistribution:
    """M3: the usability CDF U_{W,S} -- per window, the fraction of its FREE
    bytes usable for size-S objects (``sum floor(seg/S)*S / free``), weighted by
    free bytes. Windows with no free space carry no weight.

    This is a windowed, localized complement of Gorman's unusable-free-space
    index: pure external fragmentation at S, with density factored out (which
    the old MWF conflated). At W >= capacity the weighted mean is exactly
    ``allocatable_count(snap, S) * S / total_free``.
    """
    _, free, usable = _window_profile(snap, window, size)
    mask = free > 0
    if not mask.any():
        return WindowDistribution(window=window, size=size, values=[], weights=[])
    frac: FloatArray = usable[mask] / free[mask]
    return _sorted_dist(window, size, frac, free[mask])


def _snapshot_dwell(events: Sequence[Event], policy: str, every: int) -> tuple[list[Snapshot], list[float]]:
    """Replay and pair each sampled snapshot with its dwell time on the event
    clock (ts delta to the next sample; the final sample gets weight 1)."""
    snaps = [snap for snap, _ in replay(list(events), policy, snapshot_every=every)]
    dwell = [float(b.ts - a.ts) for a, b in zip(snaps, snaps[1:])] + [1.0]
    return snaps, dwell


def pooled_occupancy(
    events: Sequence[Event], policy: str, window: int, *, every: int = 1
) -> WindowDistribution:
    """Whole-trace M2: occupancy CDF pooled over the replay, each snapshot
    weighted by its dwell time on the event clock -- byte-time, so transient
    states count in proportion to how long they persisted. ``cdf_at(u)`` then
    reads "this fraction of heap byte-time sat at occupancy <= u"."""
    snaps, dwell = _snapshot_dwell(events, policy, every)
    return WindowDistribution.pool(
        [occupancy_distribution(s, window) for s in snaps], dwell
    )


def pooled_usability(
    events: Sequence[Event], policy: str, window: int, size: int, *, every: int = 1
) -> WindowDistribution:
    """Whole-trace M3: usability CDF pooled over the replay (see pooled_occupancy)."""
    snaps, dwell = _snapshot_dwell(events, policy, every)
    return WindowDistribution.pool(
        [usability_distribution(s, window, size) for s in snaps], dwell
    )


class OccupancySpectrum(BaseModel):
    """Dispersion of the occupancy distribution as a function of window scale.

    With dyadic windows, occupancy at 2W averages its two children, so ``std``
    is non-increasing in W per snapshot, and the scale at which the spread
    collapses reads off the characteristic size of contiguous used/free
    clusters. Spread persisting at W means whole W-windows sit empty --
    reclaimable at that granularity (a compacted heap with slack keeps std at
    its maximum almost to the heap size); early collapse means free space is
    diffuse below that scale and needs relocation to recover.
    """

    model_config = ConfigDict(frozen=True)

    windows: list[int]
    mean: list[float]
    std: list[float]


def occupancy_spectrum(
    events: Sequence[Event], policy: str, windows: Sequence[int], *, every: int = 1
) -> OccupancySpectrum:
    """The fragmentation spectrum: whole-trace pooled occupancy mean/std per W."""
    snaps, dwell = _snapshot_dwell(events, policy, every)
    means: list[float] = []
    stds: list[float] = []
    for w in windows:
        pooled = WindowDistribution.pool(
            [occupancy_distribution(s, w) for s in snaps], dwell
        )
        means.append(pooled.mean)
        stds.append(pooled.std)
    return OccupancySpectrum(windows=list(windows), mean=means, std=stds)


def dyadic_windows(capacity: int, *, min_window: int = 256) -> list[int]:
    """Power-of-two window lengths from ``min_window`` up to >= capacity --
    the scales at which the occupancy spectrum's variance decay is exact."""
    if min_window <= 0:
        raise ValueError("min_window must be positive")
    windows = [min_window]
    while windows[-1] < capacity:
        windows.append(windows[-1] * 2)
    return windows


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


# ---------------------------------------------------------------------------
# torch-native-allocation.md metric family
# ---------------------------------------------------------------------------
#
# These reproduce the exact definitions in that doc for a caching/segment
# allocator. They need three quantities the plain Snapshot doesn't carry, so we
# read them from the Heap after replay:
#   reserved  = physical memory held (segments)         -> heap.peak_capacity
#   allocated = rounded/split live block sizes          -> sum of live footprints
#   active    = requested (user) live sizes             -> heap.live_bytes
# cached_free (free blocks inside reserved segments) and largest_free come from
# the snapshot's free runs. For a growable non-segment heap, reserved == capacity
# and cached_free == total_free (all free space sits inside the reserved region).

class DocMetrics(BaseModel):
    """The torch-native-allocation.md metric set for one heap state.

    Caveat on `memory_density` / `external_frag`: the simulated policies place
    segments contiguously from address 0, so the address span equals the reserved
    bytes and density is always 1.0 (external frag 0). Those two metrics only
    become non-trivial when segments are *scattered* across the virtual address
    space, which a real production allocator does but these reference models do
    not. The metrics that DO discriminate policies here are `hbm_utilization`
    (active/reserved), `internal_frag` (rounding waste), `cached_free`,
    `reserved`, and `segments`; density/ext-frag are reported for completeness
    and match a real allocator only when fed a real address-resolved trace."""

    model_config = ConfigDict(frozen=True)

    reserved: int          # physical memory held (segments)
    allocated: int         # rounded/split live block sizes
    active: int            # requested live sizes
    address_span: int      # first segment start -> last segment end
    segments: int          # number of reserved segments (0 if not tracked)
    cached_free: int       # free blocks cached inside reserved segments
    largest_free: int      # largest contiguous free chunk
    total_free: int        # reserved - active  (per the doc: over held memory)
    external_frag: int     # total_free - largest_free - cached_free
    internal_frag: int     # allocated - active
    memory_density: float  # reserved / span         (1.0 ideal)
    hbm_utilization: float # active / reserved
    non_reclaimable: int   # reserved - cached_free   (the honest cost)
    fragmentation_idx: float  # (ext + int) / total_free  (0 ideal)


def doc_metrics(heap: Heap, snap: Snapshot) -> DocMetrics:
    """Compute the doc's metric family from a replayed heap + its final snapshot.

    Works for any policy; for the caching allocator `reserved` is the peak
    segments held and `segments`/`allocated` come from its own counters, while
    for the reference fits `reserved == capacity` and every free byte is
    'cached' within that reserved region."""
    active = int(heap.live_bytes)
    # allocated = sum of live rounded footprints (== active unless a policy rounds)
    allocated = int(heap.allocated_bytes())
    caching = hasattr(heap, "segments") and bool(getattr(heap, "peak_reserved", 0))
    if caching:
        reserved = int(heap.peak_reserved)          # type: ignore[attr-defined]
        segments = int(heap.segments)               # type: ignore[attr-defined]
    else:
        reserved = int(heap.peak_capacity)
        segments = 0

    runs = snap.free_runs
    largest_free = max((r.length for r in runs), default=0)
    # cached free = free space sitting inside reserved memory.
    cached_free = int(sum(r.length for r in runs))
    # total free over held memory (doc: total_free = reserved - active).
    total_free = max(0, reserved - active)
    external_frag = max(0, total_free - largest_free - cached_free)
    internal_frag = max(0, allocated - active)
    span = _address_span(heap)
    density = reserved / span if span else 1.0
    utilization = active / reserved if reserved else 0.0
    non_reclaimable = max(0, reserved - cached_free)
    frag_idx = (external_frag + internal_frag) / total_free if total_free else 0.0

    return DocMetrics(
        reserved=reserved, allocated=allocated, active=active,
        address_span=span, segments=segments, cached_free=cached_free,
        largest_free=largest_free, total_free=total_free,
        external_frag=external_frag, internal_frag=internal_frag,
        memory_density=round(density, 4), hbm_utilization=round(utilization, 4),
        non_reclaimable=non_reclaimable, fragmentation_idx=round(frag_idx, 4),
    )


def _address_span(heap: Heap) -> int:
    """First-segment-start to last-segment-end. For these models allocation
    starts at 0 and capacity is the high-water end, so span == reserved
    capacity; kept as its own function to match the doc's wording and allow a
    non-zero base later."""
    return int(heap.peak_capacity)


def doc_metrics_at_peak(events: "Sequence[Event]", policy: str = "caching") -> DocMetrics:
    """Replay `events` under `policy` and return the doc metrics at the moment of
    PEAK reserved memory -- which is what the doc's "at peak (step0_after_fwd)"
    numbers mean. Snapshots active/cached_free/allocated consistently at that
    same instant (not end-of-trace, where most tensors have been freed)."""
    from .heap import replay  # local import avoids a cycle at module load
    best_reserved = -1
    best: DocMetrics | None = None
    for snap, heap in replay(list(events), policy):
        reserved = getattr(heap, "peak_reserved", 0) or heap.capacity
        if reserved >= best_reserved:
            best_reserved = reserved
            best = doc_metrics(heap, snap)
    assert best is not None, "empty event stream"
    return best
