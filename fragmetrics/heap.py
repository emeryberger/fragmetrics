"""Heap model: a placement-policy replayer over a contiguous address space.

The heap is modelled as a single contiguous region ``[0, capacity)`` partitioned
into used and free runs. We track free space as a list of ``(start, length)``
runs kept sorted by ``start`` and always coalesced -- this **free-run multiset**
is the one shared structure every metric in ``metrics.py`` reads. Build it once
per snapshot; derive all metrics from it.

Replay semantics
----------------
- ``alloc(size)`` picks a free run per the policy, carves ``size`` bytes from it,
  records the placement, and returns the address. If no run fits, the heap
  *grows* (``capacity`` extends) -- mirroring a real allocator asking the OS for
  more memory. The growth is recorded so we can compute peak-heap / blowup.
- ``free(id)`` returns the object's bytes to the free set and coalesces with
  adjacent free runs.
- If the trace already carries ``addr`` (address-resolved trace), the policy is
  bypassed and the exact placement is honoured -- so an address-bearing trace and
  an address-free trace + policy can be compared on identical workloads.

Policies
--------
first-fit, best-fit, worst-fit, segregated-fit (size-class bins), buddy
(power-of-two), and an ``oracle`` (compaction: free space is always one run, the
ideal that normalizes blowup and the ``N*(S)`` term in F(S)).

Policies are deliberately simple, allocator-*agnostic* reference behaviours --
the point is to exercise the metrics across qualitatively different layout
regimes, not to reproduce any production allocator byte-for-byte.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from .trace import Event, Op

# ---------------------------------------------------------------------------
# Pluggable policies
# ---------------------------------------------------------------------------
#
# A *fit function* is the core placement decision, factored out so new policies
# are drop-in without touching the Heap internals. Given the current coalesced
# free-run list and a request size, it returns the INDEX of the run to carve
# from, or None to force the heap to grow. This is enough to express every
# classic online fit (first/best/worst/next) and any custom heuristic.
#
#   def my_fit(free_runs: list[FreeRun], size: int, ctx: FitContext) -> int | None: ...
#
# Register one with `register_policy("name", my_fit)` and it becomes selectable
# by name everywhere (CLI, replay, Heap). `ctx` carries small mutable per-heap
# state (e.g. next-fit's rolling cursor) so fit functions can stay pure-ish.
FitFn = Callable[["list[FreeRun]", int, "FitContext"], "int | None"]

_CUSTOM_POLICIES: dict[str, FitFn] = {}


@dataclass
class FitContext:
    """Scratch state a fit function may read/update across calls on one heap."""
    cursor: int = 0  # for next-fit: index to resume scanning from


def register_policy(name: str, fit: FitFn) -> None:
    """Register a custom fit function under `name` (usable as a --policy)."""
    _CUSTOM_POLICIES[name] = fit


def registered_policies() -> tuple[str, ...]:
    return tuple(_CUSTOM_POLICIES)


@dataclass(frozen=True, slots=True)
class FreeRun:
    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class Snapshot:
    """Immutable-ish view of heap state at one point in the replay.

    ``free_runs`` is sorted by start and fully coalesced. Everything the metrics
    need is derivable from (free_runs, capacity, live_bytes).
    """

    ts: int
    capacity: int          # current heap size (high-water grows, never shrinks here)
    live_bytes: int        # sum of currently-allocated object sizes
    free_runs: list[FreeRun]

    @property
    def total_free(self) -> int:
        return sum(r.length for r in self.free_runs)

    @property
    def max_free(self) -> int:
        return max((r.length for r in self.free_runs), default=0)

    @property
    def used_bytes(self) -> int:
        return self.capacity - self.total_free


class Heap:
    """A growable contiguous heap with a pluggable placement policy."""

    #: built-in policy names (custom ones registered via register_policy add to these)
    POLICIES = (
        "first-fit", "best-fit", "worst-fit", "next-fit",
        "segregated-fit", "buddy", "caching", "oracle",
    )

    @classmethod
    def known_policies(cls) -> tuple[str, ...]:
        return cls.POLICIES + registered_policies()

    def __init__(self, policy: str = "first-fit", *, initial_capacity: int = 0):
        if policy not in self.known_policies():
            raise ValueError(
                f"unknown policy {policy!r}; choose from {self.known_policies()}"
            )
        self.policy = policy
        self._fit_ctx = FitContext()
        # A custom fit function shadows the built-in dispatch when registered.
        self._custom_fit = _CUSTOM_POLICIES.get(policy)
        self.capacity = initial_capacity
        # free runs, sorted by start, always coalesced
        self._free: list[FreeRun] = [FreeRun(0, initial_capacity)] if initial_capacity else []
        # id -> (start, footprint, demand). ``footprint`` is the bytes actually
        # occupied on the heap; ``demand`` is what the caller requested. They
        # differ when a policy over-allocates (e.g. buddy rounding -> internal
        # fragmentation). ``live_bytes`` sums demand; ``footprint`` drives layout.
        self._live: dict[int, tuple[int, int, int]] = {}
        self.live_bytes = 0
        self.peak_capacity = initial_capacity
        self.peak_live = 0

    # ---- public replay API ------------------------------------------------

    def apply(self, ev: Event) -> None:
        if ev.op is Op.ALLOC:
            assert ev.size is not None  # guaranteed by Event validation
            self.alloc(ev.id, ev.size, addr=ev.addr)
        elif ev.op is Op.FREE:
            self.free(ev.id)

    def alloc(self, obj_id: int, size: int, *, addr: int | None = None) -> int:
        if obj_id in self._live:
            raise ValueError(f"alloc of already-live id {obj_id}")
        if self.policy == "oracle" and addr is None:
            # perfect compaction: live data is always packed, free space is one
            # run at the top, capacity == peak live. This is the ideal baseline
            # that normalizes blowup, so it must never fragment.
            start = self.live_bytes
            self.live_bytes += size
            self.peak_live = max(self.peak_live, self.live_bytes)
            self._extend_capacity(self.live_bytes)
            self._recompact()
            self._live[obj_id] = (start, size, size)
            return start
        if addr is not None:
            self._place_exact(addr, size)
            start, footprint = addr, size
        elif self.policy == "buddy":
            start, footprint = self._alloc_buddy(size)
        else:
            start, footprint = self._alloc_fit(size), size
        self._live[obj_id] = (start, footprint, size)
        self.live_bytes += size
        self.peak_live = max(self.peak_live, self.live_bytes)
        return start

    def free(self, obj_id: int) -> None:
        entry = self._live.pop(obj_id, None)
        if entry is None:
            # Unmatched free: pointer allocated before tracing began (runtime init).
            # Silently skip rather than crash — common in real captured traces.
            return
        start, footprint, demand = entry
        self.live_bytes -= demand
        if self.policy == "oracle":
            self._recompact()  # compaction closes the hole instantly
        else:
            self._insert_free(FreeRun(start, footprint))

    def _recompact(self) -> None:
        """Oracle invariant: a single free run [live_bytes, capacity)."""
        slack = self.capacity - self.live_bytes
        self._free = [FreeRun(self.live_bytes, slack)] if slack > 0 else []

    def snapshot(self, ts: int) -> Snapshot:
        return Snapshot(
            ts=ts,
            capacity=self.capacity,
            live_bytes=self.live_bytes,
            free_runs=list(self._free),
        )

    def allocated_bytes(self) -> int:
        """Sum of live rounded footprints (== live_bytes unless a policy rounds).
        The gap allocated_bytes - live_bytes is internal fragmentation."""
        return sum(footprint for _start, footprint, _demand in self._live.values())

    # ---- placement strategies --------------------------------------------

    def _alloc_fit(self, size: int) -> int:
        """first/best/worst/next fit -- or a registered custom fit -- over the
        coalesced free-run list. Returns the carved start address."""
        if self._custom_fit is not None:
            chosen = self._custom_fit(self._free, size, self._fit_ctx)
            chosen = -1 if chosen is None else chosen
        elif self.policy == "first-fit":
            chosen = -1
            for i, r in enumerate(self._free):
                if r.length >= size:
                    chosen = i
                    break
        elif self.policy == "next-fit":
            # Resume scanning from the last-used run (rolling cursor), wrapping
            # around once. Approximates a real bump-cursor allocator.
            chosen = -1
            n = len(self._free)
            for off in range(n):
                i = (self._fit_ctx.cursor + off) % n
                if self._free[i].length >= size:
                    chosen = i
                    break
        else:
            best_idx, best_key = -1, None
            for i, r in enumerate(self._free):
                if r.length < size:
                    continue
                # best-fit minimizes leftover; worst-fit maximizes it
                key = r.length if self.policy == "best-fit" else -r.length
                if best_key is None or key < best_key:
                    best_idx, best_key = i, key
            chosen = best_idx
        if chosen < 0:
            return self._grow_and_place(size)
        start = self._carve(chosen, size)
        if self.policy == "next-fit":
            # Keep the cursor valid after the carve (run may have been removed).
            self._fit_ctx.cursor = chosen % max(1, len(self._free))
        return start

    def _alloc_buddy(self, size: int) -> tuple[int, int]:
        """Buddy-style: round the request up to a power of two and allocate that
        whole block. The slack (rounded - size) is wasted as *internal*
        fragmentation -- it is not returned to the free set -- which is exactly
        the cost buddy allocators trade for fast coalescing. Returns
        ``(start, footprint)`` where footprint is the rounded block size.
        """
        rounded = 1
        while rounded < size:
            rounded <<= 1
        for i, r in enumerate(self._free):
            if r.length >= rounded:
                return self._carve(i, rounded), rounded
        return self._grow_and_place(rounded), rounded

    def _place_exact(self, addr: int, size: int) -> None:
        """Honour an address from an address-resolved trace."""
        end = addr + size
        if end > self.capacity:
            # the grown region [old_capacity, end) becomes newly-free space, then
            # gets carved below like any other free run
            self._insert_free(FreeRun(self.capacity, end - self.capacity))
            self._extend_capacity(end)
        # find the free run containing [addr, end) and carve it
        for i, r in enumerate(self._free):
            if r.start <= addr and end <= r.end:
                self._carve_within(i, addr, size)
                return
        raise ValueError(f"exact placement [{addr},{end}) overlaps a used region")

    # ---- low-level free-run maintenance ----------------------------------

    def _carve(self, idx: int, size: int) -> int:
        """Allocate ``size`` from the front of free run ``idx``."""
        r = self._free[idx]
        start = r.start
        if r.length == size:
            self._free.pop(idx)
        else:
            self._free[idx] = FreeRun(r.start + size, r.length - size)
        return start

    def _carve_within(self, idx: int, addr: int, size: int) -> None:
        """Allocate [addr, addr+size) from inside free run ``idx``, splitting it."""
        r = self._free.pop(idx)
        left = addr - r.start
        right = r.end - (addr + size)
        # reinsert remaining fragments (keep sorted; idx is the right slot for left)
        if right > 0:
            self._free.insert(idx, FreeRun(addr + size, right))
        if left > 0:
            self._free.insert(idx, FreeRun(r.start, left))

    def _grow_and_place(self, size: int) -> int:
        """No run fits: extend capacity and place at the new top."""
        # If the top of the heap is free, we only need to grow by the shortfall.
        tail_free = 0
        if self._free and self._free[-1].end == self.capacity:
            tail_free = self._free[-1].length
        start = self.capacity - tail_free
        self._extend_capacity(start + size)
        # the region [start, start+size) is now used; drop any tail free run
        if tail_free:
            self._free.pop()
        return start

    def _extend_capacity(self, new_cap: int) -> None:
        if new_cap > self.capacity:
            self.capacity = new_cap
            self.peak_capacity = max(self.peak_capacity, new_cap)

    def _insert_free(self, run: FreeRun) -> None:
        """Insert a free run and coalesce with neighbours (keeps list sorted)."""
        starts = [r.start for r in self._free]
        i = bisect_left(starts, run.start)
        start, length = run.start, run.length
        # coalesce with left neighbour
        if i > 0 and self._free[i - 1].end == start:
            start = self._free[i - 1].start
            length += self._free[i - 1].length
            self._free.pop(i - 1)
            i -= 1
        # coalesce with right neighbour
        if i < len(self._free) and self._free[i].start == start + length:
            length += self._free[i].length
            self._free.pop(i)
        self._free.insert(i, FreeRun(start, length))


class CachingHeap(Heap):
    """A caching / segment-bucketing allocator, modelling the PyTorch-CUDA style
    (and the block-segment scheme in torch-native-allocation.md).

    Unlike the reference fits, this heap distinguishes *reserved* memory (whole
    segments obtained from the "driver" and cached, never returned) from *live*
    blocks inside them. Allocation:

      1. Round the request up (min size + power-of-two-ish rounding) -> `demand`.
      2. Route to the small pool (<= boundary) or large pool.
      3. Find a cached free block that fits (best-fit within the pool); split it
         if the remainder is worth keeping. If none, reserve a NEW segment (a
         fixed segment size for the small pool; rounded request for the large
         pool) from the growing address space and carve from it.
      4. free() returns the block to the pool's cache (coalescing with adjacent
         free blocks in the same segment) -- it does NOT shrink reserved memory.

    The heap's `peak_capacity` therefore tracks *reserved* bytes (segments held),
    which is exactly the "reserved" side of the doc's `density = reserved / span`.
    Parameters default to the doc's derived values but are tunable.
    """

    def __init__(
        self,
        *,
        small_boundary: int = 256 * 1024,      # small/large pool split
        small_segment: int = 2 * 1024 * 1024,  # one segment covers the small pool
        large_roundup: int = 2 * 1024 * 1024,  # large allocs rounded to this
        min_block: int = 256,                  # smallest rounded size
        pow2_divisions: int = 2,               # roundup granularity within a power of two
        initial_capacity: int = 0,
    ):
        super().__init__(policy="first-fit", initial_capacity=initial_capacity)
        self.policy = "caching"
        self.small_boundary = small_boundary
        self.small_segment = small_segment
        self.large_roundup = large_roundup
        self.min_block = min_block
        self.pow2_divisions = max(1, pow2_divisions)
        # cached free blocks per pool: list of FreeRun (address-space runs inside
        # reserved segments). Reuses Heap's free-run machinery conceptually but
        # kept per-pool so small/large don't fragment each other.
        self._pool_free: dict[str, list[FreeRun]] = {"small": [], "large": []}
        # doc metrics: count of reserved segments, and peak of (reserved,
        # allocated-rounded, cached-free) captured at each state so the report
        # reflects the high-water mark, not just end-of-run.
        self.segments = 0
        self.peak_reserved = 0
        self.peak_allocated = 0  # peak sum of rounded block footprints (live)
        self._allocated = 0      # current sum of rounded footprints (live)

    def _round(self, size: int) -> int:
        """Round a request up to this allocator's block granularity."""
        if size <= self.min_block:
            return self.min_block
        if size <= self.small_boundary:
            # small pool: round up to a multiple of min_block
            return ((size + self.min_block - 1) // self.min_block) * self.min_block
        # large pool: round up within a power of two, `pow2_divisions` steps
        p = 1
        while p < size:
            p <<= 1
        lo = p >> 1
        step = max(self.large_roundup, (p - lo) // self.pow2_divisions)
        return min(p, lo + ((size - lo + step - 1) // step) * step)

    def alloc(self, obj_id: int, size: int, *, addr: int | None = None) -> int:
        # `addr` (an address-resolved trace's placement) is intentionally ignored:
        # a caching allocator always makes its own placement decision, which is
        # the whole point of simulating it. We model its choices, not the trace's.
        del addr
        if obj_id in self._live:
            raise ValueError(f"alloc of already-live id {obj_id}")
        demand = self._round(size)
        pool = "small" if demand <= self.small_boundary else "large"
        runs = self._pool_free[pool]
        # best-fit within the pool's cached free blocks
        best = -1
        for i, r in enumerate(runs):
            if r.length >= demand and (best < 0 or r.length < runs[best].length):
                best = i
        if best >= 0:
            r = runs.pop(best)
            start = r.start
            if r.length > demand:  # split; keep remainder cached
                runs.append(FreeRun(start + demand, r.length - demand))
        else:
            # reserve a new segment from the top of the address space
            seg = self.small_segment if pool == "small" else max(demand, self.large_roundup)
            seg = max(seg, demand)
            start = self.capacity
            self._extend_capacity(self.capacity + seg)
            self.segments += 1
            if seg > demand:  # rest of the segment is cached free
                runs.append(FreeRun(start + demand, seg - demand))
        # record: footprint == demand (rounded), so internal frag = demand-size
        self._live[obj_id] = (start, demand, size)
        self.live_bytes += size
        self._allocated += demand
        self.peak_live = max(self.peak_live, self.live_bytes)
        self.peak_reserved = max(self.peak_reserved, self.capacity)
        self.peak_allocated = max(self.peak_allocated, self._allocated)
        return start

    def free(self, obj_id: int) -> None:
        entry = self._live.pop(obj_id, None)
        if entry is None:
            return
        start, footprint, demand_sz = entry
        self.live_bytes -= demand_sz
        self._allocated -= footprint
        pool = "small" if footprint <= self.small_boundary else "large"
        # return to cache and coalesce with adjacent cached blocks in the pool
        runs = self._pool_free[pool]
        runs.append(FreeRun(start, footprint))
        runs.sort(key=lambda r: r.start)
        merged: list[FreeRun] = []
        for r in runs:
            if merged and merged[-1].end == r.start:
                merged[-1] = FreeRun(merged[-1].start, merged[-1].length + r.length)
            else:
                merged.append(r)
        self._pool_free[pool] = merged

    def snapshot(self, ts: int) -> Snapshot:
        free = sorted(
            self._pool_free["small"] + self._pool_free["large"],
            key=lambda r: r.start,
        )
        return Snapshot(ts=ts, capacity=self.capacity, live_bytes=self.live_bytes,
                        free_runs=free)


def make_heap(policy: str, **kwargs: int) -> Heap:
    """Construct the right Heap subclass for `policy`."""
    if policy == "caching":
        return CachingHeap(**kwargs)
    return Heap(policy, **kwargs)


def replay(
    events: Iterable[Event],
    policy: str = "first-fit",
    *,
    snapshot_every: int = 1,
) -> Iterator[tuple[Snapshot, Heap]]:
    """Replay a trace, yielding ``(Snapshot, Heap)`` periodically.

    ``snapshot_every`` controls sampling: 1 = every event (full time series),
    larger = coarser/cheaper. The final state is always emitted.
    """
    heap = make_heap(policy)
    last_ts: int | None = None
    for n, ev in enumerate(events):
        heap.apply(ev)
        last_ts = ev.ts
        if snapshot_every and (n % snapshot_every == 0):
            yield heap.snapshot(ev.ts), heap
    if last_ts is not None:
        yield heap.snapshot(last_ts), heap
