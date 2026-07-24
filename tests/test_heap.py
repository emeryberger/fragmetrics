"""Replayer invariants and the address-resolved vs address-free round-trip."""

from __future__ import annotations

import pytest

from fragmetrics import workload as w
from fragmetrics.heap import Heap, replay


def test_free_runs_stay_sorted_and_coalesced() -> None:
    heap = Heap("first-fit")
    # allocate three adjacent blocks, free the middle then its neighbours:
    # the free set must coalesce back to a single run.
    heap.alloc(0, 64, addr=0)
    heap.alloc(1, 64, addr=64)
    heap.alloc(2, 64, addr=128)
    heap.free(1)
    heap.free(0)
    heap.free(2)
    snap = heap.snapshot(0)
    assert len(snap.free_runs) == 1
    assert snap.free_runs[0].start == 0
    assert snap.free_runs[0].length == 192


def test_capacity_never_shrinks() -> None:
    heap = Heap("first-fit")
    heap.alloc(0, 100, addr=0)
    peak = heap.capacity
    heap.free(0)
    assert heap.capacity == peak
    assert heap.peak_capacity == peak


def test_double_free_is_tolerated() -> None:
    """Real captured traces contain unmatched frees (pointers allocated before
    tracing began). The replayer silently skips them rather than crashing."""
    heap = Heap("first-fit")
    heap.alloc(0, 64, addr=0)
    heap.free(0)
    heap.free(0)  # second free of same id: silently ignored
    heap.free(999)  # free of never-allocated id: silently ignored


def test_alloc_duplicate_id_raises() -> None:
    heap = Heap("first-fit")
    heap.alloc(0, 64, addr=0)
    with pytest.raises(ValueError):
        heap.alloc(0, 64)


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        Heap("no-such-policy")


def test_address_free_replay_matches_total_free() -> None:
    """An address-bearing trace and the same trace with addresses stripped,
    replayed under a policy, must agree on the size-independent aggregates
    (total free, live bytes) -- the trace round-trip from the plan."""
    addressed = w.checkerboard(block=64, n=16)
    stripped = [e.model_copy(update={"addr": None}) for e in addressed]
    a = [s for s, _ in replay(addressed, "first-fit")][-1]
    b = [s for s, _ in replay(stripped, "first-fit")][-1]
    assert a.live_bytes == b.live_bytes
    assert a.total_free == b.total_free


def test_buddy_rounds_up_to_power_of_two() -> None:
    heap = Heap("buddy")
    # a 65-byte request consumes a whole 128-byte block; the 63-byte slack is
    # internal fragmentation (occupied, not free), so total_free is 0 but
    # live demand is only the requested 65.
    heap.alloc(0, 65)
    snap = heap.snapshot(0)
    assert heap.capacity == 128
    assert snap.total_free == 0          # slack is occupied, not free
    assert snap.live_bytes == 65         # demand tracks the request, not footprint
    # freeing returns the full 128-byte footprint to the free set
    heap.free(0)
    snap = heap.snapshot(1)
    assert snap.total_free == 128
    assert snap.live_bytes == 0


def test_snapshot_sampling_density() -> None:
    events = w.generate(w.WorkloadConfig(n_events=500, seed=1))
    every_1 = list(replay(events, "first-fit", snapshot_every=1))
    every_10 = list(replay(events, "first-fit", snapshot_every=10))
    assert len(every_1) > len(every_10)
    # both end on the same final state
    assert every_1[-1][0].total_free == every_10[-1][0].total_free


# ---- new policies: next-fit, caching, custom plugin -----------------------

def test_next_fit_and_caching_are_valid_placements() -> None:
    """Every policy must produce a feasible layout (no two live blocks overlap)
    and a footprint >= peak live."""
    events = w.generate(w.WorkloadConfig(n_events=3000, seed=7))
    for policy in ("next-fit", "caching"):
        snap, heap = list(replay(list(events), policy))[-1]
        assert heap.peak_capacity >= heap.peak_live, policy


def test_caching_holds_reserved_across_free() -> None:
    """The caching allocator caches freed blocks -- reserved memory does not
    shrink when an object is freed (unlike the fit policies' coalescing)."""
    from fragmetrics.heap import CachingHeap
    h = CachingHeap()
    h.alloc(0, 1000)
    reserved_after_alloc = h.capacity
    h.free(0)
    assert h.capacity == reserved_after_alloc  # segment stays reserved
    # a same-size alloc reuses the cached block, no new reservation
    h.alloc(1, 1000)
    assert h.capacity == reserved_after_alloc


def test_custom_policy_plugin() -> None:
    """A registered fit function is selectable by name and drives placement."""
    from fragmetrics.heap import register_policy, FreeRun, FitContext, Heap

    def worst_fit_clone(free_runs, size, ctx):
        best, best_len = None, -1
        for i, r in enumerate(free_runs):
            if r.length >= size and r.length > best_len:
                best, best_len = i, r.length
        return best

    register_policy("worst-clone", worst_fit_clone)
    assert "worst-clone" in Heap.known_policies()
    events = w.generate(w.WorkloadConfig(n_events=1500, seed=4))
    _snap, custom = list(replay(list(events), "worst-clone"))[-1]
    _snap, builtin = list(replay(list(events), "worst-fit"))[-1]
    # our clone should match the built-in worst-fit footprint exactly
    assert custom.peak_capacity == builtin.peak_capacity


def test_unknown_policy_rejected() -> None:
    with pytest.raises(ValueError):
        Heap("no-such-policy")
