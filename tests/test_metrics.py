"""Golden tests pinning down metric behaviour on canonical layouts.

The three fixtures encode the cases the legacy occupancy metric cannot tell
apart, plus the capacity-vs-fragmentation distinction:

* coalesced            -> no external fragmentation
* checkerboard         -> maximal external fragmentation at the legacy 0.5 occupancy
* capacity-exhaustion  -> a capacity limit, NOT fragmentation
"""

from __future__ import annotations

import math

import pytest

from fragmetrics import metrics as m
from fragmetrics import workload as w
from fragmetrics.heap import Heap, Snapshot, replay
from fragmetrics.trace import Event


def _final(events: list[Event], policy: str = "first-fit") -> Snapshot:
    snaps = [snap for snap, _ in replay(events, policy)]
    return snaps[-1]


# --- coalesced: the no-fragmentation baseline -----------------------------


def test_coalesced_has_no_external_fragmentation() -> None:
    snap = _final(w.coalesced(block=64, n=16))
    # half the heap is free, all in one run
    assert snap.total_free == 512
    assert snap.max_free == 512
    assert len(snap.free_runs) == 1
    # F(S) == 0 for every size that fits at all
    for size in (16, 32, 64, 128, 256, 512):
        assert m.fragmentation_at_size(snap, size) == 0.0
    # a request that fits returns the "ok" sentinel
    assert m.fragmentation_index(snap, 256) == -1.0


# --- checkerboard: the canonical external-fragmentation case --------------


def test_checkerboard_legacy_occupancy_is_blind() -> None:
    coal = _final(w.coalesced(block=64, n=16))
    check = _final(w.checkerboard(block=64, n=16))
    occ_coal = coal.live_bytes / coal.capacity
    occ_check = check.live_bytes / check.capacity
    # identical occupancy...
    assert occ_coal == occ_check == 0.5
    # ...but the new metric separates them sharply
    assert m.fragmentation_at_size(coal, 128) == 0.0
    assert m.fragmentation_at_size(check, 128) == 1.0


def test_checkerboard_cannot_hold_double_block() -> None:
    snap = _final(w.checkerboard(block=64, n=16))
    assert snap.total_free == 512          # plenty of free bytes
    assert snap.max_free == 64             # but no run larger than one block
    assert m.allocatable_count(snap, 64) == 8   # singles fit
    assert m.allocatable_count(snap, 128) == 0  # doubles do not
    assert m.fragmentation_at_size(snap, 128) == 1.0


def test_checkerboard_fragmentation_index_signals_fragmentation() -> None:
    snap = _final(w.checkerboard(block=64, n=16))
    # request larger than max_free but smaller than total_free -> genuine frag
    fidx = m.fragmentation_index(snap, 128)
    assert 0.5 < fidx < 1.0
    # checkerboard index is near-maximal
    assert m.checkerboard_index(snap) > 0.9


# --- capacity exhaustion: NOT fragmentation -------------------------------


def test_capacity_exhaustion_is_not_fragmentation() -> None:
    snap = _final(w.capacity_exhaustion(block=64, n=16))
    # only one small free run; a large request fails for capacity, not layout
    assert len(snap.free_runs) == 1
    assert snap.total_free == 64
    # request exceeds total free -> index -> 0 (capacity), not -> 1 (fragmentation)
    assert m.fragmentation_index(snap, 128) == 0.0
    # and F(S) is 0 because even the ideal heap cannot hold it
    assert m.fragmentation_at_size(snap, 128) == 0.0


# --- M2/M3: windowing -----------------------------------------------------


def test_min_window_fill_bounds() -> None:
    snap = _final(w.checkerboard(block=64, n=16))
    mwf = m.min_window_fill(snap, window=128, size=128)
    # no 128-window can hold a 128 object (every other 64 is used) -> 0 fill
    assert mwf == 0.0
    # percentile with pct=0 recovers the min
    assert m.window_fill_percentile(snap, 128, 128, 0.0) == mwf
    # higher percentile is >= the min
    assert m.window_fill_percentile(snap, 128, 128, 95.0) >= mwf


def test_window_percentile_validates_range() -> None:
    snap = _final(w.coalesced())
    with pytest.raises(ValueError):
        m.window_fill_percentile(snap, 128, 64, pct=150.0)


# --- M5: blowup orders policies as the literature predicts ----------------


def test_blowup_oracle_is_unity() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=3000, seed=7))]
    oracle = m.replay_blowup(events, "oracle")
    assert oracle.blowup == pytest.approx(1.0)
    assert oracle.peak_heap == oracle.peak_live
    assert oracle.ext_growth_rate == 0.0


def test_blowup_ordering_best_fit_beats_worst_fit() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=3000, seed=7))]
    best = m.replay_blowup(events, "best-fit").blowup
    worst = m.replay_blowup(events, "worst-fit").blowup
    oracle = m.replay_blowup(events, "oracle").blowup
    assert oracle <= best < worst


# --- M6: cheap scalars ----------------------------------------------------


def test_gini_zero_for_equal_runs() -> None:
    snap = _final(w.checkerboard(block=64, n=16))  # all free runs equal size
    assert m.free_gini(snap) == pytest.approx(0.0, abs=1e-9)


def test_entropy_zero_for_single_run() -> None:
    snap = _final(w.coalesced())  # one free run
    assert m.free_entropy(snap) == 0.0


def test_entropy_maximal_for_equal_runs() -> None:
    snap = _final(w.checkerboard(block=64, n=16))  # 8 equal runs -> log2(8)=3 bits
    assert m.free_entropy(snap) == pytest.approx(3.0)


# --- curve summary --------------------------------------------------------


def test_fragmentation_curve_auc_monotonic_in_layout() -> None:
    coal = m.fragmentation_curve(_final(w.coalesced()), [16, 32, 64, 128, 256])
    check = m.fragmentation_curve(_final(w.checkerboard()), [16, 32, 64, 128, 256])
    assert coal.auc == 0.0
    assert check.auc > coal.auc


# --- input validation -----------------------------------------------------


def test_allocatable_count_rejects_nonpositive_size() -> None:
    snap = _final(w.coalesced())
    with pytest.raises(ValueError):
        m.allocatable_count(snap, 0)


def test_empty_heap_is_safe() -> None:
    heap = Heap("first-fit")
    snap = heap.snapshot(0)
    assert m.allocatable_count(snap, 64) == 0
    assert m.fragmentation_at_size(snap, 64) == 0.0
    assert m.checkerboard_index(snap) == 0.0
    assert m.free_gini(snap) == 0.0
    assert m.free_entropy(snap) == 0.0
    assert not math.isnan(m.summarize(snap).occupancy)
