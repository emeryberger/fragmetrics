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


# --- M2: occupancy CDF ----------------------------------------------------


def test_occupancy_mean_is_global_occupancy_at_every_scale() -> None:
    snap = _final(w.checkerboard(block=64, n=16))
    global_occ = snap.used_bytes / snap.capacity
    for window in (64, 128, 192, 256, 512, 1024, 4096):
        assert m.occupancy_distribution(snap, window).mean == pytest.approx(global_occ)


def test_occupancy_cdf_separates_compacted_from_checkerboard() -> None:
    # both heaps are half occupied; the W=128 occupancy CDF tells them apart
    coal = m.occupancy_distribution(_final(w.coalesced(block=64, n=16)), 128)
    check = m.occupancy_distribution(_final(w.checkerboard(block=64, n=16)), 128)
    # checkerboard: every window is exactly half occupied -> a point mass
    assert check.std == pytest.approx(0.0)
    assert check.quantile(0.0) == pytest.approx(0.5)
    # compacted: windows are all-full or all-empty -> maximal spread, and the
    # CDF's mass at zero occupancy is exactly the reclaimable half of the heap
    assert coal.std == pytest.approx(0.5)
    assert coal.cdf_at(0.0) == pytest.approx(0.5)


def test_occupancy_spread_is_nonincreasing_in_scale() -> None:
    # dyadic coarsening averages child windows -> variance can only shrink
    snap = _final(w.coalesced(block=64, n=16))
    stds = [m.occupancy_distribution(snap, wl).std for wl in (64, 128, 256, 512, 1024)]
    assert all(a >= b - 1e-12 for a, b in zip(stds, stds[1:]))
    # a single window is a point mass at global occupancy
    assert stds[-1] == pytest.approx(0.0)


# --- M3: usability CDF ----------------------------------------------------


def test_usability_separates_shredded_from_coalesced_free_space() -> None:
    coal = _final(w.coalesced(block=64, n=16))
    check = _final(w.checkerboard(block=64, n=16))
    # checkerboard free space at S=128: every free byte unusable
    assert m.usability_distribution(check, 128, 128).mean == pytest.approx(0.0)
    # coalesced free space (one 512 run) at S=128: every free byte usable
    assert m.usability_distribution(coal, 128, 128).mean == pytest.approx(1.0)
    # ...and unlike the old MWF, dense windows do not drag the tail down:
    # occupancy and usability answer separate questions
    assert m.usability_distribution(coal, 128, 128).quantile(0.0) == pytest.approx(1.0)


def test_usability_mean_matches_ufsi_complement_at_full_window() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=3000, seed=7))]
    snap = _final(events)
    assert snap.total_free > 0
    for size in (48, 96, 256):
        dist = m.usability_distribution(snap, snap.capacity, size)
        expected = m.allocatable_count(snap, size) * size / snap.total_free
        assert dist.mean == pytest.approx(expected)


def test_usability_weights_are_free_bytes() -> None:
    snap = _final(w.capacity_exhaustion(block=64, n=16))
    dist = m.usability_distribution(snap, 64, 32)
    assert dist.total_weight == pytest.approx(snap.total_free)


# --- M1b: usable-free curve ------------------------------------------------


def test_usable_free_fractions_on_canonical_layouts() -> None:
    check = _final(w.checkerboard(block=64, n=16))  # 8 free runs of 64
    assert m.packed_usable_fraction(check, 64) == pytest.approx(1.0)
    assert m.packed_usable_fraction(check, 128) == pytest.approx(0.0)
    assert m.contiguous_free_fraction(check, 64) == pytest.approx(1.0)
    assert m.contiguous_free_fraction(check, 65) == pytest.approx(0.0)
    coal = _final(w.coalesced(block=64, n=16))  # one free run of 512
    assert m.contiguous_free_fraction(coal, 512) == pytest.approx(1.0)
    # packing loss: one 300-byte object fits in the 512 run, stranding 212 bytes
    assert m.packed_usable_fraction(coal, 300) == pytest.approx(300 / 512)


def test_unusable_orientation_and_auc() -> None:
    sizes = [16, 32, 64, 128, 256, 512]
    coal = m.usable_free_curve(_final(w.coalesced(block=64, n=16)), sizes)
    check = m.usable_free_curve(_final(w.checkerboard(block=64, n=16)), sizes)
    for curve in (coal, check):
        assert curve.unusable == pytest.approx([1.0 - p for p in curve.packed])
        assert 0.0 <= curve.unusable_auc <= 1.0
    # checkerboard wastes more free space than compacted at every probed size
    assert check.unusable_auc > coal.unusable_auc


def test_pooled_usable_free_curve_bounds() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=500, seed=3))]
    sizes = [32, 128, 512, 2048]
    curve = m.pooled_usable_free_curve(events, "first-fit", sizes)
    assert curve.sizes == sizes
    for p, c in zip(curve.packed, curve.contiguous):
        assert 0.0 <= p <= c + 1e-12
        assert c <= 1.0 + 1e-12
    # pooling the oracle: free space is always one run, so contiguous == 1
    # wherever any free byte-time exists at all sizes below the slack
    oracle = m.pooled_usable_free_curve(events, "oracle", [1])
    assert oracle.contiguous[0] == pytest.approx(1.0)


def test_packed_bounded_by_contiguous_and_matches_m3_mean() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=3000, seed=7))]
    snap = _final(events)
    for size in (48, 96, 256, 1024):
        packed = m.packed_usable_fraction(snap, size)
        assert packed <= m.contiguous_free_fraction(snap, size) + 1e-12
        # identity: packed is the byte-weighted mean of M3 at W >= capacity
        assert packed == pytest.approx(m.usability_distribution(snap, snap.capacity, size).mean)
    curve = m.usable_free_curve(snap, [64, 128, 256])
    assert len(curve.packed) == len(curve.contiguous) == 3
    # contiguous is a survival function: monotone non-increasing in S
    assert curve.contiguous == sorted(curve.contiguous, reverse=True)


# --- M1c: workload-coupled expected unusable ------------------------------


def test_request_size_distribution_is_normalized() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=500, seed=3))]
    sizes, probs = m.request_size_distribution(events)
    assert sizes == sorted(sizes)
    assert sum(probs) == pytest.approx(1.0)


def test_workload_expected_unusable_single_size() -> None:
    # checkerboard trace: every request is one 64 B block, and the surviving
    # free runs are exactly 64 B -- the workload's own requests always fit
    events = w.checkerboard(block=64, n=16)
    wu = m.workload_expected_unusable(events, "first-fit")
    assert wu.sizes == [64]
    assert wu.probs == [1.0]
    assert wu.expected == pytest.approx(wu.unusable[0])
    assert wu.expected < 0.05
    # a hypothetical doubled request against the same replay would starve;
    # the workload-coupled scalar correctly reports near-zero instead
    curve = m.pooled_usable_free_curve(events, "first-fit", [128])
    assert curve.unusable[0] > wu.expected


# --- WindowDistribution mechanics + whole-trace pooling -------------------


def test_quantile_bounds_and_validation() -> None:
    dist = m.occupancy_distribution(_final(w.coalesced(block=64, n=16)), 128)
    assert dist.quantile(0.0) == min(dist.values)
    assert dist.quantile(100.0) == max(dist.values)
    with pytest.raises(ValueError):
        dist.quantile(150.0)


def test_pooled_occupancy_and_spectrum() -> None:
    events = [e.model_copy(update={"addr": None}) for e in w.generate(w.WorkloadConfig(n_events=500, seed=3))]
    pooled = m.pooled_occupancy(events, "first-fit", 256)
    assert pooled.total_weight > 0
    assert 0.0 <= pooled.quantile(0.0) <= pooled.quantile(100.0) <= 1.0
    spec = m.occupancy_spectrum(events, "first-fit", [256, 1024, 4096])
    assert spec.windows == [256, 1024, 4096]
    assert all(0.0 <= s <= 0.5 for s in spec.std)


def test_dyadic_windows() -> None:
    assert m.dyadic_windows(1024, min_window=256) == [256, 512, 1024]
    assert m.dyadic_windows(1500, min_window=256) == [256, 512, 1024, 2048]


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
    occ = m.occupancy_distribution(snap, 64)
    assert occ.values == [] and occ.mean == 0.0 and occ.quantile(50.0) == 0.0
    assert m.usability_distribution(snap, 64, 32).cdf_at(0.5) == 0.0
