# Better Metrics for External Memory Fragmentation

## Motivation

The metric most teams reach for is **occupancy**, `bytes-in-use / bytes-allocated`.
It is a single scalar that conflates internal and external fragmentation and says
nothing about *spatial layout* — which is what actually causes an allocation of
size `S` to fail while plenty of free bytes remain. Two heaps can have identical
0.50 occupancy yet be worlds apart: one with all free space in a single run (no
external fragmentation), one shattered into a checkerboard (maximally fragmented).

This package proposes a small **family of metrics** that are:

- **size-relative** — fragmentation is always *with respect to a request size `S`*;
- **spatial** — they read the contiguity/layout of free space, not just totals;
- **workload-coupled** — for allocator comparison, normalized against an ideal
  (compaction oracle) allocator replaying the *same* trace.

No single scalar carries all three properties, so the design is a few **curves**
(`F(S)`; the windowed occupancy/usability CDFs `O_W` and `U_{W,S}`; the
occupancy spectrum over scales) plus collapsed scalars for dashboards.

---

## Prior art

> Formulas below are stated from secondary knowledge and reproduced behaviourally
> in the prototype's golden tests. Confirming each against its primary source
> (linked) is an open verification item — see "Provenance to confirm".

| Metric | Source | Captures | Limitation |
|---|---|---|---|
| Occupancy `in-use/allocated` | folklore | overall slack | conflates internal+external; no spatial info; not request-coupled |
| Largest-free-block ratio `1 − maxfree/totalfree` | folklore | the single biggest run | ignores all but the largest block; size-agnostic |
| **Fragmentation index** `fragindex(order) ∈ [−1,1]` | Mel Gorman, Linux `__fragmentation_index` (`mm/vmstat.c`) | *discriminator*: would an order-k alloc fail for **capacity** (→0) or **fragmentation** (→1)? −1 if it would succeed | per power-of-two order; instantaneous; buddy-framed |
| **Unusable Free Space Index** `UFSI(j)` | Gorman `extfrag` (debugfs) | fraction of free **bytes** unusable for an order-j alloc — a curve over size | power-of-two orders; byte-weighted, not count |
| **Blowup** = max OS memory / max app demand | **Berger, Zorn & McKinley**, *Hoard* (ASPLOS 2000); reused OOPSLA 2002 — coined the term; in Hoard measures asymptotic growth from concurrent/per-thread allocation | fragmentation as a ratio vs an ideal allocator | single scalar; concurrency-framed originally |
| **Fragmentation** = peak heap / peak **live** (HWM) | Johnstone & Wilson, *The Memory Fragmentation Problem: Solved?* (ISMM 1998) — the empirical methodology (measure vs the live high-water mark); does **not** use the word "blowup" | rigorous offline measure | needs full replay; says *how much*, not *where/what size* |
| Robson bounds (`M·log₂ n`) | Robson 1977 | theoretical worst-case blowup | not an online indicator |
| **MMU / BMU** = min over time-windows of mutator utilization | Cheng & Blelloch, real-time GC | worst-case behavior as a function of **window length** | a *time* metric — we adapt the windowing idea to *space/size* |
| Allocator stats (`allocated/active/resident/retained`; `mallinfo`) | jemalloc, glibc | active−allocated (external); resident−active (page) | aggregate counters; no addresses; allocator-specific |
| Free-block distribution / entropy / Gini | scattered | dispersion of free space | descriptive, not predictive alone |

**Terminology note (corrected):** "blowup" is from Berger/Zorn/McKinley (Hoard),
*not* Johnstone–Wilson. M5 below cites Berger et al. for the term and ratio, and
Johnstone–Wilson for the live-high-water-mark measurement methodology.

---

## Proposed metric family (M1–M6)

Notation: free space is a multiset of contiguous runs `{b₁,…,bₘ}`,
`TotalFree = Σ bᵢ`, `MaxFree = max bᵢ`. `S` = candidate object size. All of this
derives from one shared structure — the coalesced free-run list of a snapshot
(`fragmetrics.heap.Snapshot`), built once per snapshot.

### M1 — Allocatable-Count curve `N(S)` and Fragmentation-at-size `F(S)`
*(`metrics.allocatable_count`, `fragmentation_at_size`, `fragmentation_curve`)*

A count-based generalization of UFSI to **arbitrary (non-power-of-two)** sizes —
the direct answer to "how many `S`-objects fit *now*":

```
N(S)  = Σ_i floor(bᵢ / S)        greedy, allocator-agnostic
N*(S) = floor(TotalFree / S)     ideal (perfectly coalesced)
F(S)  = 1 − N(S)/N*(S)  ∈ [0,1)  0 = no external fragmentation at S
```

`F(S)` plotted over log-spaced `S` is the headline analytical object; its area
(`FragmentationCurve.auc`) is a single scalar summary. When `N*(S)=0` (the ideal
heap cannot hold even one), `F(S)≡0` — that is a *capacity* limit, deferred to M4.

**M1b — usable-free curve** *(`metrics.usable_free_curve`,
`packed_usable_fraction`, `contiguous_free_fraction`)*: the byte-weighted
companion — what fraction of unallocated space is usable at request size `S`?
Two variants bracket it: `packed = N(S)·S/TotalFree` (charges per-run packing
loss; sawtooth within octaves, non-monotone — a 32 KiB gap serves one 32 KiB
object perfectly but strands ~half of itself against a 20 KiB one) and
`contiguous = Σ_{bᵢ≥S} bᵢ/TotalFree` (the survival function of the
byte-weighted free-extent length distribution; monotone, a true `1−CDF` over
`S`, and an upper bound on `packed`). The `contiguous` cliff sits at the
characteristic free-run size — the size-domain dual of the occupancy spectrum's
collapse scale — and `packed` equals the byte-weighted mean of M3 at
`W ≥ capacity`, generalizing Gorman's UFSI complement to arbitrary `S`.
(Carving objects "≤ S" would be degenerate — tiny objects fill anything — so
both variants carve at exactly `S`.)

Presentation conventions, chosen after comparing the options on canonical
layouts: report the **loss orientation** (`.unusable = 1 − packed`, up = bad),
which matches F(S)/UFSI, keeps few-percent losses visible off the axis
ceiling, and admits a log scale; `unusable_auc` is the scalar companion to
`AUC(F)`. `pooled_usable_free_curve` gives the whole-trace version (dwell ×
free-bytes weighting: unusable fraction of free *byte-time*). For rendering,
either smooth over a log-S window (the expected loss for a request near `S`;
bandwidth is the continuous analog of size-class granularity), or evaluate on
the allocator's own size-class grid drawn *steps-pre* — a request between
classes rounds UP, and for a size-class allocator, off-grid packing loss is
internal fragmentation already counted elsewhere, so the class grid avoids
double-counting.

**M1c — workload-coupled expected unusable**
*(`metrics.workload_expected_unusable`, `request_size_distribution`)*: replace
`unusable_auc`'s log-uniform size prior with the trace's **own empirical
request-size distribution**: `E[unusable(S)]` for a request actually drawn
from this workload — "the expected fraction of free byte-time unusable for the
next request." The distributional generalization of M5's `ext_growth_rate`
(which counts the realized contiguous-variant failures). Measured on the
mixture workload, it runs 5–20× *below* the log-uniform AUC: replay-freed
blocks are themselves request-sized, so the free-extent distribution
self-matches the request mix, and the log-uniform prior spends most of its
mass in octaves where no requests occur. Report both: `E` is the fragmentation
the workload experiences; `AUC` is a stress number for distribution shift
(what if tomorrow's requests are bigger).

### M2 — Occupancy CDF `O_W`
*(`metrics.occupancy_distribution`, `pooled_occupancy`, `occupancy_spectrum`)*

The windowing idea (MMU adapted from time to address space), redefined as a
full **distribution** rather than an order statistic. Tile the heap with
windows of length `W`; per window record the occupied fraction, weighted by
window length:

```
O_W(u) = fraction of heap bytes in windows with occupancy ≤ u
```

- **Exact identity:** the weighted mean equals global occupancy at *every* `W`
  — all the fragmentation information is in the *shape*, not the mean.
- **Low tail is actionable:** mass near 0 at `W` = page is what decommit /
  `madvise` reclaims; at `W` = 2 MiB it is huge-page bloat; *bimodality* at
  page scale is the Mesh signal.
- **Multiresolution structure:** with dyadic windows, occupancy at `2W` is the
  mean of its two children, so variance is non-increasing in `W` (a martingale
  coarsening; `O_W` collapses to a point mass at global occupancy as `W` → heap
  size). The decay of spread with scale — `occupancy_spectrum` — is the
  **fragmentation spectrum**: the collapse scale reads off the characteristic
  size of contiguous free/used clusters. Spread *persisting* at `W` means whole
  `W`-windows sit empty — reclaimable at that granularity (a compacted heap
  with slack keeps std ≈ 0.5 almost to the heap size); *early collapse* means
  free space is shredded below that scale and needs relocation to recover.
- **Whole-trace:** `pooled_occupancy` pools every replay snapshot weighted by
  its dwell time on the event clock (byte-time), so transient fragmentation
  counts in proportion to how long it persisted.

**Why not the min (the original MMU transfer):** the previous
`MWF(W,S) = min over windows of usable_free/W` conflated a fully-live window
(healthy density) with a shredded one (pathology) — a perfectly compacted heap
scored `MWF = 0`. And unlike MMU's time axis, where one bad window breaks a
real-time deadline, no consumer exists for the worst *spatial* window (an
allocation need not be satisfied in any particular window), so the min is
degenerate: the distribution is where the information lives. The min and any
percentile remain derived views: `dist.quantile(0)` / `dist.quantile(p)`.

### M3 — Usability CDF `U_{W,S}` *(`metrics.usability_distribution`, `pooled_usability`)*

The external-fragmentation half of the old MWF, with density factored out. Per
window, the fraction of its **free** bytes usable for size-`S` objects
(`Σ floor(seg/S)·S / free`), weighted by free bytes (windows with no free space
carry no weight):

```
U_{W,S}(u) = fraction of free bytes in windows with usability ≤ u
```

This is a *windowed, localized* complement of Gorman's UFSI: at `W` ≥ capacity
the weighted mean is exactly `N(S)·S / TotalFree`. The low tail is free space
shredded below `S` — pure external fragmentation at `S`, undiluted by dense
regions. Caveat: free runs are clipped at window boundaries before
`floor(seg/S)`, which inflates the low tail as `S` approaches `W`; read tails
at `W ≫ S`. `pooled_usability` gives the byte-time whole-trace version.

Also supported conceptually (replay-driven): per-request **headroom percentiles**
(`MaxFree − sᵢ` over the request stream) and a **safe-allocatable size** — the
largest `S` that succeeds in ≥99% of snapshots, an intuitive "you can safely ask
for up to X bytes" number.

### M4 — Generalized Gorman discriminator `Fidx(S)` *(`metrics.fragmentation_index`)*

Ports Linux `__fragmentation_index` to arbitrary `S`, labeling a (would-be)
failure as capacity vs fragmentation:

```
MaxFree ≥ S        ⇒ −1   (would succeed)
TotalFree < S      ⇒  0   (capacity problem, not fragmentation)
otherwise          ⇒ 1 − (S/TotalFree + 1)/blocks  → 1 as free space shatters
```

Pairs with M1/M3 to answer *why* something is at risk.

### M5 — Blowup & operational ext-frag rate *(`metrics.replay_blowup`)*

```
blowup          = peak_heap / peak_live          (Berger/Zorn/McKinley ratio,
                                                   live-HWM per Johnstone–Wilson)
ext_growth_rate = fraction of requests where  MaxFree < s ≤ TotalFree
                  i.e. "would have fit but for fragmentation"
```

Normalize by replaying the same trace through each policy and comparing to the
compaction **oracle** (blowup ≡ 1.0 by construction).

### M6 — Cheap observability scalars *(`metrics.summarize` and friends)*

- **Checkerboard index** — fraction of address boundaries that flip used↔free.
- **Free-size Gini / entropy** — dispersion of free runs (0 entropy = one run;
  `log₂ k` = k equal runs).
- `MaxFree/TotalFree`, plus legacy occupancy for side-by-side comparison.

**Headline set:** the `F(S)` curve, the `O_W`/`U_{W,S}` CDF families, and the
occupancy spectrum as analytical objects; `{AUC(F), safe-allocatable-size@P99,
blowup, ext_growth_rate}` as tracked scalars.

---

## Why the family beats occupancy — the canonical demonstration

The golden tests (`tests/test_metrics.py`) pin three layouts at **identical 0.50
occupancy** apart from the capacity case:

| Fixture | occupancy | F(128) | Fidx(128) | checkerboard | reading |
|---|---|---|---|---|---|
| coalesced | 0.50 | 0.00 | −1 (ok) | low | no external fragmentation |
| checkerboard | 0.50 | **1.00** | **0.84** | **0.94** | maximal external fragmentation |
| capacity-exhaustion | 0.94 | 0.00 | **0.00** | — | capacity limit, *not* fragmentation |

Occupancy cannot tell coalesced from checkerboard; the new family separates them
sharply, and correctly refuses to call capacity exhaustion "fragmentation."

Allocator comparison on a synthetic mixture workload reproduces the qualitative
Johnstone–Wilson ordering: `oracle (1.00) ≤ best-fit (1.10) < first-fit (1.13) <
buddy (1.43) < worst-fit (1.52)` by blowup.

---

## Implementation

```
fragmetrics/
  trace.py     pydantic Event schema + JSONL/CSV readers (validation at the boundary)
  heap.py      growable contiguous heap + placement policies; the coalesced
               free-run list is the shared basis for every metric
  workload.py  pydantic-configured stochastic generator + canonical fixtures
  metrics.py   M1–M6 (pydantic result models; numpy-vectorized internals)
  style.py     house seaborn style; font fallback validated against vector output
  report.py    five publication-quality figures; resilient multi-format save
  cli.py       `python -m fragmetrics.cli run ...`
tests/         golden + property + round-trip + figure-smoke tests
```

Policies: first-fit, best-fit, worst-fit, segregated/size-class, buddy
(power-of-two, slack as internal fragmentation), and a compaction **oracle**.
Address-resolved traces are honoured exactly; address-free traces get addresses
from the chosen policy, so external fragmentation is measurable either way.

Typing: fully annotated; passes `mypy --strict` and `pyright` (strict) cleanly.

### Figure catalogue (`report.py`)
1. `F(S)` fragmentation curve with P5–P95 band, one line per policy (headline).
2. `O_W` and `U_{W,S}` CDF families, one curve per window scale, pooled over
   the whole trace (byte-time weighting).
3. Fragmentation time series (safe-size / Fidx / checkerboard / occupancy).
4. Allocator-comparison bars (blowup, AUC(F)) vs the oracle.
5. Heap-layout strip — the literal used/free address map that makes the
   checkerboard visually unmistakable.
6. Occupancy spectrum — dispersion of pooled occupancy vs window scale, one
   line per policy (dyadic scales; concentrated vs diffuse waste).
7. Unusable-free curve (M1b, loss orientation) — unusable fraction of free
   byte-time vs request size, log-window smoothed (or on a size-class grid).

---

## Verification

- `pytest` — 64 tests: golden (coalesced/checkerboard/capacity), windowed-CDF
  identities (mean invariance, variance decay, UFSI complement), blowup
  ordering, M6 scalars, trace round-trip, figure smoke test.
- `python -m mypy fragmetrics tests` and `pyright` — clean under strict mode.
- `python -m fragmetrics.cli run --synthetic checkerboard --policy first-fit`
  emits the scalar table; add `--out DIR` for the figure catalogue.

### Provenance to confirm
- M4 closed form vs Linux `mm/vmstat.c __fragmentation_index`; UFSI vs `extfrag`.
- M5: "blowup" term + ratio — Berger/Zorn/McKinley (Hoard, ASPLOS 2000);
  live-HWM methodology — Johnstone & Wilson (ISMM 1998).
- M2/M3 windowing semantics vs Cheng & Blelloch MMU/BMU.

## Future work
- Page-level / RSS-returnability metrics (page-occupancy distribution) from the
  same free-run structure — useful for `madvise`/decommit and huge-page coalescing.
- Port the hot paths (`heap`, `metrics`) to Rust for large production traces.
