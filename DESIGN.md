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
(`F(S)`, the `MWF(W,S)` surface) plus collapsed scalars for dashboards.

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

### M2 — Spatial-MMU `MWF(W,S)` *(`metrics.min_window_fill`)*

The user's windowing idea, formalized over the **address space** instead of time.
Tile the heap with windows of length `W`; per window, usable bytes for size `S`
are `Σ floor(seg/S)·S` over the free segments in that window:

```
MWF(W,S) = min over windows  usable(win,S) / W      MMU analog
```

Small `W` exposes local fragmentation hotspots (where the next big alloc dies);
large `W` averages toward the global ratio. Fixing `S` gives a 2-D curve over `W`;
the full `(W,S)` grid is a heatmap.

### M3 — Robust P95/P99 variants *(`metrics.window_fill_percentile`)*

MMU is the **min** (worst window) — brittle, like reporting max latency. The
robust analog replaces the min with a low-tail percentile:

```
MWF_p(W,S) = p-th percentile of per-window usable fraction   (pct=0 ⇒ MMU)
```

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

**Headline set:** the `F(S)` curve and `MWF_p(W,S)` surface as analytical objects;
`{AUC(F), safe-allocatable-size@P99, blowup, ext_growth_rate}` as tracked scalars.

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
2. `MWF(W,S)` heatmap (window × size); P-percentile variant via `pct`.
3. Fragmentation time series (safe-size / Fidx / checkerboard / occupancy).
4. Allocator-comparison bars (blowup, AUC(F)) vs the oracle.
5. Heap-layout strip — the literal used/free address map that makes the
   checkerboard visually unmistakable.

---

## Verification

- `pytest` — 34 tests: golden (coalesced/checkerboard/capacity), windowing,
  blowup ordering, M6 scalars, trace round-trip, figure smoke test.
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
