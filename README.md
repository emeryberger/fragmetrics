# fragmetrics

Metrics and publication-quality visualizations for **external memory
fragmentation** — richer indicators than the coarse `bytes-in-use /
bytes-allocated` occupancy ratio.

See [DESIGN.md](DESIGN.md) for the prior-art survey, metric definitions (M1–M6),
and the rationale.

---

## Install

```bash
pip install -e .          # numpy, matplotlib, seaborn, pydantic
```

Build the trace-capture shims (C compiler required):

```bash
cd shim
make                      # libfragtrace.dylib + libfragtrace_backend.dylib (macOS)
                          # libfragtrace.so    + libfragtrace_backend.so    (Linux)
```

For the alloc8-based C++ variant (optional; requires CMake 3.16+):

```bash
cd shim
cmake -S . -B build       # fetches alloc8 from GitHub automatically
cmake --build build       # -> build/libfragtrace_alloc8.{dylib,so}
```

---

## Quick start: synthetic data

```bash
# F(S) curve, MWF surface, timeseries, policy comparison, heap-layout strip
python -m fragmetrics.cli run --synthetic random --n-events 10000 --seed 42 \
    --ignore-addresses \
    --policy first-fit --policy best-fit --policy worst-fit \
    --policy buddy --policy oracle \
    --out ./figures
```

This writes PDF + SVG + PNG for each of the five figures into `./figures/`.

Canonical fixtures for testing: `--synthetic checkerboard` (external
fragmentation), `--synthetic coalesced` (no fragmentation), `--synthetic
capacity-exhaustion` (capacity limit, not fragmentation).

---

## Quick start: real program traces

### 1. Capture a trace

```bash
# simplest: use the CLI collect subcommand (sets up preload for you)
python -m fragmetrics.cli collect -o trace.jsonl -- ./my_program --args

# or drive preload by hand (macOS)
FRAGTRACE_OUT=trace.jsonl \
  DYLD_INSERT_LIBRARIES=./shim/libfragtrace.dylib \
  ./my_program

# Linux
FRAGTRACE_OUT=trace.jsonl LD_PRELOAD=./shim/libfragtrace.so ./my_program
```

### 2. Analyze and graph

```bash
python -m fragmetrics.cli run --trace trace.jsonl --ignore-addresses \
    --policy first-fit --policy best-fit --policy oracle \
    --out ./figures
```

`--ignore-addresses` drops the captured addresses and replays the request stream
through the reference placement policies — the Johnstone–Wilson methodology for
fair allocator comparison.

---

## Comparing mimalloc, jemalloc, and the system allocator

The **backend-selectable tracer** (`shim/fragtrace_backend.c`) calls each
allocator's own entry points and records the **actual reserved size** (not just
the request). This exposes internal fragmentation that differs across allocators.

### Install allocator libraries

```bash
# Debian/Ubuntu
sudo apt-get install libmimalloc-dev libjemalloc-dev
# macOS
brew install mimalloc jemalloc
```

### Build the tracer

```bash
cd shim && make
```

### Capture with a specific backend

Set `FRAGTRACE_BACKEND={system,mimalloc,jemalloc}` and preload the allocator
alongside the tracer. **On macOS: list the allocator first, tracer last** (dyld
processes interposers in order; the last one wins the outermost symbol).

```bash
# mimalloc on macOS
FRAGTRACE_BACKEND=mimalloc FRAGTRACE_OUT=mi_trace.jsonl \
  DYLD_INSERT_LIBRARIES=/opt/homebrew/lib/libmimalloc.dylib:./libfragtrace_backend.dylib \
  ./my_program

# jemalloc on macOS
FRAGTRACE_BACKEND=jemalloc FRAGTRACE_OUT=je_trace.jsonl \
  DYLD_INSERT_LIBRARIES=/opt/homebrew/lib/libjemalloc.dylib:./libfragtrace_backend.dylib \
  ./my_program

# system allocator (default if FRAGTRACE_BACKEND is unset)
FRAGTRACE_BACKEND=system FRAGTRACE_OUT=sys_trace.jsonl \
  DYLD_INSERT_LIBRARIES=./libfragtrace_backend.dylib \
  ./my_program

# Linux (LD_PRELOAD — tracer FIRST so it wins malloc, allocator after for its symbols)
FRAGTRACE_BACKEND=mimalloc FRAGTRACE_OUT=mi_trace.jsonl \
  LD_PRELOAD="./libfragtrace_backend.so:/usr/lib/libmimalloc.so" \
  ./my_program
```

If the requested backend's symbols are not loaded, the tracer **hard-fails** at
startup with a clear error — it will never silently produce a trace that's
mislabeled.

### Compare allocators

```bash
# Analyze each trace
for t in sys_trace mi_trace je_trace; do
  python -m fragmetrics.cli run --trace ${t}.jsonl --ignore-addresses \
      --policy first-fit --policy oracle --out figs_${t}
done
```

### Verified size classes (macOS arm64)

| request | system (libSystem) | mimalloc 3.3.2 | jemalloc 5.3.0 |
|---|---|---|---|
| 1 | 16 | **8** | 8 |
| 7 | 16 | **8** | 8 |
| 17 | 32 | 32 | 32 |
| 33 | 48 | 48 | **64** |
| 100 | 112 | 112 | **128** |
| 200 | 224 | 224 | **256** |

The three allocators produce genuinely distinct size classes — the tracer
correctly forwards to the right allocator and records its real reserved sizes.

---

## alloc8-based tracing allocator (C++)

`shim/fragtrace_alloc8.cpp` is an [alloc8](https://github.com/emeryberger/alloc8)
allocator that records the **actual reserved size** via `getSize()`. It uses
alloc8's platform-independent interposition glue (LD_PRELOAD /
DYLD_INSERT_LIBRARIES / DLL), following the same `simple_heap` pattern from
alloc8's examples.

### Build (fetches alloc8 from GitHub via CMake FetchContent)

```bash
cd shim
cmake -S . -B build                 # pin alloc8: -DALLOC8_TAG=<commit-sha>
cmake --build build
# -> build/libfragtrace_alloc8.{dylib,so}
```

### Use

```bash
python -m fragmetrics.cli collect \
    --tracer shim/build/libfragtrace_alloc8.dylib \
    -o trace.jsonl -- ./my_program

# or by hand:
FRAGTRACE_OUT=trace.jsonl \
  DYLD_INSERT_LIBRARIES=./shim/build/libfragtrace_alloc8.dylib \
  ./my_program
```

---

## Simulating policies as pseudo-allocators

Any placement policy can be *replayed* over a real trace to see the footprint it
would produce — no need to build and run the allocator for real. Built-in
policies: `first-fit`, `best-fit`, `worst-fit`, `next-fit`, `segregated-fit`,
`buddy`, `caching` (PyTorch-style size-class segment bucketing), and `oracle`
(perfect compaction — the moving lower bound).

The `compare` subcommand places every rung on one trace and, if a `fraggle`
binary is available, adds the **idealloc optimal** (the best *non-moving* static
placement) and the **max-load floor**:

```bash
python -m fragmetrics.cli compare --trace app.jsonl --fraggle /path/to/fraggle
```

```
  rung                      footprint       vs best
  ----------------------------------------------------
  oracle                       85.20 MiB      +0.0%   # compaction (moving) floor
  idealloc (optimal)           85.20 MiB      +0.0%   # best non-moving static plan
  best-fit                     85.39 MiB      +0.2%
  first-fit                    86.43 MiB      +1.4%
  caching                      88.00 MiB      +3.3%
  next-fit                    105.29 MiB     +23.6%
  worst-fit                   137.89 MiB     +61.9%
```

This answers "how good is policy X vs the theoretical best, on this workload?"
without implementing X. (`--fraggle` finds the binary via the flag, `$FRAGGLE`,
`PATH`, or a sibling `../idealloc` checkout; omit it for a policies-only table.)

### Two views of a PyTorch snapshot

A memory snapshot can be looked at two ways, and they answer different questions.
Pass both to `compare` (as `label=path`) to see them side by side:

```bash
python -m fragmetrics.cli compare \
    --trace allocs=llama3_allocs.jsonl \
    --trace segments=llama3_segments.jsonl \
    --fraggle /path/to/fraggle
```

- **allocs** — individual tensor placements. A wide spread here (first-fit +9%,
  worst-fit +265% on Llama3 8B FSDP) shows how much the *placement* of tensors
  matters. This is an upper bound on placement waste.
- **segments** — the memory the allocator reserved from the driver (the footprint
  that actually costs GPU memory). On real snapshots this has few, coarse objects,
  so most policies tie — the honest, conservative "recoverable memory" number.

The printed legend spells this out, so the distinction is in the output, not a
footnote.

### Custom policies

Drop in an arbitrary placement heuristic — a function returning which free run to
carve — and it becomes selectable by name:

```python
from fragmetrics.heap import register_policy, replay

def my_fit(free_runs, size, ctx):
    # return the index of the run to allocate from, or None to grow the heap
    return next((i for i, r in enumerate(free_runs) if r.length >= size), None)

register_policy("my-fit", my_fit)
snapshot, heap = list(replay(events, "my-fit"))[-1]
print(heap.peak_capacity)   # the footprint your policy achieved
```

The `caching` allocator (size-class pools, block split/merge, segment
reservation) is tunable via `CachingHeap(small_boundary=…, small_segment=…,
large_roundup=…, …)` — the defaults match the values derived in
`torch-native-allocation.md`.

## Metrics

| | metric | what it captures | function |
|---|---|---|---|
| M1 | `F(S)` fragmentation-at-size | how many S-objects can fit vs ideal | `metrics.fragmentation_curve` |
| M2 | spatial-MMU `MWF(W,S)` | worst window's usable fraction (address-space MMU) | `metrics.min_window_fill` |
| M3 | P95/P99 window fill | robust (non-worst) percentile variant of M2 | `metrics.window_fill_percentile` |
| M4 | Gorman index `Fidx(S)` | capacity vs fragmentation discriminator | `metrics.fragmentation_index` |
| M5 | blowup + ext-frag rate | peak heap / peak live; how often frag forces growth | `metrics.replay_blowup` |
| M6 | checkerboard / Gini / entropy | cheap dashboard scalars | `metrics.summarize` |

### Why these beat occupancy

Two heaps at identical 0.50 occupancy:

| layout | occupancy | F(128) | Fidx(128) | checkerboard | reading |
|---|---|---|---|---|---|
| coalesced | 0.50 | 0.00 | −1 (ok) | low | no external fragmentation |
| checkerboard | 0.50 | **1.00** | **0.84** | **0.94** | maximal external fragmentation |

Occupancy is blind to this difference; the new family separates them sharply.

---

## Figure catalogue

All figures are publication-quality (seaborn, serif fonts, vector PDF/SVG output,
colorblind-safe palettes):

1. **F(S) fragmentation curve** — log-x over size, one line per policy, P5–P95 band.
2. **MWF(W,S) heatmap** — spatial-MMU surface (window length × object size).
3. **Fragmentation time series** — Fidx / checkerboard / occupancy over the replay.
4. **Policy comparison bars** — blowup + AUC(F) per policy vs oracle.
5. **Heap layout strip** — literal used/free address map over time.

---

## Trace format

JSONL, one event per line:

```json
{"ts":0,"op":"alloc","id":1,"size":64}
{"ts":1,"op":"alloc","id":2,"size":128,"addr":64}
{"ts":2,"op":"free","id":1}
```

| field | description |
|---|---|
| `ts` | monotonic event index (ordering only) |
| `op` | `"alloc"` or `"free"` |
| `id` | object identifier (unique among live objects) |
| `size` | bytes (required on alloc, absent on free) |
| `addr` | optional: placement address. If absent, a policy assigns one at replay |

Real captured traces use the pointer as both `id` and `addr`. Addresses are
large/sparse, so always use `--ignore-addresses` for replay-based comparison.

---

## CLI reference

```
python -m fragmetrics.cli run [OPTIONS]
  --trace PATH            JSONL/CSV trace file
  --synthetic NAME        fixture or 'random' (checkerboard|coalesced|capacity-exhaustion|random)
  --policy NAME           placement policy (repeatable): first-fit|best-fit|worst-fit|buddy|oracle
  --sizes N [N ...]       object sizes to probe (default: 16..4096 log-spaced)
  --windows N [N ...]     window lengths for MWF (default: 256..65536)
  --out DIR               write the figure catalogue here (PDF+SVG+PNG)
  --ignore-addresses      drop trace addresses; let --policy decide layout
  --n-events N            event count for --synthetic random (default: 10000)
  --seed N                RNG seed for --synthetic random

python -m fragmetrics.cli collect [OPTIONS] -- COMMAND ...
  -o PATH                 trace output path (JSONL)
  --tracer PATH           path to the interposition library (default: shim/libfragtrace.{dylib,so})
  --allocator PATH        production allocator to load under the tracer
  --timeout SECS          kill the program after N seconds
```

---

## Placement policies

| policy | description |
|---|---|
| `first-fit` | first free run that fits (baseline) |
| `best-fit` | smallest adequate free run (low fragmentation) |
| `worst-fit` | largest free run (high fragmentation) |
| `buddy` | round up to power-of-two (internal fragmentation, fast coalescing) |
| `oracle` | perfect compaction — free space is always one run (the ideal normalizer) |

The oracle has blowup = 1.0 by construction; all other policies are compared
relative to it.

---

## Platform notes

### macOS

- **SIP**: `DYLD_INSERT_LIBRARIES` is stripped for system binaries (`/bin/ls`,
  etc.) and their children. Trace your own builds.
- **Load order for backend tracer**: allocator library **first**, tracer **last**
  in `DYLD_INSERT_LIBRARIES`. dyld processes interposers in order; the last one
  wins the outermost symbol.
- **jemalloc zone trap**: jemalloc under macOS DYLD interposition hits libobjc's
  `malloc_size(ptr)=0` validation abort. Workarounds: use simple (non-ObjC)
  programs, use the alloc8 TracingHeap (which has zone registration), or use
  Linux.
- **No `__thread` in interposers**: TLS access can call malloc during early dyld
  init, re-entering the interposer. The tracer uses a plain static guard instead.

### Linux

- Standard `LD_PRELOAD` symbol override + `dlsym(RTLD_NEXT, ...)` bootstrap.
- `dlsym` may call `calloc` during init; the shim guards this with a static init
  buffer (same pattern as alloc8's `simple_heap`).
- mimalloc and jemalloc both export standard `malloc`/`free`/`malloc_usable_size`
  on Linux, so the backend tracer and the plain shim both chain correctly.

---

## Develop

```bash
# Python
pytest                        # 39 tests
mypy fragmetrics tests        # strict
pyright                       # strict

# C shims
cd shim && make && make test  # build + smoke test (valid JSONL output)

# alloc8 variant
cd shim && cmake -S . -B build && cmake --build build
```

---

## Project layout

```
fragmetrics/
  __init__.py
  trace.py          pydantic Event schema + JSONL/CSV readers
  heap.py           growable heap + placement policies (free-run multiset)
  workload.py       synthetic generator + canonical fixtures
  metrics.py        M1–M6 metric family (pydantic result models)
  style.py          seaborn house style, font fallback, vector output
  report.py         five publication-quality figures
  collect.py        trace-capture driver (preload env, load+sort)
  cli.py            CLI entry point (run + collect subcommands)
tests/              golden + property + round-trip + figure-smoke tests
shim/
  fragtrace.c                standalone request-stream tracer (make)
  fragtrace_backend.c        backend-selectable tracer with mimalloc/jemalloc (make)
  fragtrace_alloc8.cpp       alloc8-based tracer recording actual reserved size (cmake)
  CMakeLists.txt             cmake FetchContent build for the alloc8 variant
  Makefile                   builds the C shims
DESIGN.md                    prior-art survey + metric definitions + citations
pyproject.toml               project metadata + mypy/pyright config
```
