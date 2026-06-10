# fragtrace — allocation-trace capture for fragmetrics

Capture a real program's allocation trace via library interposition, then feed it
to `fragmetrics` to compute fragmentation metrics and graphs.

Two implementations:

| | file | use it when |
|---|---|---|
| **Standalone C shim** | `fragtrace.c` | zero dependencies; works today with `make`. Records the **request stream** (sizes the program asks for). |
| **alloc8 tracing allocator** | `fragtrace_alloc8.cpp` | built on [alloc8](https://github.com/emeryberger/alloc8); records the **actual reserved size** via `getSize()`, and composes with alloc8's Hoard/DieHard examples. |

## What interposition can and cannot see

Interposing `malloc`/`free` observes the **request stream** — the sizes and
lifetimes the program demands. Loading mimalloc vs jemalloc vs Hoard *underneath*
does **not** change the captured request stream: the program asks for the same
bytes regardless. So the workflow is the Johnstone–Wilson methodology:

1. **Capture** a realistic workload from a real program (under any allocator).
2. **Replay** that request stream through `fragmetrics`' reference policies
   (`--ignore-addresses --policy first-fit --policy best-fit --policy oracle …`)
   to compare fragmentation on a common, workload-coupled scale.

To measure a *production* allocator's own internal fragmentation, two options:

- Use the **alloc8 variant**, which records `getSize()` — the real reserved size,
  which differs across allocators and exposes their internal fragmentation; or
- Read the allocator's native stats: jemalloc `stats.allocated/active/resident`
  via `mallctl`, mimalloc `mi_stats_print()`, Hoard's stats. These are
  complementary to fragtrace and outside its scope.

## Standalone C shim

```bash
cd shim
make            # libfragtrace.dylib (macOS) or libfragtrace.so (Linux)
make test       # end-to-end smoke test
```

Capture, then analyze — the easy path is the `collect` subcommand, which sets up
preload for you:

```bash
python -m fragmetrics.cli collect -o trace.jsonl -- ./my_program --args
python -m fragmetrics.cli run --trace trace.jsonl --ignore-addresses \
    --policy first-fit --policy best-fit --policy oracle --out figs
```

Or drive the preload by hand:

```bash
# Linux
FRAGTRACE_OUT=trace.jsonl LD_PRELOAD=./libfragtrace.so ./my_program

# macOS (flat namespace needed to interpose reliably)
FRAGTRACE_OUT=trace.jsonl DYLD_INSERT_LIBRARIES=./libfragtrace.dylib \
    DYLD_FORCE_FLAT_NAMESPACE=1 ./my_program
```

## Backend-selectable tracer (records actual reserved size, mimalloc/jemalloc)

**`fragtrace_backend.c`** — the **recommended** shim for comparing allocators. It
calls each allocator's own entry points (`mi_malloc`, `mallocx`, etc.) and records
the actual reserved size via that allocator's usable-size function. No link-time
dependency on the allocator (symbols resolved at runtime via `dlsym`).

```bash
cd shim
make libfragtrace_backend.dylib     # macOS
make libfragtrace_backend.so        # Linux
```

Set `FRAGTRACE_BACKEND={system,mimalloc,jemalloc}` and preload the allocator
library alongside. **Critical macOS load-order**: list the allocator **first**,
the tracer **last** — dyld processes interposers in order and the last one wins
the outermost symbol:

```bash
# mimalloc on macOS (VERIFIED: produces mimalloc's size classes)
FRAGTRACE_OUT=trace.jsonl FRAGTRACE_BACKEND=mimalloc \
  DYLD_INSERT_LIBRARIES=/opt/homebrew/lib/libmimalloc.dylib:./libfragtrace_backend.dylib \
  ./my_program

# jemalloc on macOS (note: jemalloc hits libobjc's malloc_size zone trap on
# macOS; use a simple program or Linux. alloc8's zone machinery solves this.)
FRAGTRACE_OUT=trace.jsonl FRAGTRACE_BACKEND=jemalloc \
  DYLD_INSERT_LIBRARIES=/opt/homebrew/lib/libjemalloc.dylib:./libfragtrace_backend.dylib \
  ./my_program

# Linux (LD_PRELOAD — tracer FIRST so it wins malloc, allocator after for its symbols)
FRAGTRACE_OUT=trace.jsonl FRAGTRACE_BACKEND=mimalloc \
  LD_PRELOAD="./libfragtrace_backend.so:/usr/lib/libmimalloc.so" \
  ./my_program
```

If the requested backend's symbols aren't present, the tracer **hard-fails** at
startup with a clear message — it will never silently produce a trace labeled
"mimalloc" that's actually libSystem.

### Verified size classes (macOS, arm64)

| request | system (libSystem) | mimalloc 3.3.2 | jemalloc 5.3.0 |
|---|---|---|---|
| 1 | 16 | **8** | **8** |
| 7 | 16 | **8** | **8** |
| 17 | 32 | 32 | 32 |
| 33 | 48 | 48 | **64** |
| 100 | 112 | 112 | **128** |
| 200 | 224 | 224 | **256** |

All three produce genuinely distinct size classes — the tracer is correctly
forwarding to the right allocator.

### Install allocator libraries

```bash
# Debian/Ubuntu
apt-get install libmimalloc-dev libjemalloc-dev
# macOS
brew install mimalloc jemalloc        # .dylib under $(brew --prefix)/lib
```

### jemalloc macOS caveat

jemalloc under macOS DYLD interposition hits libobjc's `malloc_size(ptr)=0`
zone-validation abort (the same issue alloc8's README documents). This is not a
bug in the tracer — it's inherent to jemalloc without malloc-zone registration.
Workarounds: (a) use simple programs without ObjC, (b) use the alloc8
`TracingHeap` variant which has zone support, or (c) use Linux where this isn't
an issue.

## alloc8 tracing allocator (records actual reserved size)

`fragtrace_alloc8.cpp` is an [alloc8](https://github.com/emeryberger/alloc8)
allocator — a `TraceHeap` class (`malloc`/`free`/`memalign`/`getSize`/`lock`/
`unlock`) wrapped in `alloc8::HeapRedirect` and exported via `ALLOC8_REDIRECT`.
It forwards every call to the real system allocator (reached via `dlsym` on
Linux, à la alloc8's `simple_heap` example) and records each event, using
`getSize()` (`malloc_size` / `malloc_usable_size`) so the recorded size is the
**actual reserved bytes**, not just the request. That gap is the allocator's
internal fragmentation — invisible to the request-stream-only C shim:

```text
request:   24   8   24      (what the program asked for)
reserved:  32  16   32      (what the allocator actually gave — macOS size-classes)
```

### Build (CMake FetchContent — fetches and builds alloc8 automatically)

```bash
cd shim
cmake -S . -B build          # add -DALLOC8_TAG=<sha> to pin alloc8
cmake --build build
# -> build/libfragtrace_alloc8.{dylib,so}
```

`shim/CMakeLists.txt` declares alloc8 via `FetchContent` (no release tags are
published yet, so it defaults to the `main` branch — pin a commit SHA with
`-DALLOC8_TAG=` for reproducible builds), then builds the shared library with
alloc8's `${ALLOC8_INTERPOSE_SOURCES}` and links `alloc8::interpose`.

Then preload it like the C shim:

```bash
python -m fragmetrics.cli collect --tracer shim/build/libfragtrace_alloc8.dylib \
    -o trace.jsonl -- ./my_program
# or by hand:
DYLD_INSERT_LIBRARIES=shim/build/libfragtrace_alloc8.dylib \
    FRAGTRACE_OUT=trace.jsonl ./my_program   # macOS
LD_PRELOAD=shim/build/libfragtrace_alloc8.so \
    FRAGTRACE_OUT=trace.jsonl ./my_program   # Linux
```

> macOS note: System Integrity Protection strips `DYLD_INSERT_LIBRARIES` for
> system binaries (`/bin/ls` etc.) and their children — trace your own builds.
> For tracing a *thread-aware* allocator's per-thread behaviour, add
> `${ALLOC8_THREAD_SOURCES}` and `ALLOC8_REDIRECT_WITH_THREADS`.

## Trace format

JSONL, one event per line — the schema `fragmetrics` reads:

```json
{"ts":0,"op":"alloc","id":140234567890,"size":64,"addr":140234567890}
{"ts":1,"op":"free","id":140234567890}
```

`id`/`addr` are the returned pointer. `realloc` is recorded as `free(old)` +
`alloc(new)`. Real addresses are large and sparse, so analyze real traces with
`--ignore-addresses` (replay through a policy) rather than as a literal heap.

## Caveats

- **Threading:** events are timestamped by a monotonic atomic; lines may
  interleave under heavy threading but order is recoverable. `collect.load_trace`
  sorts by `ts`.
- **macOS:** `DYLD_INSERT_LIBRARIES` won't interpose into SIP-protected binaries;
  trace your own builds. `DYLD_FORCE_FLAT_NAMESPACE=1` is set by `collect`.
- **Linux glibc bootstrap:** `dlsym` may call `calloc` during init; the shim
  guards this (returns `NULL` pre-resolution, which glibc tolerates).
- The standalone C-shim Linux path (`LD_PRELOAD` symbol override + `dlsym`
  bootstrap) is the textbook pattern but was authored on macOS — smoke-test with
  `make test` on your Linux target.
