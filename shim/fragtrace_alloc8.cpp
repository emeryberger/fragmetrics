// fragtrace_alloc8 -- a tracing allocator built on alloc8.
//
//   https://github.com/emeryberger/alloc8
//
// alloc8 is a generic, platform-independent library for replacing the system
// allocator (the Heap-Layers / Hoard / DieHard / Scalene lineage). You define an
// allocator class with malloc/free/memalign/getSize/lock/unlock, wrap it in
// alloc8::HeapRedirect, and ALLOC8_REDIRECT generates the xxmalloc interface the
// platform wrappers (LD_PRELOAD / DYLD_INSERT_LIBRARIES / DLL) call.
//
// This allocator (TraceHeap) is a *pass-through tracer*: it forwards every call
// to the real system allocator and records a JSONL event in the schema
// fragmetrics consumes. Unlike the raw C shim (fragtrace.c), it records the
// *actual reserved size* via the backing allocator's usable-size query
// (malloc_size / malloc_usable_size) -- the size that differs across
// mimalloc / jemalloc / Hoard and exposes their internal fragmentation.
//
// The backing-allocator plumbing (dlsym + init buffer to survive early
// LD_PRELOAD init before dlsym is ready) follows alloc8's own simple_heap
// example. IMPORTANT: like every alloc8 allocator, the methods here must NOT
// call the public malloc/free -- under interposition that recurses infinitely;
// we call the *real* libc functions captured via dlsym(RTLD_NEXT, ...).
//
// Build via CMake FetchContent against alloc8 -- see CMakeLists.txt / README.md.
//
// Config (environment): FRAGTRACE_OUT -- trace file path (default fragtrace.jsonl)

#include <alloc8/alloc8.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#define FT_WRITE _write
#else
#include <fcntl.h>
#include <unistd.h>
#define FT_WRITE write
#endif

#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <malloc.h>
#endif

namespace {

// ─── async-safe JSONL writer (no stdio, no heap on the hot path) ──────────

std::atomic<int> g_fd{-1};
std::atomic<uint64_t> g_ts{0};

int trace_fd() {
  int fd = g_fd.load(std::memory_order_acquire);
  if (fd >= 0) return fd;
  const char* path = std::getenv("FRAGTRACE_OUT");
  if (!path) path = "fragtrace.jsonl";
#if defined(_WIN32)
  fd = _open(path, _O_WRONLY | _O_CREAT | _O_TRUNC, 0644);
#else
  fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
  int expected = -1;
  if (!g_fd.compare_exchange_strong(expected, fd)) {
    if (fd >= 0) {
#if defined(_WIN32)
      _close(fd);
#else
      ::close(fd);
#endif
    }
    fd = g_fd.load(std::memory_order_acquire);
  }
  return fd;
}

inline void put_str(char* b, size_t& p, const char* s) {
  size_t n = std::strlen(s);
  std::memcpy(b + p, s, n);
  p += n;
}

inline void put_u64(char* b, size_t& p, uint64_t v) {
  char tmp[20];
  int n = 0;
  if (v == 0) { b[p++] = '0'; return; }
  while (v) { tmp[n++] = char('0' + (v % 10)); v /= 10; }
  while (n) b[p++] = tmp[--n];
}

void emit_alloc(void* ptr, size_t size) {
  if (!ptr) return;
  int fd = trace_fd();
  if (fd < 0) return;
  const uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  char buf[160];
  size_t p = 0;
  put_str(buf, p, "{\"ts\":");        put_u64(buf, p, g_ts.fetch_add(1, std::memory_order_relaxed));
  put_str(buf, p, ",\"op\":\"alloc\",\"id\":"); put_u64(buf, p, id);
  put_str(buf, p, ",\"size\":");       put_u64(buf, p, static_cast<uint64_t>(size));
  put_str(buf, p, ",\"addr\":");       put_u64(buf, p, id);
  buf[p++] = '}'; buf[p++] = '\n';
  (void)!FT_WRITE(fd, buf, static_cast<unsigned>(p));
}

void emit_free(void* ptr) {
  if (!ptr) return;
  int fd = trace_fd();
  if (fd < 0) return;
  const uint64_t id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
  char buf[96];
  size_t p = 0;
  put_str(buf, p, "{\"ts\":");       put_u64(buf, p, g_ts.fetch_add(1, std::memory_order_relaxed));
  put_str(buf, p, ",\"op\":\"free\",\"id\":"); put_u64(buf, p, id);
  buf[p++] = '}'; buf[p++] = '\n';
  (void)!FT_WRITE(fd, buf, static_cast<unsigned>(p));
}

// ─── real libc backing allocator (captured via dlsym on Linux) ────────────
//
// Mirrors alloc8's simple_heap: under LD_PRELOAD our symbols shadow libc's, so
// we must reach the real ones with dlsym(RTLD_NEXT). dlsym may itself allocate
// during init, before resolution completes, so early allocations are served
// from a static init buffer. On macOS, DYLD_INTERPOSE leaves the libc symbols
// directly callable, so no bootstrap is needed.

#if defined(__linux__)
using malloc_fn = void* (*)(size_t);
using free_fn = void (*)(void*);
using aligned_alloc_fn = void* (*)(size_t, size_t);
using usable_size_fn = size_t (*)(void*);

constexpr size_t INIT_BUFFER_SIZE = 65536;
char g_initBuffer[INIT_BUFFER_SIZE];
size_t g_initBufferPos = 0;
bool g_initializing = false;

malloc_fn real_malloc = nullptr;
free_fn real_free = nullptr;
aligned_alloc_fn real_aligned_alloc = nullptr;
usable_size_fn real_usable_size = nullptr;

inline bool is_init_ptr(void* p) {
  auto c = reinterpret_cast<char*>(p);
  return c >= g_initBuffer && c < g_initBuffer + INIT_BUFFER_SIZE;
}

void* init_alloc(size_t sz) {
  size_t pos = (g_initBufferPos + 15) & ~size_t(15);
  if (pos + sz > INIT_BUFFER_SIZE) return nullptr;
  void* p = g_initBuffer + pos;
  g_initBufferPos = pos + sz;
  return p;
}

void ensure_real() {
  if (real_malloc || g_initializing) return;
  g_initializing = true;
  real_malloc = reinterpret_cast<malloc_fn>(dlsym(RTLD_NEXT, "malloc"));
  real_free = reinterpret_cast<free_fn>(dlsym(RTLD_NEXT, "free"));
  real_aligned_alloc = reinterpret_cast<aligned_alloc_fn>(dlsym(RTLD_NEXT, "aligned_alloc"));
  real_usable_size = reinterpret_cast<usable_size_fn>(dlsym(RTLD_NEXT, "malloc_usable_size"));
  g_initializing = false;
}
#endif

// ─── the alloc8 allocator: trace + forward to the real allocator ──────────

class TraceHeap {
 public:
  void* malloc(size_t sz) {
    void* ptr = backing_malloc(sz);
    // record the ACTUAL reserved size, not just the request -- this is where
    // mimalloc/jemalloc/Hoard differ and where internal fragmentation shows
    emit_alloc(ptr, ptr ? getSize(ptr) : sz);
    return ptr;
  }

  void free(void* ptr) {
    if (!ptr) return;
#if defined(__linux__)
    if (is_init_ptr(ptr)) return;  // init-buffer allocations are never freed
#endif
    emit_free(ptr);
    backing_free(ptr);
  }

  void* memalign(size_t alignment, size_t sz) {
    void* ptr = backing_memalign(alignment, sz);
    emit_alloc(ptr, ptr ? getSize(ptr) : sz);
    return ptr;
  }

  size_t getSize(void* ptr) {
    if (!ptr) return 0;
#if defined(__APPLE__)
    return malloc_size(ptr);
#elif defined(__linux__)
    if (is_init_ptr(ptr)) return 64;  // conservative
    return real_usable_size ? real_usable_size(ptr) : malloc_usable_size(ptr);
#else
    return _msize(ptr);
#endif
  }

  void lock() {}
  void unlock() {}

 private:
  static void* backing_malloc(size_t sz) {
#if defined(__linux__)
    ensure_real();
    if (g_initializing) return init_alloc(sz);
    return real_malloc ? real_malloc(sz) : std::malloc(sz);
#else
    return std::malloc(sz);
#endif
  }

  static void backing_free(void* ptr) {
#if defined(__linux__)
    if (real_free) { real_free(ptr); return; }
    std::free(ptr);
#else
    std::free(ptr);
#endif
  }

  static void* backing_memalign(size_t alignment, size_t sz) {
#if defined(__APPLE__)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, sz) != 0) ptr = nullptr;
    return ptr;
#elif defined(__linux__)
    ensure_real();
    if (g_initializing) return init_alloc(sz);
    return real_aligned_alloc ? real_aligned_alloc(alignment, sz)
                              : aligned_alloc(alignment, sz);
#elif defined(_WIN32)
    return _aligned_malloc(sz, alignment);
#else
    return aligned_alloc(alignment, sz);
#endif
  }
};

}  // namespace

// ─── generate the xxmalloc interface ──────────────────────────────────────

using TraceRedirect = alloc8::HeapRedirect<TraceHeap>;
ALLOC8_REDIRECT(TraceRedirect);
