/*
 * fragtrace -- a malloc-interposition shim that records an allocation trace.
 *
 * Loaded via LD_PRELOAD (Linux) or DYLD_INSERT_LIBRARIES (macOS), it wraps the
 * allocation entry points and emits one JSONL event per call to a trace file,
 * in the exact schema fragmetrics consumes:
 *
 *     {"ts":<n>,"op":"alloc","id":<ptr>,"size":<bytes>,"addr":<ptr>}
 *     {"ts":<n>,"op":"free","id":<ptr>}
 *
 * The object id IS the returned pointer (unique among live objects), and addr
 * is the same pointer, so the resulting trace is *address-resolved*: fragmetrics
 * can compute spatial metrics directly, or --ignore-addresses to replay the
 * request stream through reference policies (the Johnstone-Wilson methodology).
 *
 * IMPORTANT -- what this measures: interposition sees the *request stream* the
 * program makes of its allocator, not the allocator's internal free lists. So
 * swapping in mimalloc/jemalloc/Hoard underneath changes *nothing* in the trace
 * (the program asks for the same bytes); the value is capturing a realistic
 * workload to replay. To compare the real allocators' own fragmentation you'd
 * need their internal stats (jemalloc `stats.*`, mimalloc `mi_stats_print`),
 * which this shim deliberately does not touch.
 *
 * Async-safety: the hot path uses a thread-local recursion guard and a single
 * raw write(2) of a preformatted line -- no stdio, no heap allocation, no locks
 * beyond an atomic counter. Output may interleave at line granularity under
 * heavy threading; events are timestamped by a monotonic atomic so order is
 * recoverable by the loader's sort.
 *
 * Configuration (environment):
 *   FRAGTRACE_OUT   path to the trace file (default: fragtrace.jsonl)
 */

#define _GNU_SOURCE
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

/* ---- platform: how we obtain the real allocator and interpose ---------- */

#if defined(__APPLE__)
/*
 * macOS: there is no LD_PRELOAD symbol override. We define our own functions
 * and register them with DYLD_INTERPOSE, calling the real libc symbols directly
 * (they remain visible). No dlsym bootstrap is needed.
 */
#include <malloc/malloc.h>

#define REAL_MALLOC  malloc
#define REAL_FREE    free
#define REAL_CALLOC  calloc
#define REAL_REALLOC realloc
#define REAL_POSIX_MEMALIGN posix_memalign

extern void *malloc(size_t);
extern void  free(void *);
extern void *calloc(size_t, size_t);
extern void *realloc(void *, size_t);
extern int   posix_memalign(void **, size_t, size_t);

#else
/*
 * Linux/glibc: define the standard names so they override via LD_PRELOAD, and
 * resolve the real symbols lazily with dlsym(RTLD_NEXT, ...). dlsym itself may
 * call calloc once during init, so calloc has a tiny static-buffer bootstrap.
 */
#include <dlfcn.h>

static void *(*real_malloc)(size_t) = NULL;
static void  (*real_free)(void *) = NULL;
static void *(*real_calloc)(size_t, size_t) = NULL;
static void *(*real_realloc)(void *, size_t) = NULL;
static int   (*real_posix_memalign)(void **, size_t, size_t) = NULL;

#define REAL_MALLOC  real_malloc
#define REAL_FREE    real_free
#define REAL_CALLOC  real_calloc
#define REAL_REALLOC real_realloc
#define REAL_POSIX_MEMALIGN real_posix_memalign

/* one-time resolution of the underlying allocator symbols */
static void resolve_real(void) {
    if (!real_malloc) {
        real_malloc  = dlsym(RTLD_NEXT, "malloc");
        real_free    = dlsym(RTLD_NEXT, "free");
        real_calloc  = dlsym(RTLD_NEXT, "calloc");
        real_realloc = dlsym(RTLD_NEXT, "realloc");
        real_posix_memalign = dlsym(RTLD_NEXT, "posix_memalign");
    }
}
#endif

/* ---- trace output ------------------------------------------------------- */

static atomic_uint_least64_t g_ts = 0;
static _Atomic int g_fd = -1;
static __thread int g_in_hook = 0;  /* recursion guard, per thread */

/* Open the trace file once, race-tolerant (last opener wins; fd leak bounded). */
static int trace_fd(void) {
    int fd = atomic_load_explicit(&g_fd, memory_order_acquire);
    if (fd >= 0) return fd;
    const char *path = getenv("FRAGTRACE_OUT");
    if (!path) path = "fragtrace.jsonl";
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int expected = -1;
    if (!atomic_compare_exchange_strong(&g_fd, &expected, fd)) {
        if (fd >= 0) close(fd);          /* someone else won the race */
        fd = atomic_load_explicit(&g_fd, memory_order_acquire);
    }
    return fd;
}

/* Append a NUL-terminated literal to buf at *pos (length via strlen, which the
 * compiler folds to a constant for string literals -- so no hand-counted
 * lengths to get wrong). */
static void put_str(char *buf, size_t *pos, const char *s) {
    size_t n = strlen(s);
    memcpy(buf + *pos, s, n);
    *pos += n;
}

/* Append an unsigned integer to buf at *pos (async-safe, no stdio). */
static void put_u64(char *buf, size_t *pos, uint64_t v) {
    char tmp[20];
    int n = 0;
    if (v == 0) { buf[(*pos)++] = '0'; return; }
    while (v) { tmp[n++] = (char)('0' + (v % 10)); v /= 10; }
    while (n) buf[(*pos)++] = tmp[--n];
}

static void emit_alloc(void *ptr, size_t size) {
    if (!ptr) return;
    int fd = trace_fd();
    if (fd < 0) return;
    uint64_t ts = atomic_fetch_add_explicit(&g_ts, 1, memory_order_relaxed);
    uintptr_t id = (uintptr_t)ptr;
    char buf[160];
    size_t p = 0;
    put_str(buf, &p, "{\"ts\":");
    put_u64(buf, &p, ts);
    put_str(buf, &p, ",\"op\":\"alloc\",\"id\":");
    put_u64(buf, &p, (uint64_t)id);
    put_str(buf, &p, ",\"size\":");
    put_u64(buf, &p, (uint64_t)size);
    put_str(buf, &p, ",\"addr\":");
    put_u64(buf, &p, (uint64_t)id);
    buf[p++] = '}'; buf[p++] = '\n';
    (void)!write(fd, buf, p);
}

static void emit_free(void *ptr) {
    if (!ptr) return;
    int fd = trace_fd();
    if (fd < 0) return;
    uint64_t ts = atomic_fetch_add_explicit(&g_ts, 1, memory_order_relaxed);
    uintptr_t id = (uintptr_t)ptr;
    char buf[96];
    size_t p = 0;
    put_str(buf, &p, "{\"ts\":");
    put_u64(buf, &p, ts);
    put_str(buf, &p, ",\"op\":\"free\",\"id\":");
    put_u64(buf, &p, (uint64_t)id);
    buf[p++] = '}'; buf[p++] = '\n';
    (void)!write(fd, buf, p);
}

/* ---- interposed entry points ------------------------------------------- */

void *frag_malloc(size_t size) {
#if !defined(__APPLE__)
    resolve_real();
#endif
    void *ptr = REAL_MALLOC(size);
    if (!g_in_hook) { g_in_hook = 1; emit_alloc(ptr, size); g_in_hook = 0; }
    return ptr;
}

void frag_free(void *ptr) {
#if !defined(__APPLE__)
    resolve_real();
#endif
    if (!g_in_hook) { g_in_hook = 1; emit_free(ptr); g_in_hook = 0; }
    REAL_FREE(ptr);
}

void *frag_calloc(size_t n, size_t size) {
#if !defined(__APPLE__)
    resolve_real();
    if (!real_calloc) return NULL;  /* pre-resolution: dlsym's own calloc */
#endif
    void *ptr = REAL_CALLOC(n, size);
    if (!g_in_hook) { g_in_hook = 1; emit_alloc(ptr, n * size); g_in_hook = 0; }
    return ptr;
}

void *frag_realloc(void *old, size_t size) {
#if !defined(__APPLE__)
    resolve_real();
#endif
    void *ptr = REAL_REALLOC(old, size);
    if (!g_in_hook) {
        g_in_hook = 1;
        /* model realloc as free(old) + alloc(new), matching the heap replayer */
        if (old) emit_free(old);
        emit_alloc(ptr, size);
        g_in_hook = 0;
    }
    return ptr;
}

int frag_posix_memalign(void **out, size_t align, size_t size) {
#if !defined(__APPLE__)
    resolve_real();
#endif
    int rc = REAL_POSIX_MEMALIGN(out, align, size);
    if (rc == 0 && !g_in_hook) { g_in_hook = 1; emit_alloc(*out, size); g_in_hook = 0; }
    return rc;
}

/* ---- registration ------------------------------------------------------ */

#if defined(__APPLE__)
/* DYLD_INTERPOSE: pair each replacement with the symbol it replaces. */
#define DYLD_INTERPOSE(_replace, _replacee) \
    __attribute__((used)) static struct { const void *r; const void *o; } \
    _interpose_##_replacee __attribute__((section("__DATA,__interpose"))) = \
    { (const void *)(unsigned long)&_replace, (const void *)(unsigned long)&_replacee };

DYLD_INTERPOSE(frag_malloc, malloc)
DYLD_INTERPOSE(frag_free, free)
DYLD_INTERPOSE(frag_calloc, calloc)
DYLD_INTERPOSE(frag_realloc, realloc)
DYLD_INTERPOSE(frag_posix_memalign, posix_memalign)
#else
/* Linux: export the standard names so the dynamic loader binds to us first. */
void *malloc(size_t size) { return frag_malloc(size); }
void  free(void *ptr) { frag_free(ptr); }
void *calloc(size_t n, size_t size) { return frag_calloc(n, size); }
void *realloc(void *old, size_t size) { return frag_realloc(old, size); }
int   posix_memalign(void **out, size_t align, size_t size) {
    return frag_posix_memalign(out, align, size);
}
#endif
