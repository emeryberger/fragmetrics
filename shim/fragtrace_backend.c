/*
 * fragtrace_backend -- allocation tracer with explicit, runtime-selected backend.
 *
 * WHY THIS EXISTS (and why it is NOT the plain LD_PRELOAD shim):
 *
 *   The naive approach -- preload an allocator and have the tracer chain to it
 *   via dlsym(RTLD_NEXT, "malloc") -- does NOT work reliably for mimalloc and
 *   jemalloc on macOS. Homebrew mimalloc exports no plain `malloc` (only
 *   `mi_malloc`), and DYLD_INSERT_LIBRARIES does not register their malloc_zone,
 *   so `malloc`/`malloc_size` keep going to libSystem. A trace captured that way
 *   would be silently mislabeled "mimalloc" while actually measuring libSystem.
 *
 *   Instead, this tracer selects a backend at runtime via FRAGTRACE_BACKEND and
 *   calls that allocator's OWN entry points, resolved by name with dlsym. No
 *   link-time dependency on the allocator, no headers, no per-allocator build.
 *   It records the backend's ACTUAL reserved size via that allocator's
 *   usable-size function, so internal fragmentation is correctly attributed.
 *
 *   FRAGTRACE_BACKEND values:
 *     system    (default) -- libc malloc/free/malloc_size|malloc_usable_size
 *     mimalloc            -- mi_malloc / mi_free / mi_malloc_usable_size
 *     jemalloc            -- mallocx / sdallocx / malloc_usable_size (je's)
 *
 *   The chosen backend's library must be PRELOADED alongside this one. If its
 *   symbols are absent, the tracer HARD-FAILS at init rather than mislabeling.
 *
 * Usage:
 *   FRAGTRACE_BACKEND=mimalloc FRAGTRACE_OUT=trace.jsonl \
 *     DYLD_INSERT_LIBRARIES=./libfragtrace_backend.dylib:/opt/homebrew/lib/libmimalloc.dylib \
 *     ./my_program
 *
 * Config (environment):
 *   FRAGTRACE_OUT      trace file path (default: fragtrace.jsonl)
 *   FRAGTRACE_BACKEND  system | mimalloc | jemalloc (default: system)
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <malloc/malloc.h>
#endif

/* ---- init buffer (same pattern as alloc8's simple_heap) -----------------
 *
 * dlsym(RTLD_DEFAULT, ...) may internally call malloc during resolution. If
 * that enters our interposer before the backend is resolved, we'd deadlock.
 * Instead, we detect the recursion via g_initializing (thread-local) and serve
 * early allocations from a static buffer. These are never freed (tiny, bounded).
 */

#define INIT_BUF_SIZE 131072
static char g_init_buf[INIT_BUF_SIZE];
static size_t g_init_pos = 0;
static int g_initializing = 0;  /* NOT __thread: TLS access can recurse into malloc on early macOS init */

static void *init_alloc(size_t sz) {
    size_t pos = (g_init_pos + 15) & ~(size_t)15;
    if (pos + sz > INIT_BUF_SIZE) return NULL;
    void *p = g_init_buf + pos;
    g_init_pos = pos + sz;
    return p;
}

static int is_init_ptr(void *p) {
    char *c = (char *)p;
    return c >= g_init_buf && c < g_init_buf + INIT_BUF_SIZE;
}

/* ---- backend function pointers ----------------------------------------- */

typedef void *(*malloc_fn)(size_t);
typedef void (*free_fn)(void *);
typedef void *(*memalign_fn)(size_t, size_t);
typedef size_t (*usable_fn)(const void *);
typedef void *(*mallocx_fn)(size_t, int);
typedef void (*sdallocx_fn)(void *, size_t, int);

enum backend { BK_SYSTEM, BK_MIMALLOC, BK_JEMALLOC };

static enum backend g_backend = BK_SYSTEM;

static malloc_fn   b_malloc;
static free_fn     b_free;
static usable_fn   b_usable;
static memalign_fn b_memalign_mi;
static mallocx_fn  b_mallocx;
static sdallocx_fn b_sdallocx;

static atomic_int g_ready = 0;  /* 0=uninit, 1=initializing, 2=ready */
/* Recursion guard: must not use __thread (TLS init can call malloc on macOS
 * early-init) or pthread_key_create (may also malloc). A plain static int is
 * safe because:
 *   (a) the recursion we guard is within a single call chain (emit → open → ...),
 *       not cross-thread;
 *   (b) the worst case of a race (two threads both set it) is a missed event,
 *       not corruption.
 * This is the same approach Scalene's samplers use. */
static int g_in_hook = 0;
static int get_in_hook(void) { return g_in_hook; }
static void set_in_hook(int v) { g_in_hook = v; }

/* ---- trace output (async-safe) ----------------------------------------- */

static atomic_uint_least64_t g_ts = 0;
static _Atomic int g_fd = -1;

static void die(const char *msg) {
    (void)!write(2, "fragtrace: ", 11);
    (void)!write(2, msg, strlen(msg));
    (void)!write(2, "\n", 1);
    _exit(97);
}

static int trace_fd(void) {
    int fd = atomic_load_explicit(&g_fd, memory_order_acquire);
    if (fd >= 0) return fd;
    const char *path = getenv("FRAGTRACE_OUT");
    if (!path) path = "fragtrace.jsonl";
    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int expected = -1;
    if (!atomic_compare_exchange_strong(&g_fd, &expected, fd)) {
        if (fd >= 0) close(fd);
        fd = atomic_load_explicit(&g_fd, memory_order_acquire);
    }
    return fd;
}

static void put_str(char *b, size_t *p, const char *s) {
    size_t n = strlen(s);
    memcpy(b + *p, s, n);
    *p += n;
}

static void put_u64(char *b, size_t *p, uint64_t v) {
    char tmp[20];
    int n = 0;
    if (v == 0) { b[(*p)++] = '0'; return; }
    while (v) { tmp[n++] = (char)('0' + (v % 10)); v /= 10; }
    while (n) b[(*p)++] = tmp[--n];
}

static void emit_alloc(void *ptr, size_t size) {
    if (!ptr) return;
    int fd = trace_fd();
    if (fd < 0) return;
    uint64_t id = (uint64_t)(uintptr_t)ptr;
    char buf[160];
    size_t p = 0;
    put_str(buf, &p, "{\"ts\":");
    put_u64(buf, &p, atomic_fetch_add_explicit(&g_ts, 1, memory_order_relaxed));
    put_str(buf, &p, ",\"op\":\"alloc\",\"id\":");
    put_u64(buf, &p, id);
    put_str(buf, &p, ",\"size\":");
    put_u64(buf, &p, (uint64_t)size);
    put_str(buf, &p, ",\"addr\":");
    put_u64(buf, &p, id);
    buf[p++] = '}'; buf[p++] = '\n';
    (void)!write(fd, buf, p);
}

static void emit_free(void *ptr) {
    if (!ptr) return;
    int fd = trace_fd();
    if (fd < 0) return;
    uint64_t id = (uint64_t)(uintptr_t)ptr;
    char buf[96];
    size_t p = 0;
    put_str(buf, &p, "{\"ts\":");
    put_u64(buf, &p, atomic_fetch_add_explicit(&g_ts, 1, memory_order_relaxed));
    put_str(buf, &p, ",\"op\":\"free\",\"id\":");
    put_u64(buf, &p, id);
    buf[p++] = '}'; buf[p++] = '\n';
    (void)!write(fd, buf, p);
}

/* ---- backend resolution ------------------------------------------------ */

static void resolve_backend(void) {
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_ready, &expected, 1)) {
        while (atomic_load_explicit(&g_ready, memory_order_acquire) != 2) { }
        return;
    }

    /* Set the recursion guard BEFORE calling dlsym, so any malloc made by dlsym
     * during resolution hits our interposer, sees g_initializing=1, and gets
     * served from the init buffer instead of recursing into resolve_backend. */
    g_initializing = 1;

    const char *want = getenv("FRAGTRACE_BACKEND");
    if (!want || !*want || strcmp(want, "system") == 0) {
        g_backend = BK_SYSTEM;
        /* On macOS: DYLD_INTERPOSE does not redirect calls made FROM WITHIN the
         * interposing dylib. So calling malloc() from here hits libc directly --
         * no recursion, no need for dlsym. On Linux: we need RTLD_NEXT. */
#if defined(__APPLE__)
        b_malloc = (malloc_fn)malloc;
        b_free   = (free_fn)free;
        b_usable = (usable_fn)malloc_size;
#else
        b_malloc = (malloc_fn)dlsym(RTLD_NEXT, "malloc");
        b_free   = (free_fn)dlsym(RTLD_NEXT, "free");
        b_usable = (usable_fn)dlsym(RTLD_NEXT, "malloc_usable_size");
#endif
        if (!b_malloc || !b_free) die("backend 'system': malloc/free not found");
    } else if (strcmp(want, "mimalloc") == 0) {
        g_backend = BK_MIMALLOC;
        b_malloc      = (malloc_fn)dlsym(RTLD_DEFAULT, "mi_malloc");
        b_free        = (free_fn)dlsym(RTLD_DEFAULT, "mi_free");
        b_usable      = (usable_fn)dlsym(RTLD_DEFAULT, "mi_malloc_usable_size");
        b_memalign_mi = (memalign_fn)dlsym(RTLD_DEFAULT, "mi_malloc_aligned");
        if (!b_malloc || !b_free || !b_usable)
            die("backend 'mimalloc' requested but mi_malloc/mi_free/"
                "mi_malloc_usable_size not found -- preload libmimalloc");
    } else if (strcmp(want, "jemalloc") == 0) {
        g_backend = BK_JEMALLOC;
        b_mallocx  = (mallocx_fn)dlsym(RTLD_DEFAULT, "mallocx");
        b_sdallocx = (sdallocx_fn)dlsym(RTLD_DEFAULT, "sdallocx");
        b_usable   = (usable_fn)dlsym(RTLD_DEFAULT, "malloc_usable_size");
        if (!b_mallocx || !b_sdallocx)
            die("backend 'jemalloc' requested but mallocx/sdallocx "
                "not found -- preload libjemalloc");
        if (!b_usable) die("backend 'jemalloc': malloc_usable_size not found");
    } else {
        die("unknown FRAGTRACE_BACKEND (use system|mimalloc|jemalloc)");
    }

    g_initializing = 0;
    atomic_store_explicit(&g_ready, 2, memory_order_release);
}

static void ensure_ready(void) {
    if (atomic_load_explicit(&g_ready, memory_order_acquire) != 2) resolve_backend();
}

/* ---- backend operations ------------------------------------------------ */

static void *be_malloc(size_t sz) {
    switch (g_backend) {
        case BK_JEMALLOC: return b_mallocx(sz ? sz : 1, 0);
        default:          return b_malloc(sz);
    }
}

static void be_free(void *ptr) {
    switch (g_backend) {
        case BK_JEMALLOC: b_sdallocx(ptr, 0, 0); return;
        default:          b_free(ptr); return;
    }
}

static size_t be_usable(void *ptr) { return b_usable ? b_usable(ptr) : 0; }

/* ---- interposed entry points ------------------------------------------- */

void *frag_malloc(size_t size) {
    /* During init, serve from static buffer (no backend yet) */
    if (g_initializing) return init_alloc(size);
    ensure_ready();
    void *ptr = be_malloc(size);
    if (!get_in_hook()) { set_in_hook(1); emit_alloc(ptr, ptr ? be_usable(ptr) : size); set_in_hook(0); }
    return ptr;
}

void frag_free(void *ptr) {
    if (!ptr) return;
    if (is_init_ptr(ptr)) return;  /* init buffer: never freed */
    if (g_initializing) return;    /* during init, swallow frees */
    ensure_ready();
    if (!get_in_hook()) { set_in_hook(1); emit_free(ptr); set_in_hook(0); }
    be_free(ptr);
}

void *frag_calloc(size_t n, size_t size) {
    size_t total = n * size;
    if (size && total / size != n) return NULL;
    if (g_initializing) {
        void *p = init_alloc(total);
        if (p) memset(p, 0, total);
        return p;
    }
    ensure_ready();
    void *ptr = be_malloc(total);
    if (ptr) memset(ptr, 0, total);
    if (!get_in_hook()) { set_in_hook(1); emit_alloc(ptr, ptr ? be_usable(ptr) : total); set_in_hook(0); }
    return ptr;
}

void *frag_realloc(void *old, size_t size) {
    if (g_initializing) return init_alloc(size);  /* best effort during init */
    if (!old) return frag_malloc(size);
    if (size == 0) { frag_free(old); return NULL; }
    if (is_init_ptr(old)) {
        /* can't recover init buffer size; just alloc new */
        void *ptr = frag_malloc(size);
        if (ptr) memcpy(ptr, old, size);  /* may over-read, but init buf is safe */
        return ptr;
    }
    ensure_ready();
    size_t oldsz = be_usable(old);
    void *ptr = be_malloc(size);
    if (ptr) {
        memcpy(ptr, old, oldsz < size ? oldsz : size);
        if (!get_in_hook()) { set_in_hook(1); emit_free(old); emit_alloc(ptr, be_usable(ptr)); set_in_hook(0); }
        be_free(old);
    }
    return ptr;
}

int frag_posix_memalign(void **out, size_t align, size_t size) {
    if (g_initializing) {
        /* can't honor alignment from init buffer; best effort */
        void *p = init_alloc(size + align);
        if (!p) return 12;
        *out = (void *)(((uintptr_t)p + align - 1) & ~(align - 1));
        return 0;
    }
    ensure_ready();
    void *ptr = NULL;
    if (g_backend == BK_MIMALLOC && b_memalign_mi) {
        ptr = b_memalign_mi(size, align);
    } else if (g_backend == BK_JEMALLOC) {
        /* MALLOCX_ALIGN(a) = ffs(a)-1 for power-of-two a */
        int flags = 0;
        size_t a = align;
        while (a > 1) { a >>= 1; flags++; }
        ptr = b_mallocx(size ? size : 1, flags);
    } else {
#if defined(__APPLE__)
        /* On macOS, calling posix_memalign from within this dylib hits libc's
         * version directly (DYLD_INTERPOSE doesn't redirect intra-image calls) */
        if (posix_memalign(&ptr, align, size) != 0) ptr = NULL;
#else
        /* On Linux, calling posix_memalign would recurse. Use the resolved real. */
        ptr = b_malloc(size);  /* simplified; proper aligned_alloc via RTLD_NEXT */
#endif
    }
    if (!ptr) return 12;
    *out = ptr;
    if (!get_in_hook()) { set_in_hook(1); emit_alloc(ptr, be_usable(ptr)); set_in_hook(0); }
    return 0;
}

/* ---- registration ------------------------------------------------------ */

#if defined(__APPLE__)
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
void *malloc(size_t size) { return frag_malloc(size); }
void  free(void *ptr) { frag_free(ptr); }
void *calloc(size_t n, size_t size) { return frag_calloc(n, size); }
void *realloc(void *old, size_t size) { return frag_realloc(old, size); }
int   posix_memalign(void **out, size_t align, size_t size) {
    return frag_posix_memalign(out, align, size);
}
#endif
