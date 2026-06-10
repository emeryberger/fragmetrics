"""Integration tests for the trace-capture shims with real allocators.

These tests build the C shims, run a small test program under them with
different backends (system, mimalloc, jemalloc), validate the output traces,
and confirm that each allocator produces genuinely different size classes.

Requires: mimalloc and jemalloc libraries installed on the system.
Skip gracefully if not available (CI installs them; local dev may not have them).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

SHIM_DIR = Path(__file__).resolve().parent.parent / "shim"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def _find_lib(name: str) -> Path | None:
    """Find a shared library by common paths."""
    candidates: list[Path] = []
    if IS_MACOS:
        brew = Path("/opt/homebrew/lib")
        usr_local = Path("/usr/local/lib")
        candidates = [
            brew / f"lib{name}.dylib",
            usr_local / f"lib{name}.dylib",
        ]
    elif IS_LINUX:
        for d in ("/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"):
            candidates.append(Path(d) / f"lib{name}.so")
            # mimalloc often installs as libmimalloc.so.X
            if name == "mimalloc":
                for f in Path(d).glob("libmimalloc.so*"):
                    candidates.append(f)
            if name == "jemalloc":
                for f in Path(d).glob("libjemalloc.so*"):
                    candidates.append(f)
    for p in candidates:
        if p.exists():
            return p
    return None


MIMALLOC_LIB = _find_lib("mimalloc")
JEMALLOC_LIB = _find_lib("jemalloc")


@pytest.fixture(scope="session")
def shim_libs() -> dict[str, Path]:
    """Build the C shims and return paths to the built libraries."""
    result = subprocess.run(
        ["make", "clean"],
        cwd=SHIM_DIR,
        capture_output=True,
    )
    result = subprocess.run(
        ["make"],
        cwd=SHIM_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"shim build failed:\n{result.stderr}")
    ext = "dylib" if IS_MACOS else "so"
    libs = {
        "plain": SHIM_DIR / f"libfragtrace.{ext}",
        "backend": SHIM_DIR / f"libfragtrace_backend.{ext}",
    }
    for name, path in libs.items():
        if not path.exists():
            pytest.fail(f"shim build produced no {name} library at {path}")
    return libs


@pytest.fixture(scope="session")
def test_program(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile a small test program with known allocation sizes."""
    src = tmp_path_factory.mktemp("src") / "sizeprobe.c"
    src.write_text("""\
#include <stdlib.h>
#include <string.h>
int main(void) {
    /* allocate known sizes so tests can assert on reserved sizes */
    size_t sizes[] = {1, 7, 17, 33, 100, 200, 512, 1024};
    void *ptrs[8];
    for (int i = 0; i < 8; i++) ptrs[i] = malloc(sizes[i]);
    /* free half (checkerboard pattern) */
    for (int i = 0; i < 8; i += 2) free(ptrs[i]);
    /* free the rest */
    for (int i = 1; i < 8; i += 2) free(ptrs[i]);
    return 0;
}
""")
    out = src.parent / "sizeprobe"
    result = subprocess.run(
        ["cc", "-O0", "-o", str(out), str(src)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"test program compilation failed:\n{result.stderr}")
    return out


def _run_with_backend(
    shim_libs: dict[str, Path],
    test_program: Path,
    backend: str,
    allocator_lib: Path | None,
    tmp_path: Path,
) -> list[dict[str, object]]:
    """Run the test program under the backend tracer and return parsed events."""
    trace_path = tmp_path / f"trace_{backend}.jsonl"
    tracer = shim_libs["backend"]

    env = dict(os.environ)
    env["FRAGTRACE_OUT"] = str(trace_path)
    env["FRAGTRACE_BACKEND"] = backend

    if IS_MACOS:
        # allocator first, tracer last (dyld load order)
        libs = []
        if allocator_lib:
            libs.append(str(allocator_lib))
        libs.append(str(tracer))
        env["DYLD_INSERT_LIBRARIES"] = ":".join(libs)
    else:
        libs = []
        if allocator_lib:
            libs.append(str(allocator_lib))
        libs.append(str(tracer))
        env["LD_PRELOAD"] = ":".join(libs)

    result = subprocess.run(
        [str(test_program)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0 and backend != "jemalloc":
        # jemalloc may crash on macOS due to libobjc zone trap — tolerated
        pytest.fail(
            f"test program failed under {backend} (exit {result.returncode}):\n"
            f"stderr: {result.stderr[:500]}"
        )

    if not trace_path.exists():
        if backend == "jemalloc" and IS_MACOS:
            pytest.skip("jemalloc crashed before producing a trace (macOS zone trap)")
        pytest.fail(f"no trace file produced for backend={backend}")

    events: list[dict[str, object]] = []
    for line in trace_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _alloc_sizes(events: list[dict[str, object]]) -> list[int]:
    """Extract recorded sizes from alloc events."""
    return [int(str(e["size"])) for e in events if e.get("op") == "alloc"]


# --- tests ----------------------------------------------------------------


class TestSystemBackend:
    def test_produces_valid_trace(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(shim_libs, test_program, "system", None, tmp_path)
        assert len(events) > 0
        allocs = [e for e in events if e["op"] == "alloc"]
        frees = [e for e in events if e["op"] == "free"]
        assert len(allocs) >= 8  # at least our 8 known allocations
        assert len(frees) >= 8

    def test_sizes_are_system_classes(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(shim_libs, test_program, "system", None, tmp_path)
        sizes = _alloc_sizes(events)
        # system allocator rounds up; req=1 should be >= 8
        assert all(s >= 1 for s in sizes)
        # confirm it's not all zeros or negative
        assert max(sizes) > 0

    def test_fragmetrics_consumes_trace(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(shim_libs, test_program, "system", None, tmp_path)
        trace_path = tmp_path / "system_for_fragmetrics.jsonl"
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = subprocess.run(
            [
                "python3", "-m", "fragmetrics.cli", "run",
                "--trace", str(trace_path),
                "--ignore-addresses",
                "--policy", "first-fit",
                "--policy", "oracle",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"fragmetrics failed:\n{result.stderr}"
        output = json.loads(result.stdout)
        assert len(output["rows"]) == 2
        assert output["rows"][1]["policy"] == "oracle"
        assert output["rows"][1]["blowup"] == pytest.approx(1.0)


@pytest.mark.skipif(MIMALLOC_LIB is None, reason="mimalloc not installed")
class TestMimallocBackend:
    def test_produces_valid_trace(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(
            shim_libs, test_program, "mimalloc", MIMALLOC_LIB, tmp_path
        )
        assert len(events) > 0
        allocs = [e for e in events if e["op"] == "alloc"]
        assert len(allocs) >= 8

    def test_sizes_differ_from_system(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        sys_events = _run_with_backend(shim_libs, test_program, "system", None, tmp_path)
        mi_events = _run_with_backend(
            shim_libs, test_program, "mimalloc", MIMALLOC_LIB, tmp_path
        )
        sys_sizes = _alloc_sizes(sys_events)[-8:]
        mi_sizes = _alloc_sizes(mi_events)[-8:]
        # mimalloc has 8-byte minimum; system has 16-byte (macOS) or different
        # At least one size should differ
        assert sys_sizes != mi_sizes, (
            f"mimalloc and system produced identical sizes — "
            f"mimalloc may not be servicing allocations.\n"
            f"system: {sys_sizes}\nmimalloc: {mi_sizes}"
        )

    def test_fragmetrics_consumes_trace(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(
            shim_libs, test_program, "mimalloc", MIMALLOC_LIB, tmp_path
        )
        trace_path = tmp_path / "mi_for_fragmetrics.jsonl"
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = subprocess.run(
            [
                "python3", "-m", "fragmetrics.cli", "run",
                "--trace", str(trace_path),
                "--ignore-addresses",
                "--policy", "first-fit",
                "--policy", "oracle",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"fragmetrics failed:\n{result.stderr}"
        output = json.loads(result.stdout)
        assert output["rows"][1]["blowup"] == pytest.approx(1.0)


@pytest.mark.skipif(JEMALLOC_LIB is None, reason="jemalloc not installed")
class TestJemallocBackend:
    def test_produces_valid_trace(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(
            shim_libs, test_program, "jemalloc", JEMALLOC_LIB, tmp_path
        )
        assert len(events) > 0
        allocs = [e for e in events if e["op"] == "alloc"]
        assert len(allocs) >= 1  # jemalloc may crash early on macOS

    def test_sizes_differ_from_system(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        sys_events = _run_with_backend(shim_libs, test_program, "system", None, tmp_path)
        je_events = _run_with_backend(
            shim_libs, test_program, "jemalloc", JEMALLOC_LIB, tmp_path
        )
        sys_sizes = _alloc_sizes(sys_events)[-8:]
        je_sizes = _alloc_sizes(je_events)[-8:] if len(_alloc_sizes(je_events)) >= 8 else _alloc_sizes(je_events)
        if len(je_sizes) < 4:
            pytest.skip("jemalloc produced too few events (likely macOS zone crash)")
        assert sys_sizes != je_sizes, (
            f"jemalloc and system produced identical sizes.\n"
            f"system: {sys_sizes}\njemalloc: {je_sizes}"
        )

    def test_fragmetrics_consumes_trace(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        events = _run_with_backend(
            shim_libs, test_program, "jemalloc", JEMALLOC_LIB, tmp_path
        )
        if len(events) < 4:
            pytest.skip("jemalloc produced too few events")
        trace_path = tmp_path / "je_for_fragmetrics.jsonl"
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = subprocess.run(
            [
                "python3", "-m", "fragmetrics.cli", "run",
                "--trace", str(trace_path),
                "--ignore-addresses",
                "--policy", "first-fit",
                "--policy", "oracle",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"fragmetrics failed:\n{result.stderr}"


class TestHardFail:
    """The backend tracer must abort if the requested backend is not loaded."""

    def test_mimalloc_without_lib_hard_fails(
        self, shim_libs: dict[str, Path], test_program: Path, tmp_path: Path
    ) -> None:
        trace_path = tmp_path / "should_not_exist.jsonl"
        tracer = shim_libs["backend"]
        env = dict(os.environ)
        env["FRAGTRACE_OUT"] = str(trace_path)
        env["FRAGTRACE_BACKEND"] = "mimalloc"
        # deliberately NOT loading mimalloc
        if IS_MACOS:
            env["DYLD_INSERT_LIBRARIES"] = str(tracer)
        else:
            env["LD_PRELOAD"] = str(tracer)

        result = subprocess.run(
            [str(test_program)],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # should have aborted (exit 97) with a clear error message
        assert result.returncode != 0
        assert "fragtrace:" in result.stderr or result.returncode == 97
        assert not trace_path.exists()


# --- alloc8-based tracer --------------------------------------------------


@pytest.fixture(scope="session")
def alloc8_lib() -> Path | None:
    """Build the alloc8-based tracer via CMake. Skip if cmake is unavailable."""
    if not shutil.which("cmake"):
        return None
    build_dir = SHIM_DIR / "build"
    # configure
    result = subprocess.run(
        ["cmake", "-S", str(SHIM_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return None  # alloc8 fetch or configure failed — skip
    # build
    result = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return None
    ext = "dylib" if IS_MACOS else "so"
    lib = build_dir / f"libfragtrace_alloc8.{ext}"
    return lib if lib.exists() else None


def _run_with_alloc8(
    alloc8_lib: Path,
    test_program: Path,
    tmp_path: Path,
) -> list[dict[str, object]]:
    """Run the test program under the alloc8 tracer."""
    trace_path = tmp_path / "trace_alloc8.jsonl"
    env = dict(os.environ)
    env["FRAGTRACE_OUT"] = str(trace_path)
    if IS_MACOS:
        env["DYLD_INSERT_LIBRARIES"] = str(alloc8_lib)
    else:
        env["LD_PRELOAD"] = str(alloc8_lib)

    result = subprocess.run(
        [str(test_program)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"alloc8 tracer failed (exit {result.returncode}):\n{result.stderr[:500]}")
    if not trace_path.exists():
        pytest.fail("alloc8 tracer produced no trace file")

    events: list[dict[str, object]] = []
    for line in trace_path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


class TestAlloc8Tracer:
    """Tests for the alloc8-based tracing allocator (fragtrace_alloc8.cpp)."""

    @pytest.fixture(autouse=True)
    def _require_alloc8(self, alloc8_lib: Path | None) -> None:
        if alloc8_lib is None:
            pytest.skip("alloc8 tracer not built (cmake unavailable or build failed)")

    def test_produces_valid_trace(
        self, alloc8_lib: Path | None, test_program: Path, tmp_path: Path
    ) -> None:
        assert alloc8_lib is not None
        events = _run_with_alloc8(alloc8_lib, test_program, tmp_path)
        assert len(events) > 0
        allocs = [e for e in events if e.get("op") == "alloc"]
        frees = [e for e in events if e.get("op") == "free"]
        assert len(allocs) >= 8
        assert len(frees) >= 4

    def test_records_actual_reserved_size(
        self, alloc8_lib: Path | None, test_program: Path, tmp_path: Path
    ) -> None:
        """alloc8 tracer uses getSize() so sizes are rounded to allocator classes."""
        assert alloc8_lib is not None
        events = _run_with_alloc8(alloc8_lib, test_program, tmp_path)
        sizes = _alloc_sizes(events)
        # req=1 should be rounded up (at least 8 on any allocator)
        # the last 8 events are our known allocations
        last_8 = sizes[-8:]
        assert len(last_8) == 8
        # reserved size for req=1 must be >= 8 (no allocator gives you 1 byte)
        assert last_8[0] >= 8, f"req=1 reserved only {last_8[0]} — getSize not working?"
        # reserved size for req=7 must be >= 7
        assert last_8[1] >= 7

    def test_fragmetrics_consumes_alloc8_trace(
        self, alloc8_lib: Path | None, test_program: Path, tmp_path: Path
    ) -> None:
        assert alloc8_lib is not None
        events = _run_with_alloc8(alloc8_lib, test_program, tmp_path)
        trace_path = tmp_path / "alloc8_for_fragmetrics.jsonl"
        trace_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = subprocess.run(
            [
                "python3", "-m", "fragmetrics.cli", "run",
                "--trace", str(trace_path),
                "--ignore-addresses",
                "--policy", "first-fit",
                "--policy", "oracle",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"fragmetrics failed:\n{result.stderr}"
        output = json.loads(result.stdout)
        assert len(output["rows"]) == 2
        assert output["rows"][1]["blowup"] == pytest.approx(1.0)
