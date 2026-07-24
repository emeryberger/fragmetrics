"""Command-line entry point.

Examples
--------
Replay a synthetic checkerboard fixture through first-fit, emit scalars + figures::

    python -m fragmetrics.cli run --synthetic checkerboard --policy first-fit \
        --out /tmp/frag

Replay a recorded JSONL trace through several policies and compare them::

    python -m fragmetrics.cli run --trace heap.jsonl \
        --policy first-fit --policy best-fit --policy oracle --out /tmp/frag

The CLI drives the address-free path (placement decided by ``--policy``) unless
the trace already carries addresses, so allocator comparison is meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import collect as collect_mod
from . import metrics as m
from . import report, style, workload
from .heap import replay
from .trace import Event, read_trace

DEFAULT_SIZES: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_WINDOWS: tuple[int, ...] = (256, 1024, 4096, 16384, 65536)


def _load_events(args: argparse.Namespace) -> list[Event]:
    if args.synthetic:
        if args.synthetic in workload.FIXTURES:
            return workload.fixture(args.synthetic)
        if args.synthetic == "random":
            cfg = workload.WorkloadConfig(n_events=args.n_events, seed=args.seed)
            return workload.generate(cfg)
        raise SystemExit(
            f"unknown --synthetic {args.synthetic!r}; "
            f"choose from {sorted(workload.FIXTURES)} or 'random'"
        )
    if args.trace:
        return list(read_trace(args.trace))
    raise SystemExit("provide either --trace PATH or --synthetic NAME")


def _strip_addresses(events: Sequence[Event]) -> list[Event]:
    """Drop addresses so a placement policy decides layout (allocator comparison)."""
    return [ev.model_copy(update={"addr": None}) for ev in events]


def _scalars(events: Sequence[Event], policy: str, sizes: Sequence[int]) -> dict[str, object]:
    snaps = [snap for snap, _ in replay(list(events), policy)]
    final = snaps[-1]
    curve = m.fragmentation_curve(final, sizes)
    blowup = m.replay_blowup(events, policy)
    summary = m.summarize(final)
    return {
        "policy": policy,
        "occupancy": round(summary.occupancy, 4),
        "max_free_ratio": round(summary.max_free_ratio, 4),
        "checkerboard": round(summary.checkerboard, 4),
        "gini": round(summary.gini, 4),
        "entropy": round(summary.entropy, 4),
        "frag_auc": round(curve.auc, 4),
        "blowup": round(blowup.blowup, 4),
        "ext_growth_rate": round(blowup.ext_growth_rate, 4),
        "peak_heap": blowup.peak_heap,
        "peak_live": blowup.peak_live,
    }


def _cmd_run(args: argparse.Namespace) -> int:
    events = _load_events(args)
    if args.ignore_addresses:
        events = _strip_addresses(events)
    sizes = list(args.sizes) if args.sizes else list(DEFAULT_SIZES)
    windows = list(args.windows) if args.windows else list(DEFAULT_WINDOWS)
    policies: list[str] = list(args.policy) or ["first-fit"]

    # scalar table to stdout (JSON lines, one per policy)
    rows = [_scalars(events, p, sizes) for p in policies]
    json.dump({"rows": rows}, sys.stdout, indent=2)
    sys.stdout.write("\n")

    if args.out:
        out = Path(args.out)
        style.apply()
        events_by_policy = {p: events for p in policies}
        report.save_figure(report.plot_fragmentation_curve(events_by_policy, sizes), out / "f1_fragmentation_curve")
        primary = policies[0]
        final = [snap for snap, _ in replay(list(events), primary)][-1]
        report.save_figure(report.plot_mwf_surface(final, windows, sizes), out / "f2_mwf_surface")
        report.save_figure(report.plot_timeseries(events, primary, sizes[len(sizes) // 2]), out / "f3_timeseries")
        report.save_figure(report.plot_policy_comparison(events_by_policy, sizes), out / "f4_policy_comparison")
        snaps = [snap for snap, _ in replay(list(events), primary)]
        report.save_figure(report.plot_heap_layout(snaps), out / "f5_heap_layout")
        sys.stderr.write(f"figures written to {out}/\n")
    return 0


def _fraggle_optimal(trace_path: str, fraggle: str | None) -> dict | None:
    """Run fraggle on the trace to get the idealloc *optimal* (achievable) and the
    max-load floor. Returns {'max_load': B, 'achievable': B} or None if fraggle
    is unavailable / errored. fraggle is located via --fraggle, $FRAGGLE, PATH, or
    a sibling ../idealloc/target/release/fraggle checkout."""
    import os
    import re
    import shutil
    import subprocess

    cand = (
        fraggle
        or os.environ.get("FRAGGLE")
        or shutil.which("fraggle")
        or str(Path(__file__).resolve().parent.parent.parent / "idealloc"
                / "target" / "release" / "fraggle")
    )
    if not cand or not Path(cand).exists():
        sys.stderr.write(
            "note: fraggle not found (set --fraggle or $FRAGGLE, or build "
            "../idealloc); skipping the idealloc-optimal rung.\n"
        )
        return None
    try:
        out = subprocess.run(
            [cand, trace_path, "--ignore-addresses"],
            capture_output=True, text=True, timeout=600,
        ).stdout
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"note: fraggle run failed ({e}); skipping optimal rung.\n")
        return None

    # Parse fraggle's human report: "L (max load): <n> <unit>" and "achievable: ..."
    def _bytes(line: str) -> int | None:
        mobj = re.search(r"([\d.]+)\s*(B|KiB|MiB|GiB)", line)
        if not mobj:
            return None
        v = float(mobj.group(1))
        return int(v * {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[mobj.group(2)])

    res: dict[str, int] = {}
    for line in out.splitlines():
        if "max load" in line:
            b = _bytes(line)
            if b is not None:
                res["max_load"] = b
        elif "achievable" in line:
            b = _bytes(line)
            if b is not None:
                res["achievable"] = b
    return res or None


def _spectrum_for_trace(
    trace_path: str, policies: list[str], fraggle: str | None
) -> tuple[list[tuple[str, int]], int]:
    """Compute (spectrum, peak_live) for one trace file: every policy's footprint
    plus the idealloc optimal and max-load floor from fraggle. Spectrum entries
    are (rung name, footprint bytes), sorted small -> large."""
    events = _strip_addresses(list(read_trace(trace_path)))
    spectrum: list[tuple[str, int]] = []
    peak_live = 0
    for p in policies:
        _snap, heap = list(replay(list(events), p))[-1]
        spectrum.append((p, int(heap.peak_capacity)))
        peak_live = int(heap.peak_live)
    opt = _fraggle_optimal(trace_path, fraggle)
    if opt and "max_load" in opt:
        spectrum.append(("max-load (floor)", int(opt["max_load"])))
    if opt and "achievable" in opt:
        spectrum.append(("idealloc (optimal)", int(opt["achievable"])))
    spectrum.sort(key=lambda r: r[1])
    return spectrum, peak_live


def _split_label(spec: str) -> tuple[str, str]:
    """Parse a `--trace` argument, which may be `label=path` or just `path`."""
    if "=" in spec and not spec.split("=", 1)[0].endswith(("/", ".")):
        label, path = spec.split("=", 1)
        return label, path
    # default label = filename stem
    return Path(spec).stem, spec


def _cmd_compare(args: argparse.Namespace) -> int:
    """Footprint spectrum per trace: max-load floor -> idealloc optimal ->
    each policy -> oracle compaction. Pass multiple --trace label=path to see,
    e.g., a snapshot's allocs-level and segments-level views side by side."""
    policies: list[str] = list(args.policy) or [
        "first-fit", "best-fit", "worst-fit", "next-fit", "buddy", "caching", "oracle",
    ]
    if args.synthetic:
        # synthetic: single unlabelled trace, materialise to a temp file for fraggle
        import tempfile
        events = _load_events(args)
        tf = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for e in events:
            row = {"ts": e.ts, "op": e.op.value, "id": e.id}
            if e.size is not None:
                row["size"] = e.size
            tf.write(json.dumps(row) + "\n")
        tf.close()
        traces = [("synthetic", tf.name)]
    else:
        traces = [_split_label(t) for t in args.trace]

    def _h(b: int) -> str:
        for unit, div in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024), ("B", 1)):
            if b >= div:
                return f"{b/div:.2f} {unit}"
        return f"{b} B"

    report_json: dict[str, object] = {}
    for label, path in traces:
        spectrum, peak_live = _spectrum_for_trace(path, policies, args.fraggle)
        block: dict[str, object] = {
            "peak_live": peak_live,
            "spectrum": [{"rung": n, "footprint": fp} for n, fp in spectrum],
        }
        base = spectrum[0][1] if spectrum else 0
        sys.stderr.write(f"\n=== {label} ===\n")
        sys.stderr.write("  rung                      footprint       vs best\n")
        sys.stderr.write("  " + "-" * 52 + "\n")
        for name, fp in spectrum:
            over = f"+{100*(fp-base)/base:.1f}%" if base else "-"
            sys.stderr.write(f"  {name:24s}  {_h(fp):>12s}   {over:>8s}\n")

        # torch-native-allocation.md metric family, at peak reserved, per policy.
        if args.doc_metrics:
            events = _strip_addresses(list(read_trace(path)))
            dm_rows: dict[str, object] = {}
            sys.stderr.write("\n  torch-native metrics (at peak reserved):\n")
            sys.stderr.write(
                "  policy        density  util  frag_idx  reserved   "
                "cached_free  int_frag  segs\n"
            )
            sys.stderr.write("  " + "-" * 74 + "\n")
            for p in policies:
                dm = m.doc_metrics_at_peak(events, p)
                dm_rows[p] = dm.model_dump()
                sys.stderr.write(
                    f"  {p:12s}  {dm.memory_density:6.3f}  {dm.hbm_utilization:5.3f}  "
                    f"{dm.fragmentation_idx:8.3f}  {_h(dm.reserved):>9s}  "
                    f"{_h(dm.cached_free):>10s}  {_h(dm.internal_frag):>8s}  {dm.segments:4d}\n"
                )
            block["doc_metrics"] = dm_rows
        report_json[label] = block

    # A plain-English legend so the two levels aren't a mystery.
    if len(traces) > 1:
        sys.stderr.write(
            "\n  legend: each block is one view of the workload. For a PyTorch\n"
            "  snapshot, 'allocs' = individual tensor placement (upper bound on\n"
            "  placement waste); 'segments' = memory reserved from the driver (the\n"
            "  footprint that actually costs you). idealloc(optimal) is the best\n"
            "  a non-moving allocator could do; oracle allows moving/compaction.\n"
        )

    json.dump(report_json, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _default_tracer() -> Path:
    """Locate the built fragtrace library next to the package's shim/ dir."""
    suffix = "dylib" if collect_mod.is_macos() else "so"
    return Path(__file__).resolve().parent.parent / "shim" / f"libfragtrace.{suffix}"


def _cmd_collect(args: argparse.Namespace) -> int:
    argv: list[str] = list(args.command_argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit("provide a command to trace after '--', e.g. collect -o t.jsonl -- ls")
    tracer = Path(args.tracer) if args.tracer else _default_tracer()
    result = collect_mod.collect(
        argv,
        tracer=tracer,
        out=Path(args.out),
        allocator=Path(args.allocator) if args.allocator else None,
        timeout=args.timeout,
    )
    json.dump(result.model_dump(mode="json"), sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stderr.write(
        f"captured {result.n_events} events to {result.trace_path} "
        f"(program exit {result.returncode})\n"
        f"analyze with: python -m fragmetrics.cli run --trace {result.trace_path} "
        f"--ignore-addresses --policy first-fit --policy best-fit --policy oracle --out figs\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fragmetrics", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    coll = sub.add_parser("collect", help="capture a trace from a real program via the shim")
    coll.add_argument("-o", "--out", required=True, help="trace output path (JSONL)")
    coll.add_argument("--tracer", help="path to libfragtrace (default: shim/ next to package)")
    coll.add_argument("--allocator", help="production allocator lib to load under the tracer")
    coll.add_argument("--timeout", type=float, help="kill the program after N seconds")
    coll.add_argument(
        "command_argv",
        nargs=argparse.REMAINDER,
        metavar="-- COMMAND ...",
        help="the command to run (everything after '--')",
    )
    coll.set_defaults(func=_cmd_collect)

    run = sub.add_parser("run", help="compute fragmentation metrics for a trace")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", type=str, help="path to a JSONL/CSV trace")
    src.add_argument("--synthetic", type=str, help="fixture name or 'random'")
    run.add_argument("--policy", action="append", default=[], help="placement policy (repeatable)")
    run.add_argument("--sizes", type=int, nargs="+", help="object sizes to probe")
    run.add_argument("--windows", type=int, nargs="+", help="window lengths for MWF")
    run.add_argument("--out", type=str, help="directory for output figures")
    run.add_argument("--n-events", type=int, default=10_000, help="events for --synthetic random")
    run.add_argument("--seed", type=int, default=0, help="seed for --synthetic random")
    run.add_argument(
        "--ignore-addresses",
        action="store_true",
        help="drop trace addresses so --policy decides layout (allocator comparison)",
    )
    run.set_defaults(func=_cmd_run)

    cmp = sub.add_parser(
        "compare",
        help="footprint spectrum (max-load floor -> idealloc optimal -> each "
             "policy) per trace; pass several --trace label=path for side-by-side",
    )
    csrc = cmp.add_mutually_exclusive_group(required=True)
    csrc.add_argument("--trace", action="append", default=[],
                      help="trace to compare, as label=path or path (repeatable; "
                           "e.g. allocs=a.jsonl segments=s.jsonl for both views)")
    csrc.add_argument("--synthetic", type=str, help="fixture name or 'random'")
    cmp.add_argument("--policy", action="append", default=[],
                     help="policy to simulate (repeatable; default: a standard set)")
    cmp.add_argument("--fraggle", type=str,
                     help="path to the fraggle binary (for the idealloc-optimal rung)")
    cmp.add_argument("--doc-metrics", action="store_true",
                     help="also report the torch-native-allocation.md metric family "
                          "(density, utilization, fragmentation index, reserved, "
                          "cached-free, internal frag, segments) per policy, at peak")
    cmp.add_argument("--n-events", type=int, default=10_000, help="events for --synthetic random")
    cmp.add_argument("--seed", type=int, default=0, help="seed for --synthetic random")
    cmp.set_defaults(func=_cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: object = getattr(args, "func", None)
    if not callable(func):
        parser.print_help()
        return 1
    result = func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
