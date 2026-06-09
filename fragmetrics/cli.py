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
