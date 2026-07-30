"""Figure smoke test: the whole catalogue renders to non-empty files and the
font-fallback path always resolves to *some* usable family.

This guards the plan's promise that plotting never hard-fails on a machine that
lacks the preferred fonts (CI included).
"""

from __future__ import annotations

from pathlib import Path

from fragmetrics import report, style
from fragmetrics import workload as w


def test_font_resolution_never_fails() -> None:
    serif, sans = style.resolve_fonts()
    # whatever is installed, we always get usable families (DejaVu is bundled)
    assert serif
    assert sans
    # the resolved fonts must actually render to vector output
    assert style.can_render(serif, "serif")
    assert style.can_render(sans, "sans")


def test_apply_returns_resolved_fonts() -> None:
    fonts = style.apply()
    assert set(fonts) == {"serif", "sans", "base"}


def test_full_catalogue_renders(tmp_path: Path) -> None:
    style.apply()
    sizes = [16, 32, 64, 128, 256, 512]
    windows = [128, 256, 512, 1024]
    events = w.checkerboard(block=64, n=16)
    events_by_policy = {"first-fit": events, "best-fit": events, "oracle": events}
    snaps = report.collect_snapshots(events, "first-fit")

    figs = {
        "f1": report.plot_fragmentation_curve(events_by_policy, sizes),
        "f2": report.plot_window_cdfs(events, "first-fit", windows, 128),
        "f3": report.plot_timeseries(events, "first-fit", 128),
        "f4": report.plot_policy_comparison(events_by_policy, sizes),
        "f5": report.plot_heap_layout(snaps),
        "f6": report.plot_occupancy_spectrum(events_by_policy, windows),
        "f7": report.plot_unusable_curve(events_by_policy, samples=64),
        "f7c": report.plot_unusable_curve(events_by_policy, size_classes=[64, 128, 256, 512]),
    }
    for name, fig in figs.items():
        written = report.save_figure(fig, tmp_path / name)
        assert written, f"{name} produced no output files"
        for path in written:
            assert path.exists() and path.stat().st_size > 0
