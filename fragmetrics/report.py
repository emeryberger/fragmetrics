"""Publication-quality figures and tabular summaries.

The figure catalogue (each a standalone function returning a ``matplotlib``
``Figure``):

1. ``plot_fragmentation_curve``  -- F(S) over size, one line per policy, with a
   P5-P95 band across snapshots. The headline figure.
2. ``plot_mwf_surface``          -- MWF(W,S) heatmap (window x size).
3. ``plot_timeseries``           -- safe-size@P99 / ext-frag / occupancy vs time.
4. ``plot_policy_comparison``    -- blowup + AUC(F) bars per policy vs oracle.
5. ``plot_heap_layout``          -- literal used/free address strip(s) over time.

``save_all`` renders the whole catalogue for a workload to PDF+SVG+PNG.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import metrics as m
from . import style
from .heap import Snapshot, replay
from .trace import Event

FloatArray = NDArray[np.float64]

VECTOR_FORMATS: tuple[str, ...] = ("pdf", "svg", "png")


# --- data collection ------------------------------------------------------


def collect_snapshots(events: Sequence[Event], policy: str, *, every: int = 1) -> list[Snapshot]:
    """Replay once and keep the sampled snapshots (the basis for time figures)."""
    return [snap for snap, _ in replay(list(events), policy, snapshot_every=every)]


def _curve_band(
    snaps: Sequence[Snapshot], sizes: Sequence[int]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Median and P5/P95 of F(S) across snapshots, per size."""
    mat = np.array(
        [[m.fragmentation_at_size(s, sz) for sz in sizes] for s in snaps],
        dtype=np.float64,
    )
    if mat.size == 0:
        z = np.zeros(len(sizes))
        return z, z, z
    return (
        np.median(mat, axis=0),
        np.percentile(mat, 5, axis=0),
        np.percentile(mat, 95, axis=0),
    )


# --- figures --------------------------------------------------------------


def plot_fragmentation_curve(
    events_by_policy: Mapping[str, Sequence[Event]],
    sizes: Sequence[int],
    *,
    every: int = 1,
) -> Figure:
    """Figure 1 (headline): F(S) vs size, one band per policy."""
    fig = Figure(figsize=(style.COLUMN_WIDTH * 1.4, style.COLUMN_WIDTH))
    ax = fig.subplots()
    palette = style.OKABE_ITO
    for i, (policy, events) in enumerate(events_by_policy.items()):
        snaps = collect_snapshots(events, policy, every=every)
        med, lo, hi = _curve_band(snaps, sizes)
        color = palette[i % len(palette)]
        auc = m.fragmentation_curve(snaps[-1], sizes).auc if snaps else 0.0
        # AUC in the legend label avoids annotation collisions when curves overlap
        ax.plot(sizes, med, color=color, label=f"{policy} (AUC={auc:.2f})", lw=1.6)
        ax.fill_between(sizes, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("requested object size $S$ (bytes)")
    ax.set_ylabel(r"external fragmentation $F(S) = 1 - N(S)/N^{*}(S)$")
    ax.set_title("Fragmentation-at-size across allocation policies")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_mwf_surface(
    snap: Snapshot,
    windows: Sequence[int],
    sizes: Sequence[int],
    *,
    pct: float = 0.0,
) -> Figure:
    """Figure 2: MWF(W,S) heatmap. ``pct``>0 plots the percentile variant."""
    grid = np.array(
        [
            [
                m.window_fill_percentile(snap, w, s, pct) if pct > 0 else m.min_window_fill(snap, w, s)
                for s in sizes
            ]
            for w in windows
        ],
        dtype=np.float64,
    )
    fig = Figure(figsize=(style.COLUMN_WIDTH * 1.5, style.COLUMN_WIDTH))
    ax = fig.subplots()
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="crest", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
    ax.set_yticks(range(len(windows)))
    ax.set_yticklabels([str(w) for w in windows])
    ax.set_xlabel("object size $S$ (bytes)")
    ax.set_ylabel("window length $W$ (bytes)")
    label = f"P{pct:g} window fill" if pct > 0 else "min window fill (MMU)"
    ax.set_title(rf"Spatial-MMU surface: $MWF(W,S)$ — {label}")
    fig.colorbar(im, ax=ax, label="usable fraction")
    return fig


def plot_timeseries(events: Sequence[Event], policy: str, probe_size: int, *, every: int = 1) -> Figure:
    """Figure 3: fragmentation indicators over the replay timeline."""
    snaps = collect_snapshots(events, policy, every=every)
    ts = [s.ts for s in snaps]
    occ = [m.summarize(s).occupancy for s in snaps]
    fidx = [max(0.0, m.fragmentation_index(s, probe_size)) for s in snaps]
    cb = [m.checkerboard_index(s) for s in snaps]
    fig = Figure(figsize=(style.DOUBLE_COLUMN_WIDTH, style.COLUMN_WIDTH))
    ax = fig.subplots()
    ax.plot(ts, occ, label="occupancy (legacy)", lw=1.3, color=style.OKABE_ITO[7], ls="--")
    ax.plot(ts, fidx, label=rf"$F_{{idx}}(S={probe_size})$", lw=1.5, color=style.OKABE_ITO[1])
    ax.plot(ts, cb, label="checkerboard index", lw=1.5, color=style.OKABE_ITO[0])
    ax.set_xlabel("event (time)")
    ax.set_ylabel("metric value")
    ax.set_title(f"Fragmentation over time — {policy}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    return fig


def plot_policy_comparison(
    events_by_policy: Mapping[str, Sequence[Event]],
    sizes: Sequence[int],
) -> Figure:
    """Figure 4: blowup and AUC(F) per policy."""
    policies = list(events_by_policy)
    blowups: list[float] = []
    aucs: list[float] = []
    for policy, events in events_by_policy.items():
        blowups.append(m.replay_blowup(events, policy).blowup)
        snaps = collect_snapshots(events, policy)
        aucs.append(m.fragmentation_curve(snaps[-1], sizes).auc if snaps else 0.0)
    fig = Figure(figsize=(style.DOUBLE_COLUMN_WIDTH, style.COLUMN_WIDTH))
    ax_b, ax_a = fig.subplots(1, 2)
    x = np.arange(len(policies))
    ax_b.bar(x, blowups, color=style.OKABE_ITO[0])
    ax_b.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(policies, rotation=30, ha="right")
    ax_b.set_ylabel("blowup = peak heap / peak live")
    ax_b.set_title("Blowup by policy")
    ax_a.bar(x, aucs, color=style.OKABE_ITO[2])
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(policies, rotation=30, ha="right")
    ax_a.set_ylabel("AUC of $F(S)$")
    ax_a.set_title("Mean fragmentation by policy")
    fig.tight_layout()
    return fig


def plot_heap_layout(snaps: Sequence[Snapshot], *, max_strips: int = 6) -> Figure:
    """Figure 5: literal used/free address strips over time (small multiples).

    Makes the checkerboard case visually unmistakable next to its scalar.
    """
    chosen = list(snaps)[:: max(1, len(snaps) // max_strips)][:max_strips] or list(snaps)
    fig = Figure(figsize=(style.DOUBLE_COLUMN_WIDTH, 0.5 * len(chosen) + 0.6))
    axes = fig.subplots(len(chosen), 1, squeeze=False)[:, 0]
    cap = max((s.capacity for s in chosen), default=1)
    for ax, snap in zip(axes, chosen):
        # paint whole strip as "used", then overlay free runs in light
        ax.axhspan(0, 1, xmin=0, xmax=1, color=style.OKABE_ITO[1], alpha=0.55)
        for run in snap.free_runs:
            ax.axvspan(run.start, run.end, color="white")
            ax.axvspan(run.start, run.end, color=style.OKABE_ITO[0], alpha=0.25)
        ax.set_xlim(0, cap)
        ax.set_yticks([])
        ax.set_ylabel(f"t={snap.ts}", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("address (bytes) — orange = used, blue = free")
    fig.suptitle("Heap layout over time", y=0.99)
    fig.tight_layout()
    return fig


# --- output ---------------------------------------------------------------


def save_figure(fig: Figure, stem: str | Path, *, formats: Sequence[str] = VECTOR_FORMATS) -> list[Path]:
    """Save a figure to multiple formats; returns the written paths.

    Resilient per-format: a backend that fails for one format (e.g. a font that
    cannot embed in PDF) does not prevent the others from being written. Raises
    only if *every* requested format fails.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    errors: list[str] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        try:
            fig.savefig(out, format=fmt)
            written.append(out)
        except Exception as exc:  # noqa: BLE001 -- backend errors vary by format
            errors.append(f"{fmt}: {exc}")
    if not written:
        raise RuntimeError(f"all formats failed for {stem}: {'; '.join(errors)}")
    return written
