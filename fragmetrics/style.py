"""Publication-quality plotting style.

One module sets the house style so every figure in ``report.py`` is
paper/slide-ready by default. Design goals:

- seaborn ``paper`` theme as the base, matplotlib underneath for layout control.
- A clean serif/sans chosen by *availability*, never hard-failing on a machine
  that lacks the preferred fonts (CI included). We probe installed families with
  ``matplotlib.font_manager`` and fall back gracefully.
- Real math typography (mathtext/STIX) so ``$F(S)$``-style labels render well.
- Vector-first output (PDF/SVG with the chosen font), high-DPI PNG for preview.
- Perceptually-uniform, colorblind-safe palettes.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Literal

import matplotlib as mpl
import seaborn as sns
from matplotlib import font_manager
from matplotlib.figure import Figure

Context = Literal["paper", "notebook", "talk", "poster"]
BaseFamily = Literal["serif", "sans"]

# Preferred families, best first; we use the first that is actually installed.
_SERIF_PREFS: tuple[str, ...] = (
    "Source Serif 4",
    "Source Serif Pro",
    "Libertinus Serif",
    "STIX Two Text",
    "Charter",
    "DejaVu Serif",
)
_SANS_PREFS: tuple[str, ...] = (
    "Source Sans 3",
    "Source Sans Pro",
    "Inter",
    "Helvetica Neue",
    "Arial",
    "DejaVu Sans",
)

# colorblind-safe qualitative palette (Okabe-Ito), used for policy comparison
OKABE_ITO: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
)


def _installed() -> set[str]:
    return {f.name for f in font_manager.fontManager.ttflist}


def _renders_to_vector(family: str, generic: BaseFamily) -> bool:
    """True iff a family can actually be embedded in PDF *and* SVG output.

    Being *installed* is not enough: some fonts (e.g. STIX Two Text) carry OS/2
    flags that matplotlib's TrueType embedder rejects, raising at ``savefig``
    time. The probe must mirror the *production* config exactly -- the generic
    family class points at the candidate, and ``pdf.fonttype=42`` (TrueType
    embedding, the path that triggers the bug) -- otherwise a broken font slips
    through and crashes a real render later.
    """
    key = "font.serif" if generic == "serif" else "font.sans-serif"
    keys = ("font.family", key, "mathtext.fontset", "pdf.fonttype", "svg.fonttype")
    # matplotlib's shipped stubs type RcParams with a Literal of known keys, so a
    # computed str key (`key`) trips [index]/[arg-type] though the access is valid.
    saved = {k: mpl.rcParams[k] for k in keys}  # type: ignore[index]
    try:
        # rcParams MUST be set before artists are created: matplotlib binds a
        # text artist's font at creation time from the rcParams active then.
        mpl.rcParams["font.family"] = generic
        mpl.rcParams[key] = [family]  # type: ignore[index]
        mpl.rcParams["mathtext.fontset"] = "stix"
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["svg.fonttype"] = "none"
        fig = Figure()
        ax = fig.subplots()
        # Exercise the same elements real figures embed: title, mathtext, a
        # legend entry, and an annotation. The embedding bug only surfaces when
        # a styled variant is actually rasterized, so a minimal probe misses it.
        ax.plot([0, 1, 2], label="first-fit")
        ax.set_title(r"$F(S)$ Ag")
        ax.set_xlabel("size (bytes)")
        ax.annotate("AUC=0.50", xy=(1, 1))
        ax.legend()
        for fmt in ("pdf", "svg"):
            fig.savefig(io.BytesIO(), format=fmt)
        return True
    except Exception:
        return False
    finally:
        mpl.rcParams.update(saved)  # type: ignore[arg-type]


def can_render(family: str, generic: BaseFamily = "serif") -> bool:
    """Public predicate: can ``family`` be embedded in vector output as ``generic``?"""
    return _renders_to_vector(family, generic)


def _first_usable(prefs: Sequence[str], available: set[str], generic: BaseFamily, fallback: str) -> str:
    """First preferred family that is installed *and* renders to vector output."""
    for name in prefs:
        if name in available and _renders_to_vector(name, generic):
            return name
    return fallback


def resolve_fonts() -> tuple[str, str]:
    """Return ``(serif, sans)`` family names that are installed and renderable.

    Always returns usable families -- matplotlib's bundled DejaVu fonts are the
    guaranteed fallbacks, so this never raises and figures never hard-fail, even
    when a "nicer" font is present but broken for embedding.
    """
    available = _installed()
    serif = _first_usable(_SERIF_PREFS, available, "serif", "DejaVu Serif")
    sans = _first_usable(_SANS_PREFS, available, "sans", "DejaVu Sans")
    return serif, sans


def apply(*, context: Context = "paper", base_family: BaseFamily = "serif") -> dict[str, str]:
    """Apply the house style globally. Returns the resolved font mapping.

    ``context`` is a seaborn scaling context ("paper", "talk", "poster").
    ``base_family`` selects serif or sans as the default text family.
    """
    serif, sans = resolve_fonts()
    sns.set_theme(context=context, style="whitegrid", palette=list(OKABE_ITO))
    chosen = serif if base_family == "serif" else sans
    mpl.rcParams.update(
        {
            "font.family": base_family,
            "font.serif": [serif, "DejaVu Serif"],
            "font.sans-serif": [sans, "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,   # embed TrueType (editable text in vector output)
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return {"serif": serif, "sans": sans, "base": chosen}


# single/double journal column widths in inches
COLUMN_WIDTH = 3.4
DOUBLE_COLUMN_WIDTH = 7.0
