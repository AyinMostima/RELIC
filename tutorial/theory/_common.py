"""Shared utilities for RELIC system-characteristic tutorials.

Provides:
- Times New Roman matplotlib configuration (publication-quality)
- Colorblind-safe palette with dark/light variants
- Figure saving helper (memory-safe PNG)
- Layout helpers (panel labels, external legends, text boxes)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Elegant color palettes
# ---------------------------------------------------------------------------
# Primary palette: deep, rich tones (colorblind-safe, Wong 2011 inspired)
PALETTE = {
    "blue":       "#2D6A8F",
    "vermillion": "#C44E52",
    "teal":       "#1A9988",
    "purple":     "#7B6888",
    "orange":     "#DD8452",
    "sky":        "#55A8D5",
    "gold":       "#CCB974",
    "slate":      "#4C4C66",
}
CB_PALETTE = list(PALETTE.values())

# Soft pastel variant for fills/backgrounds
FILL_PALETTE = [
    "#B8D4E3",  # soft blue
    "#E8B4B4",  # soft red
    "#A8DAD0",  # soft teal
    "#C5B8D4",  # soft purple
    "#F0D0A8",  # soft orange
    "#AAD0E8",  # soft sky
    "#E8DFB0",  # soft gold
    "#B0B0C0",  # soft slate
]

# RELIC branding colors
RELIC_ACCENT = "#1B3A4B"
RELIC_HIGHLIGHT = "#3498DB"

# ---------------------------------------------------------------------------
# Times New Roman / Publication-style matplotlib configuration
# ---------------------------------------------------------------------------
# Font: Times New Roman (serif) for publication quality.
# Reference: Nature artwork guidelines (2024)

NATURE_RC = {
    # Font -- Times New Roman (serif family)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    # Math font -- match serif
    "mathtext.fontset": "stix",
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    # Axes -- clean, minimal
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.facecolor": "white",
    # Figure -- memory-safe DPI
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "savefig.facecolor": "white",
    # Legend -- minimal frame
    "legend.frameon": False,
    "legend.borderaxespad": 0.5,
}


def apply_nature_style():
    """Apply Times New Roman publication-style RC parameters globally."""
    mpl.rcParams.update(NATURE_RC)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CB_PALETTE)


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add a bold lowercase panel label (a, b, c ...) in Nature style.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str -- e.g. 'a', 'b', 'c'
    x, y : float -- position in axes coordinates
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color="#222222",
        fontfamily="serif",
    )


def savefig(fig, name, formats=("png",)):
    """Save figure to the figures directory (memory-safe).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str -- base filename (without extension)
    formats : tuple of str
    """
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        size_kb = path.stat().st_size / 1024
        print(f"  Saved: {path.name}  ({size_kb:.0f} KB)")


def add_legend_outside(ax, **kwargs):
    """Place legend below axes or in best blank area, frameless, bold text.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    **kwargs : passed to ax.legend()
    """
    defaults = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        fontsize=6,
        frameon=False,
        ncol=3,
        borderaxespad=0,
        prop={"weight": "bold"},
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


def add_text_box(ax, text, loc="upper right", fontsize=7):
    """Add a text box with semi-transparent background.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str
    loc : str -- 'upper right', 'upper left', 'lower right', 'lower left'
    fontsize : float
    """
    loc_map = {
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left":  (0.03, 0.97, "left", "top"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left":  (0.03, 0.03, "left", "bottom"),
    }
    x, y, ha, va = loc_map.get(loc, (0.97, 0.97, "right", "top"))
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize, ha=ha, va=va,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="none", alpha=0.85),
    )


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------

def mm_to_inches(mm):
    """Convert millimeters to inches for matplotlib figure sizing."""
    return mm / 25.4


SINGLE_COL_WIDTH = mm_to_inches(89)   # 89 mm  single column
DOUBLE_COL_WIDTH = mm_to_inches(183)  # 183 mm double column


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_carbon_pools(n_pools=6, total_c=1000.0, seed=42):
    """Generate synthetic initial carbon pool sizes.

    Parameters
    ----------
    n_pools : int
    total_c : float -- total carbon [g C m^-2]
    seed : int

    Returns
    -------
    pools : np.ndarray of shape (n_pools,)
    names : list of str
    """
    rng = np.random.default_rng(seed)
    fracs = rng.dirichlet(np.ones(n_pools))
    pools = fracs * total_c
    names = [
        "Leaf", "Wood", "Root",
        "Litter", "SOM_fast", "SOM_slow",
    ][:n_pools]
    return pools, names
