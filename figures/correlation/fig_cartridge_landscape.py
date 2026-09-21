"""
Figure: how cartridge selection depends on lipophilicity and polarity.

Three panels:
  a  Enrichment of each cartridge class across logP octiles.
  b  Same, across TPSA octiles.
  c  logP x TPSA plane annotated with the locally dominant cartridge class.

Enrichment is the log2 ratio between the conditional and the marginal
prevalence of a cartridge class,

    E(c | bin) = log2[ P(c | bin) / P(c) ],

so 0 means "used exactly as often as average" and +1 means "twice as often".
A ratio rather than a raw rate is essential here because marginal prevalences
span 43% (C18) to 6% (ENVI-Carb); raw rates would only show which cartridge is
popular, not which molecules select it.

All axes are binned by quantile, so every column holds the same number of
molecules and the plots are immune to the heavy tails of these descriptors
(logP spans -35 to +28, TPSA reaches 2,600 A^2).

    python -m figures.correlation.fig_cartridge_landscape
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from figures.style import (  # noqa: E402
    COL15, COL2, INK, SPEData, apply_style, colorbar, new_panel,
    panel_label, save,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Aggregated placeholders, not physical sorbents — excluded from the analysis.
EXCLUDE = {"Unknown", "Null Cartridge"}

N_CARTRIDGE = 10
N_BINS = 8
MIN_CELL = 40          # minimum molecules per bin for a cell to be shown
VMAX = 1.5

# Depleted (cool) -> average (white) -> enriched (warm). Warm = "more of it"
# is the reading most people apply without consulting the colour bar.
ENRICH_CMAP = LinearSegmentedColormap.from_list(
    "enrich",
    ["#1f4e79", "#5583ac", "#b6cbdf", "#ffffff",
     "#f0cdb8", "#d78b62", "#a33d17"],
)

# Muted qualitative set for the categorical mosaic. tab10 is too saturated to
# sit next to the sequential panels without dominating the figure.
CLASS_PALETTE = ["#4f79a8", "#4c8f8b", "#c26b56", "#6e9e6b",
                 "#d19c4a", "#9c7ba8"]


def real_cartridges(data: SPEData, k: int) -> List[Tuple[int, str, int]]:
    out: List[Tuple[int, str, int]] = []
    for idx, name, count in data.top_cartridges(k + len(EXCLUDE)):
        if name in EXCLUDE:
            continue
        out.append((idx, name, count))
        if len(out) >= k:
            break
    return out


def quantile_bins(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    edges = np.unique(np.percentile(values[finite], np.linspace(0, 100, n_bins + 1)))
    assign = np.digitize(values, edges[1:-1], right=True)
    return edges, assign


def edge_labels(edges: np.ndarray) -> List[str]:
    """Compact interval labels; open-ended at both extremes to hide outliers.

    Precision follows the spread rather than the magnitude: a TPSA axis running
    from 17 to 147 needs no decimals, while a logP axis running from 0.4 to 5.4
    needs two, and printing both to a fixed width makes the narrow panels
    collide.
    """
    span = float(edges[-2] - edges[1]) if len(edges) > 3 else float(
        edges[-1] - edges[0])
    decimals = 0 if span >= 20 else (1 if span >= 2 else 2)

    def fmt(v: float) -> str:
        return f"{v:.{decimals}f}"

    n = len(edges) - 1
    labels = []
    for i in range(n):
        if i == 0:
            labels.append(f"\u2264{fmt(edges[1])}")
        elif i == n - 1:
            labels.append(f">{fmt(edges[n - 1])}")
        else:
            labels.append(f"{fmt(edges[i])}\u2013{fmt(edges[i + 1])}")
    return labels


def enrichment_matrix(
    data: SPEData,
    values: np.ndarray,
    cartridges: Sequence[Tuple[int, str, int]],
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    edges, assign = quantile_bins(values, n_bins)
    n_bin = len(edges) - 1
    matrix = np.full((len(cartridges), n_bin), np.nan)

    for b in range(n_bin):
        sel = finite & (assign == b)
        if int(sel.sum()) < MIN_CELL:
            continue
        for r, (ci, _, _) in enumerate(cartridges):
            col = data.y_cartridge[:, ci]
            marginal = col[finite].mean()
            if marginal <= 0:
                continue
            matrix[r, b] = np.log2((col[sel].mean() + 1e-3) / (marginal + 1e-3))
    return matrix, edges


def draw_heatmap(ax, matrix, edges, cartridges, xlabel, *, show_ylabels: bool = True):
    norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=ENRICH_CMAP,
                   norm=norm, interpolation="nearest")
    im.cmap.set_bad("#f2f2f2")

    n_bin = matrix.shape[1]
    ax.set_xticks(np.arange(n_bin))
    ax.set_xticklabels(edge_labels(edges), rotation=52, ha="right")
    ax.set_yticks(np.arange(len(cartridges)))
    if show_ylabels:
        ax.set_yticklabels([n for _, n, _ in cartridges])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    ax.set_xlabel(xlabel, labelpad=2)

    ax.set_xticks(np.arange(-0.5, n_bin, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cartridges), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=1.6, width=0.5, pad=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # Ring the strongest association in every column.
    for b in range(n_bin):
        col = matrix[:, b]
        if np.all(np.isnan(col)):
            continue
        r = int(np.nanargmax(col))
        if col[r] > 0.35:
            ax.plot(b, r, marker="o", ms=2.4, mfc="none", mec=INK, mew=0.7)
    return im


def dominant_map(ax, data: SPEData, cartridges, n=7):
    """logP x TPSA grid on quantile (index) axes, coloured by dominant class."""
    logp = data.descriptor("LogP_value")
    tpsa = data.descriptor("TPSA_value")
    ok = np.isfinite(logp) & np.isfinite(tpsa)

    xe, xi = quantile_bins(logp, n)
    ye, yi = quantile_bins(tpsa, n)
    nx, ny = len(xe) - 1, len(ye) - 1

    marginals = {ci: data.y_cartridge[ok, ci].mean() for ci, _, _ in cartridges}
    colour_of = {ci: CLASS_PALETTE[i % len(CLASS_PALETTE)]
                 for i, (ci, _, _) in enumerate(cartridges)}

    used: List[int] = []
    for gx in range(nx):
        for gy in range(ny):
            sel = ok & (xi == gx) & (yi == gy)
            if int(sel.sum()) < MIN_CELL:
                ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1,
                                           facecolor="#f2f2f2", edgecolor="white",
                                           linewidth=0.7))
                continue
            best_ci, best_e = None, -np.inf
            for ci, _, _ in cartridges:
                e = np.log2((data.y_cartridge[sel, ci].mean() + 1e-3)
                            / (marginals[ci] + 1e-3))
                if e > best_e:
                    best_e, best_ci = e, ci
            ax.add_patch(plt.Rectangle(
                (gx - 0.5, gy - 0.5), 1, 1,
                facecolor=colour_of[best_ci],
                alpha=float(np.clip(0.22 + 0.55 * best_e, 0.16, 0.90)),
                edgecolor="white", linewidth=0.7))
            if best_ci not in used:
                used.append(best_ci)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(np.arange(nx))
    ax.set_xticklabels(edge_labels(xe), rotation=52, ha="right")
    ax.set_yticks(np.arange(ny))
    ax.set_yticklabels(edge_labels(ye))
    ax.set_xlabel("logP septile", labelpad=2)
    ax.set_ylabel("TPSA septile (\u00c5\u00b2)", labelpad=2)
    ax.tick_params(axis="both", length=1.6, width=0.5, pad=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    name_of = {ci: nm for ci, nm, _ in cartridges}
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour_of[ci], alpha=0.78,
                             edgecolor="white", linewidth=0.5) for ci in used]
    ax.legend(handles, [name_of[ci] for ci in used], loc="upper left",
              bbox_to_anchor=(1.03, 1.03), handlelength=1.0,
              handleheight=0.9, labelspacing=0.34, borderaxespad=0,
              title="Dominant cartridge",
              title_fontproperties={"size": 8, "weight": "bold"})


CBAR_LABEL = "Enrichment  log$_2$[ P(cartridge | bin) / P(cartridge) ]"
RING_NOTE = ("\u25cb  most enriched cartridge in the bin "
             "(shown where enrichment exceeds 1.3\u00d7)")


def _standalone_heatmap(matrix, edges, cartridges, xlabel, stem) -> None:
    fig, ax = new_panel(COL15, 3.35)
    im = draw_heatmap(ax, matrix, edges, cartridges, xlabel)
    colorbar(fig, im, ax, CBAR_LABEL, ticks=[-1.5, -0.75, 0, 0.75, 1.5])
    ax.text(0.0, -0.28, RING_NOTE, transform=ax.transAxes,
            fontsize=7.5, color=INK, ha="left", va="top")
    save(fig, stem, OUT_DIR, target_width=COL15)


def main() -> None:
    apply_style()
    data = SPEData(".")
    cartridges = real_cartridges(data, N_CARTRIDGE)

    m_a, e_a = enrichment_matrix(
        data, data.descriptor("LogP_value"), cartridges, N_BINS)
    m_b, e_b = enrichment_matrix(
        data, data.descriptor("TPSA_value"), cartridges, N_BINS)

    _standalone_heatmap(m_a, e_a, cartridges, "logP octile",
                        "fig_cartridge_landscape_a")
    _standalone_heatmap(m_b, e_b, cartridges, "TPSA octile (\u00c5\u00b2)",
                        "fig_cartridge_landscape_b")

    fig, ax = new_panel(COL15, 3.20)
    dominant_map(ax, data, cartridges[:6])
    save(fig, "fig_cartridge_landscape_c", OUT_DIR, target_width=COL15)

    fig = plt.figure(figsize=(COL2, 2.95))
    top, height = 0.345, 0.615

    ax_a = fig.add_axes([0.090, top, 0.250, height])
    im = draw_heatmap(ax_a, m_a, e_a, cartridges, "logP octile")
    panel_label(ax_a, "a", dx=-0.575, dy=1.02)

    ax_b = fig.add_axes([0.365, top, 0.205, height])
    draw_heatmap(ax_b, m_b, e_b, cartridges, "TPSA octile (\u00c5\u00b2)",
                 show_ylabels=False)
    panel_label(ax_b, "b", dx=-0.075, dy=1.02)

    ax_c = fig.add_axes([0.670, top, 0.185, height])
    # Six classes keep the mosaic legible; the tail classes never dominate a
    # cell of this size anyway.
    dominant_map(ax_c, data, cartridges[:6])
    panel_label(ax_c, "c", dx=-0.30, dy=1.02)

    cax = fig.add_axes([0.098, 0.088, 0.200, 0.030])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal",
                        ticks=[-1.5, -0.75, 0, 0.75, 1.5])
    cbar.set_label(CBAR_LABEL, fontsize=7.0, labelpad=2)
    cbar.ax.tick_params(labelsize=7.0, length=1.8, width=0.5, pad=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.text(-0.025, 0.5, "depleted", transform=cbar.ax.transAxes,
                 fontsize=7.0, color=INK, ha="right", va="center")
    cbar.ax.text(1.025, 0.5, "enriched", transform=cbar.ax.transAxes,
                 fontsize=7.0, color=INK, ha="left", va="center")

    key = fig.add_axes([0.430, 0.088, 0.020, 0.030])
    key.set_axis_off()
    key.set_xlim(-1, 1)
    key.set_ylim(-1, 1)
    key.plot(0, 0, marker="o", ms=2.4, mfc="none", mec=INK, mew=0.7)
    fig.text(0.462, 0.103, RING_NOTE.replace("\u25cb  ", ""),
             fontsize=7.5, color=INK, va="center")

    save(fig, "fig_cartridge_landscape", OUT_DIR)


if __name__ == "__main__":
    main()
