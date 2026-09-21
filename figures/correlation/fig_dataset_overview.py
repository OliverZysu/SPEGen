"""
Figure: the shape of the supervision signal.

Four panels:
  a  Step prevalence across pollutants that carry protocol annotation.
  b  Rank-frequency curve of the 378-token solvent vocabulary (log-log).
  c  Label-set size distributions for cartridge and solvent.
  d  Skewness and excess kurtosis of the 21 descriptors before and after the
     rank-Gauss transform.

Panels a-c set up the two facts that dominate every result in the paper: the
targets are sets rather than single labels, and the solvent vocabulary is
long-tailed. Panel d motivates the single most effective modelling change,
which was a change to the input representation rather than to the network.

    python -m figures.correlation.fig_dataset_overview
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from figures.style import (  # noqa: E402
    ACCENT, ACCENT_2, ACCENT_3, COL2, COL_PANEL, GRID, INK, SPEData,
    apply_style, new_panel, panel_label, save,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def panel_steps(ax, data: SPEData) -> None:
    hi = data.has_info
    rates = [100 * data.y_step[hi, t].mean() for t in range(len(data.step_names))]
    names = [s.replace("Sample loading", "Loading") for s in data.step_names]
    y = np.arange(len(names))[::-1]

    ax.barh(y, np.full(len(names), 100.0), height=0.62,
            color="#e7eef5", edgecolor="none", zorder=1)
    ax.barh(y, rates, height=0.62, color=ACCENT, edgecolor="none", zorder=2)
    for yi, r in zip(y, rates):
        ax.text(r - 1.8, yi, f"{r:.1f}", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Pollutants reporting the step (%)", labelpad=3)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=1.6, width=0.5, pad=2)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)


def panel_vocabulary(ax, data: SPEData) -> None:
    counts = np.sort(data.solvent_any_step().sum(axis=0))[::-1]
    counts = counts[counts > 0]
    rank = np.arange(1, len(counts) + 1)

    ax.loglog(rank, counts, color=ACCENT, linewidth=1.4, zorder=3)
    ax.fill_between(rank, 0.8, counts, color=ACCENT, alpha=0.12, linewidth=0,
                    zorder=2)

    top10 = counts[:10].sum() / counts.sum()
    ax.axvline(10, color=ACCENT_2, linewidth=0.8, linestyle=(0, (3, 2)),
               zorder=4)
    # Extra headroom above the curve so the note sits in empty space.
    ax.set_ylim(0.8, float(counts[0]) * 3.2)
    ax.text(11, counts[0] * 2.35,
            f"top 10 tokens carry {100 * top10:.0f}% of all use",
            fontsize=8, color=INK, va="center", ha="left", zorder=5)

    rare = int((counts < 10).sum())
    ax.axhline(10, color="#666666", linewidth=0.7, linestyle=(0, (1, 2)),
               zorder=1)
    # Bottom-left is empty on a rank-frequency curve; a right-aligned note
    # at y = 10 spans most of the log-x axis and cuts the tail.
    ax.text(1.15, 1.25, f"{rare} tokens seen < 10 times",
            fontsize=8, color=INK, va="bottom", ha="left", zorder=5)

    ax.set_xlabel("Solvent token rank", labelpad=3)
    ax.set_ylabel("Pollutants using the token", labelpad=3)
    ax.tick_params(length=1.6, width=0.5, pad=2)
    ax.tick_params(which="minor", length=0.9, width=0.4)


def panel_set_sizes(ax, data: SPEData) -> None:
    hi = data.has_info
    n_car = data.y_cartridge.sum(axis=1)
    n_sol = (data.y_solvent.sum(axis=1) > 0).sum(axis=1)[hi]

    bins = np.arange(0, 26) - 0.5
    peaks = []
    series = []
    for values, colour, label in ((n_car, ACCENT, "Cartridge classes"),
                                  (n_sol, ACCENT_2, "Distinct solvents")):
        hist, _ = np.histogram(np.clip(values, 0, 25), bins=bins)
        pct = 100 * hist / hist.sum()
        peaks.append(pct.max())
        series.append((values, colour, label, pct))

    ymax = max(peaks) * 1.18
    ax.set_ylim(0, ymax)
    for values, colour, label, pct in series:
        ax.step(np.arange(0, 25), pct, where="mid",
                color=colour, linewidth=1.3, label=label, zorder=3)
        ax.fill_between(np.arange(0, 25), 0, pct, step="mid",
                        color=colour, alpha=0.12, linewidth=0, zorder=2)
        ax.axvline(values.mean(), color=colour, linewidth=0.8,
                   linestyle=(0, (3, 2)), zorder=4)
        ax.text(values.mean(), ymax * 0.97, f"mean {values.mean():.1f}",
                fontsize=7.5, color=colour, ha="center", va="top", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                          alpha=0.9))

    ax.set_xlim(-0.5, 24.5)
    ax.set_xlabel("Labels per pollutant  (\u2265 25 pooled)", labelpad=3)
    ax.set_ylabel("Pollutants (%)", labelpad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)
    ax.legend(loc="upper right", handlelength=1.2, frameon=True,
              fancybox=False, edgecolor="none", framealpha=0.92)


def panel_transform(ax, data: SPEData) -> None:
    """Slope chart of excess kurtosis, z-score against rank-Gauss.

    A before/after scatter was the first attempt and it failed to communicate:
    the eye cannot tell which of two interleaved point clouds is the "after"
    state. A slope chart makes the direction of every descriptor explicit.
    """
    from spe.data import apply_feature_transform

    raw = data.raw.to_numpy(dtype=float)
    before, _, _ = apply_feature_transform(raw, data.train_idx, kind="zscore")
    after, _, _ = apply_feature_transform(raw, data.train_idx, kind="rank_gauss")

    k_before = stats.kurtosis(before, axis=0)
    k_after = stats.kurtosis(after, axis=0)
    n_unique = np.array([len(np.unique(raw[:, j])) for j in range(raw.shape[1])])

    # Rank-Gauss maps tied values to a shared percentile, so a descriptor that
    # takes only a handful of distinct values cannot be spread out. Separating
    # those keeps the panel honest about where the transform does nothing.
    discrete = n_unique <= 12

    for j in range(raw.shape[1]):
        colour = ACCENT_2 if discrete[j] else ACCENT_3
        ax.plot([0, 1], [k_before[j], k_after[j]], color=colour,
                linewidth=0.7, alpha=0.55 if discrete[j] else 0.75, zorder=2)
        ax.plot([0, 1], [k_before[j], k_after[j]], marker="o", ms=2.2,
                color=colour, linestyle="none", zorder=3)

    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-2.0, 8e4)
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["z-score", "rank-Gauss"])
    ax.set_ylabel("Excess kurtosis", labelpad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)
    ax.tick_params(which="minor", length=0.9, width=0.4)

    continuous = ~discrete
    note = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.92)
    ax.text(0.04, 0.97,
            f"{int(continuous.sum())} continuous descriptors, "
            f"median {np.median(k_before[continuous]):.0f} \u2192 "
            f"{abs(np.median(k_after[continuous])):.1f}",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            va="top", ha="left", zorder=5, bbox=note)
    ax.text(0.04, 0.88,
            f"{int(discrete.sum())} near-constant counters resist the transform",
            transform=ax.transAxes, fontsize=7.5, color=INK,
            va="top", ha="left", zorder=5, bbox=note)


def main() -> None:
    apply_style()
    data = SPEData(".")

    fig, ax = new_panel(COL_PANEL, 2.35)
    panel_steps(ax, data)
    save(fig, "fig_dataset_overview_a", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL_PANEL, 2.70)
    panel_vocabulary(ax, data)
    save(fig, "fig_dataset_overview_b", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL_PANEL, 2.70)
    panel_set_sizes(ax, data)
    save(fig, "fig_dataset_overview_c", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL_PANEL, 2.70)
    panel_transform(ax, data)
    save(fig, "fig_dataset_overview_d", OUT_DIR, target_width=COL_PANEL)

    fig = plt.figure(figsize=(COL2, 3.55))
    w, h = 0.205, 0.335
    left, right = 0.085, 0.348

    ax_a = fig.add_axes([left, 0.585, w, h])
    panel_steps(ax_a, data)
    panel_label(ax_a, "a", dx=-0.38, dy=1.05)

    ax_b = fig.add_axes([right, 0.585, w, h])
    panel_vocabulary(ax_b, data)
    panel_label(ax_b, "b", dx=-0.28, dy=1.05)

    ax_c = fig.add_axes([left, 0.115, w, h])
    panel_set_sizes(ax_c, data)
    panel_label(ax_c, "c", dx=-0.38, dy=1.05)

    ax_d = fig.add_axes([right, 0.115, w, h])
    panel_transform(ax_d, data)
    panel_label(ax_d, "d", dx=-0.28, dy=1.05)

    save(fig, "fig_dataset_overview", OUT_DIR)


if __name__ == "__main__":
    main()
