"""
Figure: held-out performance against the tabular baselines.

  a  Every model on every reported metric, min-max normalised within each
     metric so eleven different scales share one axis.
  b  Margin of the best variant over the strongest baseline on each metric,
     with the seed-to-seed standard deviation as the error bar.
  c  Where the top-1 chain breaks: cartridge, then step given cartridge, then
     solvent given the correct prefix.

All models take the same 21 molecular descriptors and the same split, so the
comparison isolates the modelling choices.

    python -m figures.results.fig_benchmark
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402

from figures.results.data import (  # noqa: E402
    BASELINES, DISPLAY, LOWER_IS_BETTER, METRIC_LABEL, OURS,
    load_baseline_chain, load_summary,
)
from figures.style import (  # noqa: E402
    ACCENT, ACCENT_2, COL15, COL2, COL_PANEL, GRID, INK, align_labels,
    apply_style, column_label, fit_width, new_panel, save,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

METRICS = ["car_top1", "chain", "hit1", "sss", "hc", "gus",
           "car_f1", "sol_f1", "ratio_w10", "ratio_mae", "car_em"]

# The metric names set two margins at once: rotated, they set the height under
# panel a; horizontal, they set the left margin of panel b and with it the
# position of the panel letters. Both panels use the short forms so that
# neither margin is driven by a single long name.
SHORT_LABEL = {
    "chain": "Chain acc.", "sss": "Scheme sim.", "hc": "Hier. consist.",
    "ratio_w10": "Ratio \u22640.10", "car_em": "Cartridge EM",
    "car_f1": "Cartridge F1", "sol_f1": "Solvent F1", "gus": "Global utility",
    "ratio_mae": "Ratio MAE", "car_top1": "Cartridge top-1", "hit1": "Hit@1",
}

OURS_COLOR = {
    "Ours (balanced)": ACCENT,
    "Ours (chain-oriented)": "#4a7c59",
    "Ours (exact-match)": "#8a6d9e",
    "Ours (bagging ensemble)": ACCENT_2,
}


def oriented(metric: str, value: float) -> float:
    """Flip sign so that larger is always better."""
    return -value if metric in LOWER_IS_BETTER else value


def panel_normalised(ax, summary) -> None:
    n_metric = len(METRICS)
    for j, metric in enumerate(METRICS):
        values = {m: oriented(metric, summary[m][metric][0])
                  for m in BASELINES + OURS if metric in summary[m]}
        lo, hi = min(values.values()), max(values.values())
        rng = max(hi - lo, 1e-12)

        ax.axvline(j, color="#f0f0f0", linewidth=0.6, zorder=0)
        for m, v in values.items():
            norm = (v - lo) / rng
            if m in OURS:
                ax.plot(j, norm, marker="o", ms=4.0, color=OURS_COLOR[m],
                        mew=0, zorder=4)
            else:
                ax.plot(j, norm, marker="_", ms=7.0, color="#9aa4ae",
                        mew=1.1, zorder=2)

    ax.set_xticks(np.arange(n_metric))
    ax.set_xticklabels([SHORT_LABEL.get(m, METRIC_LABEL[m]) for m in METRICS],
                       rotation=40, ha="right")
    ax.set_ylabel("Normalised score\n(0 = worst, 1 = best model)", labelpad=3,
                  linespacing=1.45)
    ax.set_xlim(-0.6, n_metric - 0.4)
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(length=1.6, width=0.5, pad=2)

    handles = [plt.Line2D([], [], marker="_", ms=7, mew=1.1, color="#9aa4ae",
                          linestyle="none", label="Tabular baselines")]
    handles += [plt.Line2D([], [], marker="o", ms=4, mew=0,
                           color=OURS_COLOR[m], linestyle="none",
                           label=DISPLAY[m]) for m in OURS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=5, handletextpad=0.3, columnspacing=1.0,
              borderaxespad=0)


def panel_margin(ax, summary) -> None:
    """Best variant minus best baseline, in units of the metric itself."""
    rows: List[Tuple[str, float, float, str]] = []
    for metric in METRICS:
        best_base = max(oriented(metric, summary[m][metric][0]) for m in BASELINES)
        best_ours, best_std, best_name = -np.inf, 0.0, ""
        for m in OURS:
            v = oriented(metric, summary[m][metric][0])
            if v > best_ours:
                best_ours, best_name = v, m
                best_std = summary[m][metric][1] or 0.0
        scale = 100.0 if metric == "gus" else 1.0
        rows.append((metric, (best_ours - best_base) / scale,
                     best_std / scale, best_name))

    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    for yi, (metric, delta, std, name) in zip(y, rows):
        colour = OURS_COLOR[name] if delta > 0 else "#b0b7bd"
        ax.barh(yi, delta, height=0.62, color=colour, edgecolor="none",
                zorder=2)
        if std > 0:
            ax.errorbar(delta, yi, xerr=std, fmt="none", ecolor=INK,
                        elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=3)
        # Clear the error bar, not the bar: the whisker is what the number
        # would otherwise sit on top of, and it is longer than the gap that
        # looks generous against the bar alone.
        edge = delta + (std if delta >= 0 else -std)
        offset = 0.0045 if delta >= 0 else -0.0045
        ax.text(edge + offset, yi, f"{delta:+.3f}",
                va="center", ha="left" if delta >= 0 else "right",
                fontsize=7.5, color=INK, zorder=3)

    ax.axvline(0, color=INK, linewidth=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABEL.get(r[0], METRIC_LABEL[r[0]])
                        for r in rows])
    ax.set_xlabel("Best variant \u2212 best baseline", labelpad=3)
    ax.set_xlim(-0.10, 0.135)
    ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)

    n_win = sum(1 for r in rows if r[1] > 0)
    ax.text(0.98, 0.03, f"{n_win} of {len(rows)} metrics won",
            transform=ax.transAxes, color=INK,
            fontweight="bold", ha="right", va="bottom")


def panel_chain(ax, chain: Dict[str, Dict[str, float]], ours: Dict[str, float]) -> None:
    """Conditional accuracy of each link, ours against the baseline spread.

    Drawn as ranges rather than as lines across the four stages: the step link
    sits above 96% for every model, so a line plot renders as an inverted V
    whose only visible feature is a stage nobody fails.
    """
    stages = [("cartridge_top1_acc", "Cartridge top-1"),
              ("step_top1_acc_given_car", "Step | cartridge"),
              ("solvent_top1_acc_given_prefix", "Solvent | prefix"),
              ("chain_acc_no_ratio", "Full chain")]
    y = np.arange(len(stages))[::-1]

    for yi, (key, _) in zip(y, stages):
        values = [100 * chain[m][key] for m in chain]
        ax.plot([min(values), max(values)], [yi, yi], color="#d6dade",
                linewidth=4.0, solid_capstyle="round", zorder=1)
        for v in values:
            ax.plot(v, yi, marker="|", ms=5.5, mew=0.9, color="#9aa4ae", zorder=2)

    best_base = max(chain, key=lambda m: chain[m]["chain_acc_no_ratio"])
    ax.plot([100 * chain[best_base][k] for k, _ in stages], y, linestyle="none",
            marker="o", ms=3.4, mew=0, color="#5b656e", zorder=3,
            label=f"{DISPLAY[best_base]} (best baseline)")
    ax.plot([100 * ours[k] for k, _ in stages], y, linestyle="none",
            marker="o", ms=4.2, mew=0, color=ACCENT_2, zorder=4,
            label="Ours (bagging)")

    for yi, (key, _) in zip(y, stages):
        ax.text(101.5, yi, f"{100 * ours[key]:.1f}", va="center", ha="left",
                fontsize=8, color=INK, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in stages])
    ax.set_xlim(40, 108)
    ax.set_xticks([40, 60, 80, 100])
    ax.set_xlabel("Accuracy (%)", labelpad=3)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
              handlelength=0.9, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0)


def main() -> None:
    apply_style()
    summary = load_summary()
    chain = load_baseline_chain()

    import json
    from figures.results.data import ENSEMBLE_JSON, ROOT
    with open(os.path.join(ROOT, ENSEMBLE_JSON), encoding="utf-8") as fh:
        ours_chain = json.load(fh)["metrics"]["overall_clean"]

    fig, ax = new_panel(COL2, 2.55)
    panel_normalised(ax, summary)
    save(fig, "fig_benchmark_a", OUT_DIR, target_width=COL2)

    fig, ax = new_panel(COL_PANEL, 3.15)
    panel_margin(ax, summary)
    save(fig, "fig_benchmark_b", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL15, 2.55)
    panel_chain(ax, chain, ours_chain)
    save(fig, "fig_benchmark_c", OUT_DIR, target_width=COL15)

    fig = plt.figure(figsize=(COL2, 3.72))

    ax_a = fig.add_axes([0.105, 0.660, 0.865, 0.275])
    panel_normalised(ax_a, summary)
    label_a = column_label(fig, ax_a, "a", 0.0, dy=1.24)

    ax_b = fig.add_axes([0.105, 0.095, 0.312, 0.320])
    panel_margin(ax_b, summary)
    label_b = column_label(fig, ax_b, "b", 0.0, dy=1.03)

    ax_c = fig.add_axes([0.620, 0.095, 0.365, 0.320])
    panel_chain(ax_c, chain, ours_chain)
    label_c = column_label(fig, ax_c, "c", 0.5, dy=1.03)

    groups = [(label_a, [ax_a]), (label_b, [ax_b]), (label_c, [ax_c])]
    for _ in range(2):
        fit_width(fig, COL2)
        align_labels(fig, groups)

    save(fig, "fig_benchmark", OUT_DIR)


if __name__ == "__main__":
    main()
