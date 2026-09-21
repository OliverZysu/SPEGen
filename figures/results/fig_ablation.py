"""
Figure: what actually moved the numbers.

  a  Cumulative effect of the changes that survived validation, against the
     run-to-run noise floor measured on a fixed configuration.
  b  Changes that were tried and rejected, plotted as a deviation from their
     own control so that the noise band is directly readable.
  c  The precision-recall trade-off that separates the reported variants:
     predicted cartridge set size against precision and recall.

Panel b exists because the search space matters as much as the winner. Four of
the eleven interventions we tried are within noise of their control and three
are worse; reporting only the survivors would misrepresent how much of the
design space is inert.

    python -m figures.results.fig_ablation
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402

from figures.results.data import (  # noqa: E402
    ENSEMBLE_JSON, OURS_RUNS, ROOT, STAGES, aggregate, dig, load_runs,
)
from figures.style import (  # noqa: E402
    ACCENT, ACCENT_2, ACCENT_3, COL15, COL2, COL_PANEL, GRID, INK, MUTED,
    apply_style, column_label, new_panel, save,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET = ("overall_clean", "cartridge_top1_acc")

# Rejected interventions, each paired with the control it should be read
# against. Round-one entries share the pre-transform control; round-two
# entries share the rank-Gauss control.
CONTROL_R1 = "output/screen21/R0_control_seed42"
CONTROL_R2 = "output/improve21/s1_C12_rank_gauss_init42"

REJECTED: List[Tuple[str, str, str]] = [
    ("Piecewise-linear embedding", "output/screen21/R5_ple_seed42", CONTROL_R1),
    ("Periodic embedding", "output/screen21/R6_periodic_seed42", CONTROL_R1),
    ("Uncertainty weighting", "output/screen21/R9_uncertainty_seed42", CONTROL_R1),
    ("Signed log1p transform", "output/improve21/s1_C12_log1p_init42", CONTROL_R2),
    ("Median/IQR transform", "output/improve21/s1_C12_robust_init42", CONTROL_R2),
    ("Rank loss: any-positive", "output/improve21/s3_L1_any_pos_init42", CONTROL_R2),
    ("Rank loss: RankNet", "output/improve21/s3_L2_ranknet_init42", CONTROL_R2),
    ("Rank loss: top-1 hinge", "output/improve21/s3_L3_margin_init42", CONTROL_R2),
]

NOISE_SOURCES = [
    "output/noise21/R0_control_init*",
    "output/noise21/R1_step2_init*",
    "output/improve21/s1_C12_zscore_init*",
    "output/improve21/s1_C2_zscore_init*",
]


def panel_stages(ax) -> None:
    labels: List[str] = []
    means: List[float] = []
    errs: List[float] = []

    for label, pattern in STAGES:
        mean, std, _ = aggregate(pattern, TARGET)
        labels.append(label)
        means.append(100 * mean)
        errs.append(100 * std)

    with open(os.path.join(ROOT, ENSEMBLE_JSON), encoding="utf-8") as fh:
        ens = json.load(fh)
    labels.append("+ bootstrap\nbagging")
    means.append(100 * ens["metrics"]["overall_clean"]["cartridge_top1_acc"])
    errs.append(0.0)

    x = np.arange(len(labels))
    colours = ["#b8c1c9", ACCENT, ACCENT_3, ACCENT_2]
    ax.bar(x, means, width=0.60, color=colours, edgecolor="none", zorder=2)
    ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.7,
                capsize=1.8, capthick=0.7, zorder=3)

    ax.text(0.0, 1.02, "error bars: 1 s.d. over 4 initialisation seeds",
            transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom")

    for xi, (m, e) in enumerate(zip(means, errs)):
        gain = "" if xi == 0 else f"   {m - means[xi - 1]:+.1f}"
        ax.text(xi, m + e + 0.30, f"{m:.1f}{gain}", ha="center", va="bottom",
                fontsize=7.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=1.4)
    ax.set_ylim(54, 63.5)
    ax.set_ylabel("Cartridge top-1 accuracy (%)", labelpad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=1.5)


def panel_rejected(ax) -> None:
    controls: Dict[str, float] = {}
    for _, _, control in REJECTED:
        if control not in controls:
            runs = load_runs(control)
            controls[control] = dig(runs[0], TARGET) if runs else np.nan

    rows = []
    for label, pattern, control in REJECTED:
        runs = load_runs(pattern)
        if not runs:
            continue
        value = dig(runs[0], TARGET)
        if value is None or not np.isfinite(controls[control]):
            continue
        rows.append((label, 100 * (value - controls[control])))
    rows.sort(key=lambda r: r[1])

    # Panel b compares single runs, so the yardstick is the spread of
    # individual runs rather than the standard error of a 4-seed mean. The
    # most pessimistic repeated configuration available sets the band.
    band = 200 * max(aggregate(pattern, TARGET)[1] for pattern in NOISE_SOURCES)

    y = np.arange(len(rows))
    ax.axvspan(-band, band, color=MUTED, alpha=0.10, linewidth=0, zorder=0)
    for yi, (label, delta) in zip(y, rows):
        colour = "#b0b7bd" if abs(delta) <= band else ACCENT_2
        ax.barh(yi, delta, height=0.60, color=colour, edgecolor="none", zorder=2)
        ax.text(delta + (0.12 if delta >= 0 else -0.12), yi, f"{delta:+.1f}",
                va="center", ha="left" if delta >= 0 else "right",
                fontsize=7.5, color=INK, zorder=3)

    ax.axvline(0, color=INK, linewidth=0.7, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Change in cartridge top-1 vs. own control (pp)", labelpad=3)
    ax.set_xlim(-11.0, 3.5)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)
    ax.text(0.0, 1.02, "shaded band = \u00b12 s.d. seed noise",
            transform=ax.transAxes, fontsize=7.5, color=INK, va="bottom")


VARIANT_LABEL = {
    "Ours (balanced)": "Balanced",
    "Ours (chain-oriented)": "Chain-oriented",
    "Ours (exact-match)": "Exact-match",
    "Ours (bagging ensemble)": "Bagging ensemble",
}


def variant_stats() -> List[Dict[str, float]]:
    """Cartridge precision, recall, F1 and set sizes for the four variants."""
    keys = [("cartridge", "micro_precision"), ("cartridge", "micro_recall"),
            ("cartridge", "micro_f1"),
            ("pred_set_size", "cartridge_pred_mean"),
            ("pred_set_size", "cartridge_true_mean"),
            ("pred_set_size", "solvent_pred_mean"),
            ("pred_set_size", "solvent_true_mean")]
    names = ["precision", "recall", "f1", "car_pred", "car_true",
             "sol_pred", "sol_true"]

    out: List[Dict[str, float]] = []
    for name, pattern in OURS_RUNS.items():
        runs = load_runs(pattern)
        if not runs:
            continue
        row = {"label": VARIANT_LABEL[name]}
        for key, field in zip(keys, names):
            values = [dig(m, key) for m in runs]
            values = [v for v in values if v is not None]
            row[field] = float(np.mean(values)) if values else np.nan
        out.append(row)

    with open(os.path.join(ROOT, ENSEMBLE_JSON), encoding="utf-8") as fh:
        ens = json.load(fh)["metrics"]
    row = {"label": VARIANT_LABEL["Ours (bagging ensemble)"]}
    for key, field in zip(keys, names):
        row[field] = dig(ens, key) if dig(ens, key) is not None else np.nan
    out.append(row)
    return out


def panel_tradeoff(ax, rows: List[Dict[str, float]]) -> None:
    """Precision, recall and F1 per variant.

    A precision-recall scatter was the first attempt: the three F1-oriented
    variants land within one marker width of each other, so the labels were
    unreadable and the only legible feature was the exact-match outlier.
    """
    y = np.arange(len(rows))[::-1]
    series = [("recall", "Recall", ACCENT_3, "o"),
              ("precision", "Precision", ACCENT_2, "s"),
              ("f1", "F1", INK, "D")]

    for yi, row in zip(y, rows):
        lo = min(100 * row[k] for k, _, _, _ in series)
        hi = max(100 * row[k] for k, _, _, _ in series)
        ax.plot([lo, hi], [yi, yi], color="#dcdcdc", linewidth=1.0, zorder=1)
        for key, _, colour, marker in series:
            ax.plot(100 * row[key], yi, marker=marker, ms=3.6, mew=0,
                    color=colour, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Cartridge micro-average (%)", labelpad=3)
    ax.set_xlim(15, 80)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)

    handles = [plt.Line2D([], [], marker=mk, ms=3.6, mew=0, color=col,
                          linestyle="none", label=lab)
               for _, lab, col, mk in series]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=3, handletextpad=0.3, columnspacing=1.2,
              borderaxespad=0)


def panel_set_size(ax, rows: List[Dict[str, float]]) -> None:
    """Predicted against reported set size — the source of the F1/EM conflict."""
    y = np.arange(len(rows))[::-1]
    height = 0.30

    ax.barh(y + height / 2, [r["car_pred"] for r in rows], height=height,
            color=ACCENT, edgecolor="none", label="Cartridge, predicted",
            zorder=2)
    ax.barh(y - height / 2, [r["sol_pred"] for r in rows], height=height,
            color="#a9c0d6", edgecolor="none", label="Solvent, predicted",
            zorder=2)

    for yi, row in zip(y, rows):
        for offset, key in ((height / 2, "car_true"), (-height / 2, "sol_true")):
            ax.plot([row[key], row[key]], [yi + offset - height / 2,
                                           yi + offset + height / 2],
                    color=ACCENT_2, linewidth=1.3, zorder=4)
        ax.text(row["car_pred"] + 0.4, yi + height / 2, f"{row['car_pred']:.1f}",
                va="center", ha="left", fontsize=7.5, color=INK)
        ax.text(row["sol_pred"] + 0.4, yi - height / 2, f"{row['sol_pred']:.1f}",
                va="center", ha="left", fontsize=7.5, color=INK)

    ax.plot([], [], color=ACCENT_2, linewidth=1.3, label="Reported size")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Labels per pollutant", labelpad=3)
    ax.set_xlim(0, 30)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=1.6, width=0.5, pad=2)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3,
              handlelength=1.0, handletextpad=0.35,
              columnspacing=1.0, borderaxespad=0)


def main() -> None:
    apply_style()

    rows = variant_stats()

    fig, ax = new_panel(COL_PANEL, 2.85)
    panel_stages(ax)
    save(fig, "fig_ablation_a", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL15, 3.15)
    panel_rejected(ax)
    save(fig, "fig_ablation_b", OUT_DIR, target_width=COL15)

    fig, ax = new_panel(COL_PANEL, 2.55)
    panel_tradeoff(ax, rows)
    save(fig, "fig_ablation_c", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL15, 2.70)
    panel_set_size(ax, rows)
    save(fig, "fig_ablation_d", OUT_DIR, target_width=COL15)

    fig = plt.figure(figsize=(COL2, 4.30))

    left, right = 0.008, 0.418

    ax_a = fig.add_axes([0.088, 0.620, 0.290, 0.310])
    panel_stages(ax_a)
    column_label(fig, ax_a, "a", left, dy=1.10)

    ax_b = fig.add_axes([0.600, 0.620, 0.330, 0.310])
    panel_rejected(ax_b)
    column_label(fig, ax_b, "b", right, dy=1.10)

    ax_c = fig.add_axes([0.140, 0.115, 0.240, 0.290])
    panel_tradeoff(ax_c, rows)
    column_label(fig, ax_c, "c", left, dy=1.10)

    ax_d = fig.add_axes([0.600, 0.115, 0.330, 0.290])
    panel_set_size(ax_d, rows)
    column_label(fig, ax_d, "d", right, dy=1.10)

    save(fig, "fig_ablation", OUT_DIR)


if __name__ == "__main__":
    main()
