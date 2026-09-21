"""
Figure: corpus construction and the multi-task generator.

  a  A worked example: the published methods linked to one pollutant, and the
     union/mean aggregation that turns them into a single set-valued target.
  b  The four supervision tensors and the statistics of the label space.
  c  The network, the decoder and the ensemble, annotated with tensor shapes
     and with the loss attached to each head.

Everything numeric is read from the corpus and from a trained run, including
the worked example in a and the two insets in c, so the schematic cannot drift
away from the system it describes.

    python -m figures.results.fig_pipeline
"""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from figures.style import (  # noqa: E402
    ACCENT, ACCENT_2, ACCENT_3, COL15, COL2, COL_PANEL, INK, MUTED,
    SPEData, apply_style, column_label, save,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RUN = "output/improve21/hp/t08_h512_l2_d0.3_lr0.001_wd0_bs64_init42"

# Ketobemidone: four independent published methods that agree on the early
# steps and disagree completely on elution, which is the property that forces
# the set-valued formulation.
EXAMPLE_CID = "10101"

STEP_SHORT = ["Load", "Condition", "Wash", "Elute", "Reconstitute"]

# Bench shorthand keeps the example table readable at 7 pt. Subscripts go
# through mathtext because Arial has no subscript glyphs.
ABBREV = {
    "methanol": "MeOH", "water": "H$_2$O", "acetonitrile": "MeCN",
    "ethyl acetate": "EtOAc", "formic acid": "HCOOH",
    "ammonium acetate": "NH$_4$OAc", "ammonium hydroxide": "NH$_4$OH",
    "ammonium formate": "NH$_4$HCO$_2$", "potassium phosphate": "K-phosphate",
    "isopropanol": "IPA", "acetone": "acetone", "hexane": "hexane",
    "phosphate": "phosphate", "toluene": "toluene", "chloroform": "CHCl$_3$",
    "diethyl ether": "Et$_2$O",
}

FILL = {
    "data": "#eef2f6", "input": "#e7eef5", "encoder": "#dbe6f1",
    "head": "#e9f0e9", "decode": "#f7ede6", "output": "#f4ece8",
    "target": "#eef2f6",
}
EDGE = {
    "data": "#9fb2c4", "input": "#7f9db8", "encoder": ACCENT,
    "head": ACCENT_3, "decode": "#c69madd"[:7], "output": ACCENT_2,
    "target": "#9fb2c4",
}
EDGE["decode"] = "#c1906f"


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def box(ax, x, y, w, h, kind="data", lw=0.7, alpha=1.0, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=FILL[kind], edgecolor=EDGE[kind], linewidth=lw,
        alpha=alpha, zorder=zorder))


def arrow(ax, start, end, colour="#96a0aa", rad=0.0, lw=0.9, scale=7.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale,
        connectionstyle=f"arc3,rad={rad}", color=colour, linewidth=lw,
        shrinkA=0, shrinkB=0, zorder=2))


def shorten(token: str) -> str:
    parts = [ABBREV.get(p.strip(), p.strip()) for p in token.split("+")]
    return "+".join(parts)


# --------------------------------------------------------------------------
# data for the worked example
# --------------------------------------------------------------------------

def example_methods() -> Tuple[str, List[List[List[str]]], List[List[str]]]:
    """Per-method step -> solvent tokens for the example, and their union."""
    import pandas as pd

    from spe.data import prepare_data
    from utils.parsing import STEP_NAMES

    proto = pd.read_csv(os.path.join(ROOT, "data/spe_processed_data_new.csv"))

    def first_key(raw: str) -> str:
        try:
            value = ast.literal_eval(raw)
            return value[0] if isinstance(value, list) and value else str(raw)
        except (ValueError, SyntaxError):
            return str(raw)

    proto["key"] = proto["CasMp"].map(first_key)
    proto = proto.drop_duplicates("key").set_index("key")

    prepared = prepare_data(
        data_dir=os.path.join(ROOT, "data"),
        cache_path=os.path.join(ROOT, "data/cache/dataset_cache.pkl"),
        seed=42, train_ratio=0.8, val_ratio=0.1, require_spe_info=False,
        unk_solvent_token="__UNK__", use_functional_groups=False,
        fg_transform="binary", fg_min_pos=10, decompose_solvent=False,
        feature_transform="zscore",
    )
    index = {str(c): i for i, c in enumerate(prepared.ds.cids)}
    row_id = index[EXAMPLE_CID]

    names = pd.read_csv(os.path.join(ROOT, "data/processed_molecular_data.csv"),
                        usecols=["CID", "Pollutant Name"])
    names["cid"] = names["CID"].astype(str).str.extract(r"(\d+)")
    label = names.set_index("cid").loc[EXAMPLE_CID, "Pollutant Name"]
    if not isinstance(label, str):
        label = str(np.asarray(label).ravel()[0])

    methods: List[List[List[str]]] = []
    for key in prepared.ds.method_lists[row_id]:
        if key not in proto.index:
            continue
        record = proto.loc[key]
        cells = []
        for step in STEP_NAMES:
            raw = record[step]
            if not isinstance(raw, str) or raw.strip() in ("", "unknown"):
                cells.append([])
                continue
            tokens = [shorten(t.strip()) for t in raw.split(";")
                      if t.strip() and t.strip() != "unknown"]
            cells.append(tokens)
        if any(cells):
            methods.append(cells)

    union: List[List[str]] = []
    for t in range(len(STEP_NAMES)):
        seen: List[str] = []
        for cells in methods:
            for token in cells[t]:
                if token not in seen:
                    seen.append(token)
        union.append(seen)
    return label, methods, union


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

CELL_FS = 6.3
CELL_LEAD = 1.30


def panel_example(ax, label: str, methods, union, panel_h_in: float) -> None:
    """Row heights follow the cell contents, in inches converted to axes units.

    A fixed row height either clips the union cell, which holds every solvent
    any laboratory used, or wastes half the panel on the sparse rows.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    line = CELL_FS / 72.0 * CELL_LEAD / panel_h_in   # one text line, axes units
    pad = 0.024
    gap = 0.010

    n_step = len(STEP_SHORT)
    left = 0.108
    col_w = (1.0 - left) / n_step

    ax.text(0.0, 0.965, f"Published methods for {label}",
            fontsize=7.6, fontweight="bold", color=INK, va="baseline")
    ax.text(0.0, 0.898,
            f"{len(methods)} independent literature protocols, one pollutant",
            fontsize=6.8, color=INK, va="baseline")

    top = 0.812
    for j, name in enumerate(STEP_SHORT):
        ax.text(left + j * col_w + col_w * 0.48, top + 0.016, name,
                ha="center", va="bottom", fontsize=6.9, fontweight="bold",
                color=INK)

    def cell_text(tokens: Sequence[str]) -> str:
        return "\n".join(tokens) if tokens else "\u2014"

    def draw_row(y_top: float, cells, tag: str, highlight: bool) -> float:
        rows = max(1, max(len(c) for c in cells))
        h = pad + line * rows
        y = y_top - h
        ax.text(0.0, y + h / 2, tag, fontsize=6.8,
                fontweight="bold" if highlight else "normal",
                color=ACCENT_2 if highlight else MUTED, va="center")
        for j in range(n_step):
            x = left + j * col_w
            ax.add_patch(Rectangle(
                (x, y), col_w * 0.955, h,
                facecolor="#faf1ec" if highlight else "#f6f7f9",
                edgecolor=ACCENT_2 if highlight else "#dfe3e8",
                linewidth=0.7 if highlight else 0.5, zorder=2))
            ax.text(x + col_w * 0.478, y + h / 2, cell_text(cells[j]),
                    ha="center", va="center", fontsize=CELL_FS,
                    color=INK if cells[j] else "#b9bfc6", zorder=3,
                    linespacing=CELL_LEAD)
        return y

    cursor = top
    for i, cells in enumerate(methods):
        cursor = draw_row(cursor, cells, f"method {i + 1}", False) - gap

    # A single wide arrow reads as one aggregation step; per-column arrows
    # collided with the cell text and implied five separate operations.
    cursor -= 0.012
    ax.annotate("", xy=(0.52, cursor - 0.030), xytext=(0.52, cursor + 0.004),
                arrowprops=dict(arrowstyle="-|>", color="#c2a08d",
                                linewidth=1.0, mutation_scale=8))
    ax.text(0.535, cursor - 0.013,
            "union over methods; repeated ratios averaged",
            fontsize=6.4, color=ACCENT_2, va="center", ha="left")
    cursor -= 0.042

    bottom = draw_row(cursor, union, "target", True)

    ax.text(0.0, bottom - 0.028,
            "Four laboratories elute the same compound four different ways;\n"
            "scoring against any single one would mark the other three wrong.",
            fontsize=6.4, color=INK, va="top", linespacing=1.5)


def panel_targets(ax, data: SPEData) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.955, "Set-valued supervision", fontsize=7.6,
            fontweight="bold", color=INK, va="baseline")
    ax.text(0.0, 0.878,
            f"{len(data.y_cartridge):,} pollutants \u00b7 "
            f"{int(data.has_info.sum()):,} with protocol annotation",
            fontsize=6.8, color=INK, va="baseline")

    annotated = data.has_info
    rows = [
        ("c", "cartridge set", "{0,1}", "42",
         f"{data.y_cartridge.sum(1).mean():.2f} labels per pollutant"),
        ("m", "step mask", "{0,1}", "5",
         f"{100 * data.y_step[annotated].mean():.0f}% of steps present"),
        ("S", "solvent | step", "{0,1}", "5 \u00d7 378",
         f"{data.solvent_any_step()[annotated].sum(1).mean():.2f} distinct "
         "solvents"),
        ("R", "mixture ratio", "[0,1]", "5 \u00d7 378",
         "39.7% of positive cells supervised"),
    ]

    top, row_h = 0.635, 0.196
    for i, (symbol, name, domain, shape, note) in enumerate(rows):
        y = top - i * row_h
        box(ax, 0.0, y, 1.0, row_h * 0.84, "target", lw=0.6)
        ax.text(0.035, y + row_h * 0.52, f"$\\bf{{{symbol}}}$", fontsize=8.2,
                color=ACCENT, va="center", ha="left")
        ax.text(0.125, y + row_h * 0.56, name, fontsize=6.9,
                fontweight="bold", color=INK, va="center")
        ax.text(0.125, y + row_h * 0.23, note, fontsize=6.3, color=INK,
                va="center")
        ax.text(0.985, y + row_h * 0.41, f"{domain}$^{{{shape}}}$",
                fontsize=6.9, color=INK, va="center", ha="right")


def panel_tail(ax, data: SPEData) -> None:
    """Sorted solvent-token frequency: the long tail the losses must cope with."""
    counts = np.sort(
        data.solvent_any_step()[data.has_info].sum(0))[::-1]
    counts = counts[counts > 0]
    rank = np.arange(1, counts.size + 1)

    ax.plot(rank, counts, color=ACCENT, linewidth=1.0)
    ax.fill_between(rank, 0.7, counts, color=ACCENT, alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max(counts.size, 10))
    ax.set_ylim(0.7, counts.max() * 4.0)
    ax.set_xlabel("solvent token, rank", labelpad=2)
    ax.set_ylabel("pollutants", labelpad=2)
    ax.tick_params(labelsize=7.0, pad=2)

    share = counts[:10].sum() / counts.sum()
    rare = int((counts < 10).sum())
    ax.text(0.04, 0.08,
            f"top 10 tokens: {100 * share:.0f}% of positives\n"
            f"{rare} of {counts.size} tokens used < 10 times",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            color=INK, linespacing=1.4)


def panel_model(ax, thresholds: Dict[str, np.ndarray], data: SPEData) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    mid = 0.545
    row_y, row_h = 0.420, 0.250

    # -- input ------------------------------------------------------------
    box(ax, 0.000, row_y, 0.100, row_h, "input")
    ax.text(0.050, 0.600, "descriptors", ha="center", fontsize=6.9,
            fontweight="bold", color=INK)
    ax.text(0.050, 0.518, "$\\bf{x}\\in\\mathbb{R}^{21}$", ha="center",
            fontsize=7.4, color=INK)
    ax.text(0.050, 0.450, "logP, TPSA, \u2026", ha="center", fontsize=6.1,
            color=MUTED)

    box(ax, 0.136, row_y, 0.125, row_h, "input")
    ax.text(0.198, 0.622, "rank-Gauss", ha="center", fontsize=6.9,
            fontweight="bold", color=INK)
    inset_transform(ax, data, 0.148, 0.436, 0.101, 0.168)

    # -- encoder, drawn as a stack because bagging replicates it ----------
    exw, ex0 = 0.130, 0.297
    for k in range(3):
        off = 0.008 * (2 - k)
        box(ax, ex0 + off, row_y + off, exw, row_h, "encoder",
            alpha=0.5 if k else 1.0, zorder=3 + k)
    ecx = ex0 + exw / 2 + 0.008
    ax.text(ecx, 0.612, "shared encoder", ha="center", fontsize=6.9,
            fontweight="bold", color=INK, zorder=7)
    ax.text(ecx, 0.548, "MLP 512 \u00d7 2, ReLU", ha="center", fontsize=6.2,
            color=MUTED, zorder=7)
    ax.text(ecx, 0.500, "dropout 0.3", ha="center", fontsize=6.2, color=MUTED,
            zorder=7)
    ax.text(ecx, 0.446, "$\\bf{z}\\in\\mathbb{R}^{512}$", ha="center",
            fontsize=6.9, color=INK, zorder=7)

    for x0, x1 in ((0.100, 0.136), (0.261, 0.297)):
        arrow(ax, (x0, mid), (x1, mid))

    # -- heads ------------------------------------------------------------
    heads = [
        ("cartridge", "42 logits", "focal + rank", 0.815),
        ("step mask", "5 logits", "weighted BCE", 0.590),
        ("solvent | step", "5 \u00d7 378 logits", "asymmetric", 0.365),
        ("ratio", "20 buckets", "cross-entropy", 0.140),
    ]
    hx, hw, hh = 0.463, 0.150, 0.175
    for name, shape, loss, y in heads:
        box(ax, hx, y, hw, hh, "head")
        ax.text(hx + hw / 2, y + hh * 0.74, name, ha="center", va="center",
                fontsize=6.8, fontweight="bold", color=INK)
        ax.text(hx + hw / 2, y + hh * 0.45, shape, ha="center", va="center",
                fontsize=6.2, color=INK)
        ax.text(hx + hw / 2, y + hh * 0.16, loss, ha="center", va="center",
                fontsize=6.1, color=ACCENT_3, style="italic")
        arrow(ax, (0.435, mid), (hx, y + hh / 2), rad=-0.16)

    ax.text(ecx, 0.400, "+ neighbour smoothing", ha="center", va="top",
            fontsize=6.1, color=MUTED, zorder=7)

    # -- decoder ----------------------------------------------------------
    dx, dw = 0.649, 0.190
    box(ax, dx, 0.372, dw, 0.366, "decode")
    ax.text(dx + dw / 2, 0.692, "decoder", ha="center", fontsize=6.9,
            fontweight="bold", color=INK)
    ax.text(dx + dw / 2, 0.648, "thresholds tuned on validation", ha="center",
            va="center", fontsize=6.1, color=MUTED)
    inset_thresholds(ax, thresholds, dx + 0.032, 0.462, dw - 0.064, 0.152)
    ax.text(dx + dw / 2, 0.412,
            "$\\hat{S}=\\mathbb{1}[\\,p>\\tau\\,]\\odot\\hat{m}$", ha="center",
            va="center", fontsize=7.0, color=INK)
    for _, _, _, y in heads:
        arrow(ax, (hx + hw, y + hh / 2), (dx, mid), rad=0.16)

    ax.text(0.470, 0.028,
            "\u00d7 8 bootstrap members; head probabilities averaged before "
            "decoding", ha="center", va="center", fontsize=6.3, color=INK)

    # -- output -----------------------------------------------------------
    ox, ow = 0.875, 0.125
    box(ax, ox, row_y, ow, row_h, "output")
    ax.text(ox + ow / 2, 0.600, "SPE protocol", ha="center", fontsize=6.9,
            fontweight="bold", color=INK)
    ax.text(ox + ow / 2, 0.505,
            "cartridge set\nstep-wise solvents\nmixture ratios", ha="center",
            va="center", fontsize=6.2, color=INK, linespacing=1.4)
    arrow(ax, (dx + dw, mid), (ox, mid))


def inset_transform(ax, data: SPEData, x, y, w, h) -> None:
    """Before/after histograms of the descriptor with the heaviest tail."""
    from spe.data import apply_feature_transform

    raw = data.raw.to_numpy(dtype=float)
    zscore, _, _ = apply_feature_transform(raw, data.train_idx, kind="zscore")
    gauss, _, _ = apply_feature_transform(raw, data.train_idx, kind="rank_gauss")
    from scipy import stats
    j = int(np.argmax(stats.kurtosis(zscore, axis=0)))

    sub_h = h / 2 - 0.026
    for k, (values, colour, tag) in enumerate((
            (zscore[:, j], "#aeb5bc", "z-score"),
            (gauss[:, j], ACCENT, "rank-Gauss"))):
        sub_y = y + (1 - k) * (h / 2 + 0.004)
        # A common axis would render the z-scored column as a single spike;
        # each row is scaled to its own support so the shape is visible, and
        # the excess kurtosis is printed to keep the comparison honest.
        counts, edges = np.histogram(values, bins=30,
                                     range=np.percentile(values, [0.3, 99.7]))
        counts = counts / max(counts.max(), 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        span = max(centres[-1] - centres[0], 1e-9)
        # The histograms are squeezed into the left two thirds so the two
        # labels have clear space of their own on the right.
        px = x + (centres - centres[0]) / span * (w * 0.62)
        ax.fill_between(px, sub_y, sub_y + counts * sub_h, color=colour,
                        linewidth=0, zorder=5)
        ax.text(x + w, sub_y + sub_h * 0.86, tag, fontsize=6.0, color=colour,
                ha="right", va="center", zorder=6)


def inset_thresholds(ax, thresholds: Dict[str, np.ndarray], x, y, w, h) -> None:
    values = np.asarray(thresholds["solvent"], dtype=float).ravel()
    counts, edges = np.histogram(values, bins=24, range=(0.15, 0.85))
    counts = counts / max(counts.max(), 1)
    bar_h = h * 0.66
    px = x + (0.5 * (edges[:-1] + edges[1:]) - 0.15) / 0.70 * w
    ax.fill_between(px, y, y + counts * bar_h, color="#c1906f", linewidth=0,
                    zorder=5)
    half = x + (0.5 - 0.15) / 0.70 * w
    ax.plot([half, half], [y, y + bar_h * 1.10], color=INK, linewidth=0.6,
            linestyle=(0, (2, 1.6)), zorder=6)
    ax.text(half + 0.005, y + bar_h * 1.10, "0.5", fontsize=6.0, color=INK,
            ha="left", va="top", zorder=6)
    ax.text(x + w / 2, y + h, f"{values.size:,} solvent thresholds \u03c4",
            fontsize=6.0, color=INK, ha="center", va="top", zorder=6)


def main() -> None:
    apply_style()
    data = SPEData(ROOT)
    label, methods, union = example_methods()
    with open(os.path.join(ROOT, RUN, "tuned_thresholds_per_label.json"),
              encoding="utf-8") as fh:
        thresholds = json.load(fh)

    fig_h_a = 3.15
    fig, ax = plt.subplots(figsize=(COL2 * 0.72, fig_h_a))
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    panel_example(ax, label, methods, union, fig_h_a * 0.96)
    save(fig, "fig_pipeline_a", OUT_DIR, target_width=COL15)

    fig = plt.figure(figsize=(COL_PANEL, 4.35))
    ax_b = fig.add_axes([0.06, 0.38, 0.90, 0.58])
    panel_targets(ax_b, data)
    ax_tail = fig.add_axes([0.18, 0.08, 0.74, 0.26])
    panel_tail(ax_tail, data)
    save(fig, "fig_pipeline_b", OUT_DIR, target_width=COL_PANEL)

    fig, ax = plt.subplots(figsize=(COL2, 2.85))
    ax.set_position([0.01, 0.02, 0.98, 0.96])
    panel_model(ax, thresholds, data)
    save(fig, "fig_pipeline_c", OUT_DIR, target_width=COL2)

    fig_h = 5.00
    top_frac = 0.535

    fig = plt.figure(figsize=(COL2, fig_h))

    ax_a = fig.add_axes([0.035, 0.455, 0.575, top_frac])
    panel_example(ax_a, label, methods, union, top_frac * fig_h)
    column_label(fig, ax_a, "a", 0.000, dy=1.00)

    ax_b = fig.add_axes([0.690, 0.700, 0.298, 0.290])
    panel_targets(ax_b, data)
    column_label(fig, ax_b, "b", 0.642, dy=1.00)

    ax_tail = fig.add_axes([0.740, 0.505, 0.235, 0.120])
    panel_tail(ax_tail, data)

    ax_c = fig.add_axes([0.035, 0.020, 0.953, 0.355])
    panel_model(ax_c, thresholds, data)
    column_label(fig, ax_c, "c", 0.000, dy=1.03)

    save(fig, "fig_pipeline", OUT_DIR)


if __name__ == "__main__":
    main()
