"""
Figure: which solvents enter which step, and how the choice tracks the analyte.

Three panels:
  a  Solvent usage conditional on the operational step, P(solvent | step).
  b  Prevalence of the leading elution solvents across logP octiles, with
     Wilson 95% confidence intervals.
  c  Standardised mean difference (Cohen's d) between molecules that do and do
     not receive a given elution solvent, for eight interpretable descriptors.

Panel a describes the workflow grammar (conditioning is methanol/water,
elution is organic, reconstitution mirrors the mobile phase). Panels b and c
describe the analyte-dependent part of the choice, which is what the model has
to learn from descriptors alone.

    python -m figures.correlation.fig_solvent_association
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from figures.style import (  # noqa: E402
    CATEGORICAL, COL15, COL2, COL_PANEL, GRID, INK, SEQ, SPEData, apply_style,
    colorbar, new_panel, panel_label, save,
)
from figures.correlation.fig_cartridge_landscape import (  # noqa: E402
    ENRICH_CMAP, edge_labels, quantile_bins,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ELUTE = 3          # index of 'Elute' in STEP_NAMES
N_SOLVENT = 12
N_BINS = 8
MIN_CELL = 40

# Descriptors chosen for panel c: interpretable, non-redundant, and the ones an
# analytical chemist would actually reason about. Mass-like descriptors are
# represented once (MolecularWeight) because ExactMass, MonoIsotopicWeight and
# HeavyAtomCount are near-collinear with it.
# Tick labels for panel d. The grouped ion-exchange classes carry every member
# name, which is unreadable rotated at this size; the full names stay in Fig. 3.
COMPACT = {
    "MAX, MCX, AX, CX, Screen-A/C": "MAX / MCX / AX / CX",
    "SCX, PRS, BCX, CCX": "SCX / PRS / BCX",
    "WAX, PSA, DEA": "WAX / PSA / DEA",
    "PEP / HRP / DVB": "PEP / HRP / DVB",
}

PANEL_C_DESCRIPTORS = [
    ("LogP_value", "logP"),
    ("TPSA_value", "TPSA"),
    ("MolecularWeight_value", "Mol. weight"),
    ("HBondDonor_value", "HB donors"),
    ("HBondAcceptor_value", "HB acceptors"),
    ("RotatableBond_value", "Rotatable bonds"),
    ("FCSP3_value", "Fsp3"),
    ("Complexity_value", "Complexity"),
]


def wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    """Wilson score interval; behaves sensibly for the sparse tail solvents."""
    k = np.asarray(k, dtype=float)
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def cohens_d(values: np.ndarray, group: np.ndarray) -> float:
    """Pooled-SD standardised mean difference between group==1 and group==0."""
    ok = np.isfinite(values)
    a = values[ok & (group == 1)]
    b = values[ok & (group == 0)]
    if len(a) < 20 or len(b) < 20:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                     / max(len(a) + len(b) - 2, 1))
    if pooled <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def panel_step_usage(ax, data: SPEData, solvents: Sequence[Tuple[int, str, int]]):
    n_steps = len(data.step_names)
    matrix = np.zeros((len(solvents), n_steps))
    for t in range(n_steps):
        present = data.y_step[:, t] == 1
        denom = max(int(present.sum()), 1)
        for r, (si, _, _) in enumerate(solvents):
            matrix[r, t] = data.y_solvent[present, t, si].mean()

    im = ax.imshow(matrix, aspect="auto", cmap=SEQ, vmin=0,
                   vmax=float(matrix.max()), interpolation="nearest")
    ax.set_xticks(np.arange(n_steps))
    ax.set_xticklabels([s.replace("Sample loading", "Loading")
                        for s in data.step_names],
                       rotation=52, ha="right")
    ax.set_yticks(np.arange(len(solvents)))
    ax.set_yticklabels([n for _, n, _ in solvents])

    # Print the rate inside cells that are large enough to matter.
    for r in range(len(solvents)):
        for t in range(n_steps):
            v = matrix[r, t]
            if v < 0.08:
                continue
            ax.text(t, r, f"{100 * v:.0f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if v > 0.55 * matrix.max() else INK)

    ax.set_xticks(np.arange(-0.5, n_steps, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(solvents), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=1.6, width=0.5, pad=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    return im


def panel_logp_curves(ax, data: SPEData, solvents: Sequence[Tuple[int, str, int]]):
    logp = data.descriptor("LogP_value")
    present = data.y_step[:, ELUTE] == 1
    edges, assign = quantile_bins(logp, N_BINS)
    n_bin = len(edges) - 1
    centres = np.arange(n_bin)

    for j, (si, name, _) in enumerate(solvents):
        k = np.zeros(n_bin)
        n = np.zeros(n_bin)
        for b in range(n_bin):
            sel = present & np.isfinite(logp) & (assign == b)
            n[b] = int(sel.sum())
            k[b] = int(data.y_solvent[sel, ELUTE, si].sum())
        p, lo, hi = wilson(k, n)
        colour = CATEGORICAL[j % len(CATEGORICAL)]
        ax.fill_between(centres, 100 * lo, 100 * hi, color=colour, alpha=0.13,
                        linewidth=0)
        ax.plot(centres, 100 * p, color=colour, marker="o", ms=2.4,
                mew=0, label=name, linewidth=1.1)

    ax.set_xticks(centres)
    ax.set_xticklabels(edge_labels(edges), rotation=52, ha="right")
    ax.set_xlabel("logP octile", labelpad=2)
    ax.set_ylabel("Use in the elution step (%)", labelpad=3)
    ax.set_xlim(-0.35, n_bin - 0.65)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=1.6, width=0.5, pad=2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.02),
              handlelength=1.2, labelspacing=0.4, borderaxespad=0,
              title="Elution solvent",
              title_fontproperties={"size": 8, "weight": "bold"})


def _grid_cosmetics(ax, n_col: int, n_row: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=1.6, width=0.5, pad=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def panel_effect_sizes(ax, data: SPEData, solvents: Sequence[Tuple[int, str, int]]):
    present = data.y_step[:, ELUTE] == 1
    matrix = np.full((len(solvents), len(PANEL_C_DESCRIPTORS)), np.nan)
    for r, (si, _, _) in enumerate(solvents):
        group = data.y_solvent[present, ELUTE, si]
        for c, (col, _) in enumerate(PANEL_C_DESCRIPTORS):
            matrix[r, c] = cohens_d(data.descriptor(col)[present], group)

    vmax = 0.8
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=ENRICH_CMAP,
                   norm=norm, interpolation="nearest")
    im.cmap.set_bad("#f2f2f2")

    ax.set_xticks(np.arange(len(PANEL_C_DESCRIPTORS)))
    ax.set_xticklabels([lab for _, lab in PANEL_C_DESCRIPTORS], rotation=52,
                       ha="right")
    ax.set_yticks(np.arange(len(solvents)))
    ax.set_yticklabels([n for _, n, _ in solvents])
    _grid_cosmetics(ax, len(PANEL_C_DESCRIPTORS), len(solvents))
    return im


def panel_cartridge_coupling(ax, data: SPEData,
                             solvents: Sequence[Tuple[int, str, int]],
                             cartridges: Sequence[Tuple[int, str, int]],
                             *, show_ylabels: bool = True):
    """Relative preference of each cartridge for each elution solvent — the
    output-side coupling the decoder has to respect when it emits a cartridge
    and a solvent jointly.

    Two normalisations are wrong here and both were tried first. Raw
    conditional probabilities are flat, because pollutants carry several
    cartridge labels at once and methanol sits near 85% in every column. The
    ratio to the global marginal is uniformly positive, because any cartridge
    label selects protocol-rich pollutants. Centring each row on its own mean
    across cartridges removes both confounds and leaves the contrast of
    interest: given that this solvent is used, which cartridge favours it.
    """
    present = data.y_step[:, ELUTE] == 1
    rates = np.full((len(solvents), len(cartridges)), np.nan)
    for c, (ci, _, _) in enumerate(cartridges):
        sel = present & (data.y_cartridge[:, ci] == 1)
        if int(sel.sum()) < MIN_CELL:
            continue
        for r, (si, _, _) in enumerate(solvents):
            rates[r, c] = data.y_solvent[sel, ELUTE, si].mean()

    row_ref = np.nanmean(rates, axis=1, keepdims=True)
    matrix = np.log2((rates + 1e-3) / (row_ref + 1e-3))

    vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=ENRICH_CMAP,
                   norm=norm, interpolation="nearest")
    im.cmap.set_bad("#f2f2f2")

    ax.set_xticks(np.arange(len(cartridges)))
    ax.set_xticklabels([COMPACT.get(n, n) for _, n, _ in cartridges],
                       rotation=52, ha="right")
    ax.set_yticks(np.arange(len(solvents)))
    if show_ylabels:
        ax.set_yticklabels([n for _, n, _ in solvents])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    _grid_cosmetics(ax, len(cartridges), len(solvents))
    return im


def main() -> None:
    apply_style()
    data = SPEData(".")

    step_solvents = data.top_solvents(N_SOLVENT)
    elute_solvents = data.top_solvents(6, step=ELUTE)
    effect_solvents = data.top_solvents(10, step=ELUTE)

    from figures.correlation.fig_cartridge_landscape import real_cartridges
    cartridges = real_cartridges(data, 8)

    fig, ax = new_panel(COL_PANEL, 3.55)
    im_a = panel_step_usage(ax, data, step_solvents)
    colorbar(fig, im_a, ax, "P(solvent | step present)")
    save(fig, "fig_solvent_association_a", OUT_DIR, target_width=COL_PANEL)

    fig, ax = new_panel(COL15, 2.85)
    panel_logp_curves(ax, data, elute_solvents)
    save(fig, "fig_solvent_association_b", OUT_DIR, target_width=COL15)

    fig, ax = new_panel(COL15, 3.20)
    im_c = panel_effect_sizes(ax, data, effect_solvents)
    colorbar(fig, im_c, ax, "Cohen's $d$  (users \u2212 non-users)",
             ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    save(fig, "fig_solvent_association_c", OUT_DIR, target_width=COL15)

    fig, ax = new_panel(COL15, 3.20)
    im_d = panel_cartridge_coupling(ax, data, effect_solvents, cartridges)
    colorbar(fig, im_d, ax, "log$_2$[ P(solvent | cartridge) / row mean ]",
             ticks=[-1.0, -0.5, 0, 0.5, 1.0])
    save(fig, "fig_solvent_association_d", OUT_DIR, target_width=COL15)

    fig = plt.figure(figsize=(COL2, 3.95))

    ax_a = fig.add_axes([0.112, 0.655, 0.200, 0.312])
    im_a = panel_step_usage(ax_a, data, step_solvents)
    panel_label(ax_a, "a", dx=-0.72, dy=1.04)

    cax_a = fig.add_axes([0.322, 0.655, 0.011, 0.312])
    cb_a = fig.colorbar(im_a, cax=cax_a)
    cb_a.set_label("P(solvent | step present)", fontsize=7.0, labelpad=3)
    cb_a.ax.tick_params(labelsize=6.5, length=1.6, width=0.5, pad=1.5)
    cb_a.outline.set_linewidth(0.4)

    ax_b = fig.add_axes([0.492, 0.655, 0.238, 0.312])
    panel_logp_curves(ax_b, data, elute_solvents)
    panel_label(ax_b, "b", dx=-0.31, dy=1.04)

    ax_c = fig.add_axes([0.112, 0.150, 0.235, 0.345])
    im_c = panel_effect_sizes(ax_c, data, effect_solvents)
    panel_label(ax_c, "c", dx=-0.62, dy=1.03)

    cax_c = fig.add_axes([0.357, 0.150, 0.011, 0.345])
    cb_c = fig.colorbar(im_c, cax=cax_c, ticks=[-0.8, -0.4, 0, 0.4, 0.8])
    cb_c.set_label("Cohen's $d$  (users \u2212 non-users)", fontsize=7.0, labelpad=3)
    cb_c.ax.tick_params(labelsize=6.5, length=1.6, width=0.5, pad=1.5)
    cb_c.outline.set_linewidth(0.4)

    ax_d = fig.add_axes([0.545, 0.150, 0.245, 0.345])
    im_d = panel_cartridge_coupling(ax_d, data, effect_solvents, cartridges,
                                   show_ylabels=False)
    panel_label(ax_d, "d", dx=-0.075, dy=1.03)

    cax_d = fig.add_axes([0.800, 0.150, 0.011, 0.345])
    cb_d = fig.colorbar(im_d, cax=cax_d, ticks=[-1.0, -0.5, 0, 0.5, 1.0])
    cb_d.set_label("log$_2$[ P(solvent | cartridge) / row mean ]",
                   fontsize=7.0, labelpad=3)
    cb_d.ax.tick_params(labelsize=6.5, length=1.6, width=0.5, pad=1.5)
    cb_d.outline.set_linewidth(0.4)

    save(fig, "fig_solvent_association", OUT_DIR)


if __name__ == "__main__":
    main()
