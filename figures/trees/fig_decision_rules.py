"""
Figure: descriptor-level decision rules for cartridge and elution solvent.

Both panels are shallow CART trees fitted on the 21 molecular descriptors, the
same input the neural generator sees. They are read as an interpretability
device, not as a competing predictor: depth is capped at three so the whole
rule set fits on the page.

The multi-label protocol space has no canonical single target, so each tree is
fitted on the unambiguous subset of the corpus:

  a  pollutants for which the literature reports exactly one cartridge class
     (n = 5,126 before the tail classes are dropped);
  b  pollutants for which the literature reports exactly one elution solvent
     (n = 4,129 before the tail classes are dropped).

Restricting to those subsets removes the arbitrary choice of "which of the
reported cartridges is the right answer" and leaves a well-posed multiclass
problem. Accuracy is reported on a held-out 20% split so the rules can be
judged, but the point of the figure is the rule set, not the score.

    python -m figures.trees.fig_decision_rules
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from figures.style import (  # noqa: E402
    COL2, DESCRIPTOR_LABELS, INK, SPEData, apply_style, save,
    short_cartridge,
)
from figures.trees.render import (  # noqa: E402
    class_legend, draw_tree, export_rules, node_counts,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ELUTE = 3
MAX_DEPTH = 3
MIN_LEAF = 60
SEED = 42

EXCLUDE_CARTRIDGE = {"Unknown", "Null Cartridge"}
N_CARTRIDGE_CLASSES = 6
N_SOLVENT_CLASSES = 6

# Warm/cool pairs that stay distinguishable in greyscale and for the common
# forms of colour-vision deficiency.
CLASS_COLORS = [
    "#2f5d8f", "#b4553a", "#4a7c59", "#8a6d9e", "#c9a227", "#4f8a8b",
    "#7d8a99", "#a4543a",
]


def fit_tree(x: np.ndarray, y: np.ndarray, n_classes: int):
    """Fit a depth-capped CART and score it by balanced accuracy.

    Class weighting is necessary rather than cosmetic: C18 and methanol cover
    roughly a third of their respective subsets, and an unweighted depth-3 tree
    spends all of its splits separating the two largest classes. Balanced
    accuracy (mean per-class recall) is the matching metric, with 1/K as the
    chance level.
    """
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y)
    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
        class_weight="balanced", random_state=SEED,
    )
    tree.fit(x_tr, y_tr)
    score = float(balanced_accuracy_score(y_te, tree.predict(x_te)))
    return tree, score, 1.0 / n_classes, node_counts(tree, x, y, n_classes)


def single_cartridge_task(data: SPEData):
    single = data.y_cartridge.sum(axis=1) == 1
    label_idx = np.argmax(data.y_cartridge, axis=1)

    keep_classes: List[int] = []
    for ci, name, _ in data.top_cartridges(N_CARTRIDGE_CLASSES + len(EXCLUDE_CARTRIDGE)):
        if name in EXCLUDE_CARTRIDGE:
            continue
        keep_classes.append(ci)
        if len(keep_classes) >= N_CARTRIDGE_CLASSES:
            break

    mask = single & np.isin(label_idx, keep_classes)
    remap = {ci: k for k, ci in enumerate(keep_classes)}
    y = np.array([remap[i] for i in label_idx[mask]])
    names = [short_cartridge(data.cartridge_cols[ci]) for ci in keep_classes]
    return data.raw.to_numpy()[mask], y, names


def single_solvent_task(data: SPEData):
    present = data.y_step[:, ELUTE] == 1
    single = present & (data.y_solvent[:, ELUTE, :].sum(axis=1) == 1)
    label_idx = np.argmax(data.y_solvent[:, ELUTE, :], axis=1)

    counts = np.bincount(label_idx[single], minlength=len(data.solvent_vocab))
    keep_classes = [int(i) for i in np.argsort(-counts)[:N_SOLVENT_CLASSES]]

    mask = single & np.isin(label_idx, keep_classes)
    remap = {si: k for k, si in enumerate(keep_classes)}
    y = np.array([remap[i] for i in label_idx[mask]])
    names = [data.solvent_vocab[si] for si in keep_classes]
    return data.raw.to_numpy()[mask], y, names


def main() -> None:
    apply_style()
    data = SPEData(".")
    features = [DESCRIPTOR_LABELS[c] for c in data.descriptor_cols]

    x_car, y_car, car_names = single_cartridge_task(data)
    tree_car, acc_car, base_car, counts_car = fit_tree(x_car, y_car, len(car_names))

    x_sol, y_sol, sol_names = single_solvent_task(data)
    tree_sol, acc_sol, base_sol, counts_sol = fit_tree(x_sol, y_sol, len(sol_names))

    # One tree per figure. Stacked in a single 7.2 x 4.0 in block the pair no
    # longer fits alongside the preceding figure on a page, and the resulting
    # third-of-a-page gap costs the manuscript more than the adjacency is
    # worth; the solvent tree carries the same argument and moves to the
    # Appendix.
    def one(tree, names, counts, title, stem, max_leaf_label=None):
        fig = plt.figure(figsize=(COL2, 2.30))
        ax = fig.add_axes([0.015, 0.145, 0.970, 0.720])
        kwargs = {} if max_leaf_label is None else {
            "max_leaf_label": max_leaf_label}
        draw_tree(ax, tree, features, names, CLASS_COLORS, counts=counts,
                  **kwargs)
        ax.set_title(title, fontsize=8.0, color=INK, pad=5, loc="left",
                     x=0.035)
        class_legend(ax, names, CLASS_COLORS, ncol=6, y=-0.075)
        save(fig, stem, OUT_DIR)

    one(tree_car, car_names, counts_car,
        f"Cartridge class  \u00b7  n = {len(y_car):,} single-cartridge "
        f"pollutants  \u00b7  held-out balanced accuracy {acc_car:.2f} "
        f"(chance {base_car:.2f})",
        "fig_decision_rules")
    one(tree_sol, sol_names, counts_sol,
        f"Elution solvent  \u00b7  n = {len(y_sol):,} single-solvent "
        f"protocols  \u00b7  held-out balanced accuracy {acc_sol:.2f} "
        f"(chance {base_sol:.2f})",
        "fig_decision_rules_solvent", max_leaf_label=20)

    rules = os.path.join(OUT_DIR, "decision_rules.txt")
    with open(rules, "w", encoding="utf-8") as fh:
        fh.write("Cartridge class "
                 f"(n = {len(y_car):,}, balanced accuracy {acc_car:.3f})\n"
                 f"{'=' * 72}\n\n")
        fh.write(export_rules(tree_car, features, car_names, counts_car))
        fh.write("\n\n\nElution solvent "
                 f"(n = {len(y_sol):,}, balanced accuracy {acc_sol:.3f})\n"
                 f"{'=' * 72}\n\n")
        fh.write(export_rules(tree_sol, features, sol_names, counts_sol))
        fh.write("\n")
    print(f"[saved] {rules}")


if __name__ == "__main__":
    main()
