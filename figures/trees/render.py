"""
A matplotlib renderer for shallow scikit-learn decision trees.

``sklearn.tree.plot_tree`` is a debugging tool: it prints gini, the raw class
vector and the untruncated feature name into every node, sizes boxes by text
length and lays nodes out on a uniform grid so wide trees overlap. None of that
survives contact with a journal page.

This renderer instead:

* lays nodes out by leaf order (Reingold-Tilford style), so subtrees never
  collide and the drawing stays compact when the tree is unbalanced;
* draws internal nodes as a single line of chemistry, ``logP > 5.19``, with the
  test on the edges rather than inside the box;
* draws leaves as a stacked class bar plus the majority label and support, so
  the reader sees purity without reading numbers;
* scales node width with the fraction of samples routed through it, which makes
  the dominant path visible at a glance.

The public entry point is :func:`draw_tree`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from figures.style import INK  # noqa: E402


@dataclass
class Node:
    node_id: int
    depth: int
    is_leaf: bool
    n_samples: int
    fraction: float
    counts: np.ndarray                      # class histogram at this node
    feature: str = ""
    threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    x: float = 0.0
    y: float = 0.0
    span: Tuple[float, float] = (0.0, 0.0)  # leaf-index range covered


def node_counts(tree, x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Unweighted class histogram per node, shape (n_nodes, n_classes).

    ``tree_.value`` holds *weighted* counts whenever the tree was fitted with
    ``class_weight``, so reading purity off it reports the reweighted corpus
    rather than the real one. Routing the samples through the tree again costs
    nothing at this scale and keeps the leaf annotations literal.
    """
    path = tree.decision_path(x).toarray().astype(bool)
    counts = np.zeros((path.shape[1], n_classes), dtype=float)
    for k in range(n_classes):
        counts[:, k] = path[y == k].sum(axis=0)
    return counts


def build(tree, feature_names: Sequence[str], node_id: int = 0,
          depth: int = 0, total: Optional[int] = None,
          counts_override: Optional[np.ndarray] = None) -> Node:
    """Convert a fitted ``sklearn`` tree into the ``Node`` structure above."""
    t = tree.tree_
    if counts_override is not None:
        counts = counts_override[node_id].astype(float)
        n = int(counts.sum())
    else:
        counts = t.value[node_id][0].astype(float)
        n = int(t.n_node_samples[node_id])
    if total is None:
        total = (int(counts_override[0].sum()) if counts_override is not None
                 else int(t.n_node_samples[0]))
    is_leaf = t.children_left[node_id] == -1

    node = Node(node_id=node_id, depth=depth, is_leaf=is_leaf, n_samples=n,
                fraction=n / max(total, 1), counts=counts)
    if not is_leaf:
        node.feature = feature_names[t.feature[node_id]]
        node.threshold = float(t.threshold[node_id])
        node.left = build(tree, feature_names, int(t.children_left[node_id]),
                          depth + 1, total, counts_override)
        node.right = build(tree, feature_names, int(t.children_right[node_id]),
                           depth + 1, total, counts_override)
    return node


def _layout(node: Node, cursor: List[float]) -> None:
    """Assign x by in-order leaf position; y by depth. Leaves get unit width."""
    if node.is_leaf:
        node.x = cursor[0]
        node.span = (cursor[0], cursor[0])
        cursor[0] += 1.0
        return
    _layout(node.left, cursor)
    _layout(node.right, cursor)
    node.x = 0.5 * (node.left.x + node.right.x)
    node.span = (node.left.span[0], node.right.span[1])


def _walk(node: Node) -> List[Node]:
    out = [node]
    if not node.is_leaf:
        out += _walk(node.left) + _walk(node.right)
    return out


def _fit_width(ax, artist, max_width: float, floor: float = 4.2) -> None:
    """Shrink a text artist until it is no wider than ``max_width`` data units.

    Leaf boxes are laid out on the leaf grid and so have a width the class
    names know nothing about; ``dichloromethane`` overruns an eight-leaf row at
    any fixed size that keeps the four-leaf rows legible. Measuring is the only
    way to keep both.
    """
    renderer = ax.figure.canvas.get_renderer()
    inverse = ax.transData.inverted()
    for _ in range(16):
        box = artist.get_window_extent(renderer=renderer)
        left, _ = inverse.transform((box.x0, box.y0))
        right, _ = inverse.transform((box.x1, box.y0))
        size = artist.get_fontsize()
        if right - left <= max_width or size <= floor:
            return
        artist.set_fontsize(max(floor, size * 0.94))


def _fmt_threshold(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_tree(
    ax,
    tree,
    feature_names: Sequence[str],
    class_names: Sequence[str],
    class_colors: Sequence,
    *,
    counts: Optional[np.ndarray] = None,
    node_h: float = 0.30,
    leaf_h: float = 0.66,
    max_leaf_label: int = 18,
) -> None:
    """Render ``tree`` onto ``ax``. The axes are configured entirely here.

    Pass ``counts`` from :func:`node_counts` whenever the tree was fitted with
    class weights, otherwise the leaf percentages describe the reweighted
    corpus instead of the observed one.

    Each leaf is labelled with the class the tree actually predicts and
    annotated with that class's observed share and its enrichment over the
    corpus prior. Under class weighting the predicted class is frequently not
    the raw majority — a leaf can be 70% methanol and still predict
    dichloromethane because dichloromethane is nine times more concentrated
    there than in the corpus. Printing the raw majority instead would
    contradict the model the figure is supposed to explain; printing the share
    without the enrichment would make every leaf look like a failure.
    """
    root = build(tree, feature_names, counts_override=counts)
    predicted = tree.tree_.value[:, 0, :].argmax(axis=1)
    prior = (counts[0] if counts is not None else tree.tree_.value[0][0])
    prior = np.asarray(prior, dtype=float)
    prior = prior / max(prior.sum(), 1.0)
    _layout(root, [0.0])
    nodes = _walk(root)
    n_leaves = sum(1 for n in nodes if n.is_leaf)
    max_depth = max(n.depth for n in nodes)

    for n in nodes:
        n.y = -float(n.depth)

    # The limits fix the data-to-point scale, and the leaf labels below are
    # sized by measuring against it, so they have to be set first.
    ax.set_xlim(-0.75, n_leaves - 0.25)
    ax.set_ylim(-max_depth - 0.85, 0.42)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # -- edges ------------------------------------------------------------
    # Elbow connectors (down, across, down) read as a hierarchy; straight
    # diagonals turn into a thicket as soon as the tree is unbalanced.
    for n in nodes:
        if n.is_leaf:
            continue
        for child, label in ((n.left, "yes"), (n.right, "no")):
            mid = n.y - 0.5
            width = 0.45 + 2.6 * child.fraction
            ax.plot([n.x, n.x], [n.y - node_h / 2, mid],
                    color="#b9b9b9", lw=width, solid_capstyle="round", zorder=1)
            ax.plot([n.x, child.x], [mid, mid],
                    color="#b9b9b9", lw=width, solid_capstyle="round", zorder=1)
            ax.plot([child.x, child.x],
                    [mid, child.y + (leaf_h if child.is_leaf else node_h) / 2],
                    color="#b9b9b9", lw=width, solid_capstyle="round", zorder=1)
            ax.text(child.x + (-0.06 if label == "yes" else 0.06),
                    mid + 0.055, label, fontsize=6.0, color=INK,
                    ha="right" if label == "yes" else "left", va="bottom",
                    zorder=3)

    # -- internal nodes ---------------------------------------------------
    for n in nodes:
        if n.is_leaf:
            continue
        text = f"{n.feature}  \u2264 {_fmt_threshold(n.threshold)}"
        half_w = 0.030 * len(text) + 0.10
        ax.add_patch(FancyBboxPatch(
            (n.x - half_w, n.y - node_h / 2), 2 * half_w, node_h,
            boxstyle="round,pad=0,rounding_size=0.09",
            facecolor="white", edgecolor=INK, linewidth=0.7, zorder=4))
        ax.text(n.x, n.y, text, ha="center", va="center", fontsize=6.0,
                color=INK, zorder=5)
        # The label sits directly on the vertical connector coming down from
        # the parent, so it needs to punch a hole in it.
        ax.text(n.x, n.y + node_h / 2 + 0.095, f"n = {n.n_samples:,}",
                ha="center", va="bottom", fontsize=5.5, color=INK, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    # -- leaves -----------------------------------------------------------
    for n in nodes:
        if not n.is_leaf:
            continue
        total = max(n.counts.sum(), 1.0)
        share = n.counts / total
        order = np.argsort(-share)
        best = int(predicted[n.node_id])
        lift = share[best] / max(prior[best], 1e-9)

        half_w = 0.465
        x0, y0 = n.x - half_w, n.y - leaf_h / 2

        # Stacked purity bar across the top third of the leaf box. The
        # predicted class is outlined, because under class weighting it is
        # often not the widest segment and an unmarked bar would look as
        # though the label contradicted the data.
        bar_h = 0.18 * leaf_h
        cursor = x0
        for k in order:
            if share[k] <= 0:
                continue
            w = 2 * half_w * share[k]
            ax.add_patch(Rectangle((cursor, y0 + leaf_h - bar_h), w, bar_h,
                                   facecolor=class_colors[k], edgecolor="none",
                                   zorder=5))
            if k == int(predicted[n.node_id]):
                ax.add_patch(Rectangle(
                    (cursor, y0 + leaf_h - bar_h), w, bar_h, facecolor="none",
                    edgecolor=INK, linewidth=0.8, zorder=7))
            cursor += w

        ax.add_patch(FancyBboxPatch(
            (x0, y0), 2 * half_w, leaf_h,
            boxstyle="round,pad=0,rounding_size=0.09",
            facecolor="#fbfbfb", edgecolor="#c8c8c8", linewidth=0.6, zorder=4))

        label = class_names[best]
        if len(label) > max_leaf_label:
            label = label[: max_leaf_label - 1] + "\u2026"
        name = ax.text(n.x, y0 + 0.56 * leaf_h, label, ha="center",
                       va="center", fontsize=6.0, fontweight="bold",
                       color=INK, zorder=6)
        stats = ax.text(n.x, y0 + 0.19 * leaf_h,
                        f"{100 * share[best]:.0f}%  \u00b7  {lift:.1f}\u00d7  "
                        f"\u00b7  n = {n.n_samples:,}",
                        ha="center", va="center", fontsize=5.5, color=INK,
                        zorder=6)
        fit_to = 2 * half_w - 0.09
        _fit_width(ax, name, fit_to)
        _fit_width(ax, stats, fit_to)


def export_rules(tree, feature_names: Sequence[str], class_names: Sequence[str],
                 counts: Optional[np.ndarray] = None) -> str:
    """Root-to-leaf rules as plain text, for the supplementary material."""
    root = build(tree, feature_names, counts_override=counts)
    predicted = tree.tree_.value[:, 0, :].argmax(axis=1)
    prior = np.asarray(counts[0] if counts is not None else tree.tree_.value[0][0],
                       dtype=float)
    prior = prior / max(prior.sum(), 1.0)

    lines: List[str] = []

    def walk(node: Node, conditions: List[str]) -> None:
        if node.is_leaf:
            k = int(predicted[node.node_id])
            share = node.counts[k] / max(node.counts.sum(), 1.0)
            head = " AND ".join(conditions) if conditions else "(root)"
            lines.append(
                f"IF {head}\n"
                f"   THEN {class_names[k]}"
                f"   [n = {node.n_samples}, share = {share:.1%}, "
                f"enrichment = {share / max(prior[k], 1e-9):.1f}x]"
            )
            return
        test = f"{node.feature} <= {_fmt_threshold(node.threshold)}"
        walk(node.left, conditions + [test])
        walk(node.right, conditions + [f"NOT ({test})"])

    walk(root, [])
    return "\n\n".join(lines)


def class_legend(ax, class_names: Sequence[str], class_colors: Sequence,
                 ncol: int = 4, y: float = -0.02) -> None:
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
               for c in class_colors]
    ax.legend(handles, list(class_names), loc="upper center",
              bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=6.0,
              handlelength=1.0, handleheight=0.8, columnspacing=1.1,
              labelspacing=0.3, borderaxespad=0)
