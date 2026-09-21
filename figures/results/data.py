"""Loaders that pull experiment numbers straight out of the run directories.

Nothing in ``figures/results`` hard-codes a metric value: rerun the
experiments and the figures follow.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BASELINES = ["DecisionTree", "RandomForest", "SVM", "MLP", "XGBoost", "LightGBM"]

OURS = ["Ours (balanced)", "Ours (chain-oriented)",
        "Ours (exact-match)", "Ours (bagging ensemble)"]

# Short display names, sized for a figure axis rather than a table cell.
DISPLAY = {
    "DecisionTree": "Decision tree",
    "RandomForest": "Random forest",
    "SVM": "SVM",
    "MLP": "MLP",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Ours (balanced)": "Ours: balanced",
    "Ours (chain-oriented)": "Ours: chain",
    "Ours (exact-match)": "Ours: exact-match",
    "Ours (bagging ensemble)": "Ours: bagging",
}

METRIC_LABEL = {
    "car_f1": "Cartridge F1",
    "car_em": "Cartridge exact match",
    "sol_f1": "Solvent F1",
    "ratio_mae": "Ratio MAE",
    "ratio_w10": "Ratio within 0.10",
    "car_top1": "Cartridge top-1",
    "chain": "Chain accuracy",
    "hit1": "Hit@1",
    "sss": "Scheme similarity",
    "hc": "Hierarchical consistency",
    "gus": "Global utility",
}

LOWER_IS_BETTER = {"ratio_mae"}


def load_summary(path: str = "output/paper_tables/all_metrics.csv"
                 ) -> Dict[str, Dict[str, Tuple[float, Optional[float]]]]:
    """``{model: {metric: (mean, std or None)}}`` from the generated table."""
    out: Dict[str, Dict[str, Tuple[float, Optional[float]]]] = {}
    with open(os.path.join(ROOT, path), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            std = row["std"].strip()
            out.setdefault(row["model"], {})[row["metric"]] = (
                float(row["mean"]), float(std) if std else None)
    return out


def load_baseline_chain(path: str = "output/baseline_output/overall_metrics_clean.json"
                        ) -> Dict[str, Dict[str, float]]:
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        raw = json.load(fh)
    return {name: block["clean"] for name, block in raw.items()}


def load_runs(pattern: str) -> List[dict]:
    """All ``test_metrics.json`` payloads matching a glob of run directories.

    Run globs also pick up the sibling ``*.log`` files the training scripts
    leave behind, so anything that is not a directory holding the metrics file
    is skipped rather than parsed.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
        metrics_path = os.path.join(path, "test_metrics.json")
        if not os.path.isdir(path) or not os.path.exists(metrics_path):
            continue
        with open(metrics_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        out.append(payload.get("metrics", payload))
    return out


def dig(payload: dict, path: Sequence[str]) -> Optional[float]:
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else None


def aggregate(pattern: str, path: Sequence[str]) -> Tuple[float, float, int]:
    values = [v for v in (dig(m, path) for m in load_runs(pattern)) if v is not None]
    if not values:
        return float("nan"), float("nan"), 0
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr)


# Run globs for the variants reported in the paper. Kept next to the loaders so
# a renamed experiment directory breaks in exactly one place.
OURS_RUNS: Dict[str, str] = {
    "Ours (balanced)": "output/improve21/s5_C2t08_init*",
    "Ours (chain-oriented)":
        "output/improve21/hp/t08_h512_l2_d0.3_lr0.001_wd0_bs64_init*",
    "Ours (exact-match)": "output/improve21/s8_EMt08_init*",
}

ENSEMBLE_JSON = "output/improve21/ens_bagt08.json"

# Stage-by-stage improvement, in the order the experiments were run.
STAGES: List[Tuple[str, str]] = [
    ("Baseline\n(z-score)", "output/improve21/s1_C12_zscore_init*"),
    ("+ rank-Gauss", "output/improve21/s1_C12_rank_gauss_init*"),
    ("+ tuned\nencoder",
     "output/improve21/hp/t08_h512_l2_d0.3_lr0.001_wd0_bs64_init*"),
]
