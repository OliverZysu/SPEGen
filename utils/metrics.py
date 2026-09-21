from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


def sigmoid_to_pred(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(np.int32)


def multilabel_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute micro precision/recall/F1 and exact match for multi-label.
    y_true, y_pred: (N, L) binary arrays
    """
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)

    exact = float((y_true == y_pred).all(axis=1).mean())

    # "Hit rate": predicted set intersects true set (for each sample)
    hits = []
    for i in range(y_true.shape[0]):
        t = set(np.where(y_true[i] == 1)[0].tolist())
        p = set(np.where(y_pred[i] == 1)[0].tolist())
        hits.append(1.0 if len(t) > 0 and len(t.intersection(p)) > 0 else 0.0)
    hit_rate = float(np.mean(hits)) if len(hits) else 0.0

    # Jaccard average (only for samples where union non-empty)
    jacc = []
    for i in range(y_true.shape[0]):
        t = set(np.where(y_true[i] == 1)[0].tolist())
        p = set(np.where(y_pred[i] == 1)[0].tolist())
        u = len(t.union(p))
        if u == 0:
            continue
        jacc.append(len(t.intersection(p)) / u)
    jaccard = float(np.mean(jacc)) if len(jacc) else 0.0

    return {
        "micro_precision": float(prec),
        "micro_recall": float(rec),
        "micro_f1": float(f1),
        "exact_match_rate": float(exact),
        "hit_rate": float(hit_rate),
        "jaccard": float(jaccard),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def top1_accuracy_from_multihot(y_true: np.ndarray, probs: np.ndarray) -> float:
    """
    y_true: (N, K) multihot
    probs:  (N, K)
    accuracy is 1 if argmax(probs) is in true positives
    """
    if y_true.shape[0] == 0:
        return 0.0
    y_true = y_true.astype(np.int32)
    top1 = probs.argmax(axis=1)
    correct = 0
    for i in range(y_true.shape[0]):
        if y_true[i, top1[i]] == 1:
            correct += 1
    return float(correct / y_true.shape[0])


def compute_pos_weight(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    For BCEWithLogitsLoss pos_weight: neg/pos per class.
    y: (N, L) binary
    """
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    return (neg + eps) / (pos + eps)


def compute_pos_weight_3d(y: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """
    y: (N, A, B)
    returns pos_weight: (A, B)
    """
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    return (neg + eps) / (pos + eps)


def regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Basic regression metrics for ratio prediction.

    This is used for the *ratio* subtask (continuous value in [0,1]).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return {"n": 0}

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    # R^2 (guarded)
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - (np.sum(err ** 2) / (denom + 1e-12)))

    within_0_05 = float(np.mean(np.abs(err) <= 0.05))
    within_0_10 = float(np.mean(np.abs(err) <= 0.10))

    return {
        "n": int(y_true.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "within_0.05": within_0_05,
        "within_0.10": within_0_10,
    }

