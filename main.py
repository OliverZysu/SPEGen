import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from model import SPEBaselineModel, batch_graph_smoothness
from utils.dataloader import build_datasets, build_loaders, split_indices
from utils.io import ensure_dir, save_json, save_jsonl
from utils.metrics import (
    compute_pos_weight,
    compute_pos_weight_3d,
    multilabel_stats,
    regression_stats,
    sigmoid_to_pred,
)
from utils.parsing import STEP_NAMES, build_method_step_map_subset_from_csv
from utils.seed import set_seed


def standardize_inplace(xs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    xs -= mean[None, :]
    xs /= (std[None, :] + 1e-8)


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


def safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    return float(v) if isinstance(v, (int, float)) else float(default)


def apply_solvent_step_calibration(
    sol_prob: np.ndarray,
    step_prob: np.ndarray,
    beta: float = 0.0,
) -> np.ndarray:
    """Calibrate solvent probabilities by step confidence: p(sol|step) * p(step)^beta."""
    b = float(beta)
    if b <= 0.0:
        return sol_prob
    step_factor = np.power(np.clip(step_prob, 1e-6, 1.0), b).astype(np.float32)
    return sol_prob * step_factor[:, :, None]


def _apply_solvent_topk_mask(
    sol_pred: np.ndarray,
    sol_prob: np.ndarray,
    topk_per_step: int,
) -> np.ndarray:
    """Keep at most top-k predicted solvents for each (sample, step)."""
    k = int(topk_per_step)
    if k <= 0:
        return sol_pred
    out = sol_pred.copy()
    n, n_steps, _ = out.shape
    for i in range(n):
        for j in range(n_steps):
            pos = np.where(out[i, j] == 1)[0]
            if pos.size <= k:
                continue
            keep_idx = pos[np.argsort(-sol_prob[i, j, pos])[:k]]
            out[i, j, :] = 0
            out[i, j, keep_idx] = 1
    return out


def build_solvent_pred(
    sol_prob: np.ndarray,
    step_prob: np.ndarray,
    step_thr_arr: np.ndarray,
    sol_thr_arr: np.ndarray,
    beta: float = 0.0,
    topk_per_step: int = 0,
    gate_by_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build solvent predictions with step-calibration and optional per-step top-k pruning.
    Returns (sol_pred, calibrated_sol_prob).
    """
    sol_prob_adj = apply_solvent_step_calibration(sol_prob, step_prob, beta=beta)
    sol_pred = (sol_prob_adj >= sol_thr_arr[None, :, :]).astype(np.int32)
    if gate_by_step:
        step_mask = (step_prob >= step_thr_arr[None, :])[:, :, None]
        sol_pred = (sol_pred * step_mask.astype(np.int32)).astype(np.int32)
    sol_pred = _apply_solvent_topk_mask(sol_pred, sol_prob_adj, topk_per_step=topk_per_step)
    return sol_pred, sol_prob_adj


def parse_int_list_csv(text: str) -> List[int]:
    vals: List[int] = []
    for x in str(text).split(","):
        x = x.strip()
        if x == "":
            continue
        vals.append(int(x))
    if len(vals) == 0:
        vals = [0]
    return sorted(set(vals))


def _row_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample Jaccard for binary multihot arrays (N, L)."""
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    inter = np.logical_and(y_true == 1, y_pred == 1).sum(axis=1).astype(np.float64)
    uni = np.logical_or(y_true == 1, y_pred == 1).sum(axis=1).astype(np.float64)
    out = np.ones((y_true.shape[0],), dtype=np.float64)
    m = uni > 0
    out[m] = inter[m] / uni[m]
    return out


def compute_overall_metrics(
    car_true: np.ndarray,
    car_pred: np.ndarray,
    car_prob: np.ndarray,
    step_true: np.ndarray,
    step_pred: np.ndarray,
    sol_true: np.ndarray,
    sol_pred: np.ndarray,
    step_prob: np.ndarray,
    sol_prob_adj: np.ndarray,
    has_info_all: np.ndarray,
    ratio_pair_records: List[Tuple[int, int, int, float, float]],
    ratio_tau: float = 0.10,
    w_car: float = 0.2,
    w_step: float = 0.2,
    w_sol: float = 0.4,
    w_ratio: float = 0.2,
) -> Dict[str, float]:
    """Compute 5 overall end-to-end indicators."""
    n = car_true.shape[0]
    has_info = has_info_all.astype(np.int32) == 1

    # Aggregate ratio information per sample and per (sample, step, solvent) pair.
    ratio_ok_per_sample = np.ones((n,), dtype=np.float64)
    ratio_score_per_sample = np.ones((n,), dtype=np.float64)
    pair_abs_err: Dict[Tuple[int, int, int], List[float]] = {}
    sample_abs_err: Dict[int, List[float]] = {}
    for sid, st, so, y_t, y_p in ratio_pair_records:
        e = abs(float(y_p) - float(y_t))
        pair_abs_err.setdefault((int(sid), int(st), int(so)), []).append(e)
        sample_abs_err.setdefault(int(sid), []).append(e)
    for sid, errs in sample_abs_err.items():
        errs_arr = np.asarray(errs, dtype=np.float64)
        ratio_ok_per_sample[sid] = float(np.all(errs_arr <= ratio_tau))
        ratio_score_per_sample[sid] = float(np.mean(np.maximum(0.0, 1.0 - errs_arr / ratio_tau)))

    # 1) Strict end-to-end accuracy.
    car_exact = np.all(car_true == car_pred, axis=1)
    step_exact = np.all(step_true == step_pred, axis=1)
    sol_exact = np.all(sol_true == sol_pred.reshape(sol_true.shape), axis=(1, 2))
    strict_ok = np.ones((n,), dtype=bool)
    strict_ok &= car_exact
    strict_ok[has_info] &= step_exact[has_info]
    strict_ok[has_info] &= sol_exact[has_info]
    strict_ok[has_info] &= (ratio_ok_per_sample[has_info] >= 0.5)
    strict_end_to_end_acc = float(np.mean(strict_ok.astype(np.float64)))

    # 2) Scheme similarity score (continuous).
    s_car = _row_jaccard(car_true, car_pred)
    s_step = _row_jaccard(step_true, step_pred)
    s_sol = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        if not has_info[i]:
            s_sol[i] = 1.0
            continue
        step_scores = []
        for j in range(sol_true.shape[1]):
            yt = sol_true[i, j]
            yp = sol_pred[i, j]
            inter = float(np.logical_and(yt == 1, yp == 1).sum())
            uni = float(np.logical_or(yt == 1, yp == 1).sum())
            step_scores.append(1.0 if uni == 0 else inter / uni)
        s_sol[i] = float(np.mean(step_scores)) if len(step_scores) > 0 else 1.0
    sss = float(
        np.mean(
            w_car * s_car
            + w_step * np.where(has_info, s_step, 1.0)
            + w_sol * np.where(has_info, s_sol, 1.0)
            + w_ratio * np.where(has_info, ratio_score_per_sample, 1.0)
        )
    )

    # 3) Hierarchical consistency accuracy.
    hc = np.ones((n,), dtype=np.float64)
    for i in range(n):
        if not has_info[i]:
            hc[i] = 1.0
            continue
        step_scores = []
        for j in range(step_true.shape[1]):
            has_sol_pred = bool(np.any(sol_pred[i, j] == 1))
            if step_true[i, j] == 0:
                step_scores.append(1.0 if not has_sol_pred else 0.0)
            else:
                step_scores.append(1.0 if has_sol_pred else 0.0)
        hc[i] = float(np.mean(step_scores)) if len(step_scores) > 0 else 1.0
    hierarchical_consistency_acc = float(np.mean(hc))

    # 4) Method top-1 hit@1 (approximation).
    top1_car = np.argmax(car_prob, axis=1)
    top1_step = np.argmax(step_prob, axis=1)
    hit = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        ok = bool(car_true[i, top1_car[i]] == 1)
        if has_info[i]:
            st = int(top1_step[i])
            so = int(np.argmax(sol_prob_adj[i, st]))
            ok = ok and bool(step_true[i, st] == 1) and bool(sol_true[i, st, so] == 1)
            errs = pair_abs_err.get((int(i), st, so), None)
            if errs is not None and len(errs) > 0:
                ok = ok and bool(np.mean(np.asarray(errs, dtype=np.float64)) <= ratio_tau)
        hit[i] = 1.0 if ok else 0.0
    method_hit_at_1 = float(np.mean(hit))

    # 5) Global utility score (0-100).
    car_f1 = safe_get(multilabel_stats(car_true, car_pred), "micro_f1", 0.0)
    step_f1 = safe_get(multilabel_stats(step_true[has_info], step_pred[has_info]), "micro_f1", 0.0) if has_info.any() else 0.0
    n2, a2, b2 = sol_true.shape
    sol_true_flat = sol_true.reshape(n2, a2 * b2)
    sol_pred_flat = sol_pred.reshape(n2, a2 * b2)
    sol_f1 = safe_get(multilabel_stats(sol_true_flat[has_info], sol_pred_flat[has_info]), "micro_f1", 0.0) if has_info.any() else 0.0
    ratio_within = float(np.mean(np.asarray([e <= ratio_tau for errs in sample_abs_err.values() for e in errs], dtype=np.float64))) if len(sample_abs_err) > 0 else 0.0
    global_utility_score = float(100.0 * (0.25 * car_f1 + 0.25 * step_f1 + 0.35 * sol_f1 + 0.15 * ratio_within))

    return {
        "strict_end_to_end_acc": strict_end_to_end_acc,
        "scheme_similarity_score": sss,
        "hierarchical_consistency_acc": hierarchical_consistency_acc,
        "method_hit_at_1": method_hit_at_1,
        "global_utility_score": global_utility_score,
    }


@torch.no_grad()
def collect_eval_tensors(
    model: SPEBaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    solvent_step_beta: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Collect true/prob arrays for threshold tuning and model selection."""
    model.eval()
    all_car_true, all_car_prob = [], []
    all_step_true, all_step_prob = [], []
    all_sol_true, all_sol_prob = [], []
    all_has_info = []

    for batch in loader:
        x = batch["x"].to(device)
        y_car = batch["y_cartridge"].cpu().numpy()
        y_step = batch["y_step"].cpu().numpy()
        y_sol = batch["y_solvent"].cpu().numpy()
        has_info = batch["has_spe_info"].cpu().numpy().astype(np.int32)

        out = model(x)
        car_prob = torch.sigmoid(out["cartridge_logits"]).cpu().numpy()
        step_prob = torch.sigmoid(out["step_logits"]).cpu().numpy()
        sol_prob = torch.sigmoid(out["solvent_logits"]).cpu().numpy()
        sol_prob = apply_solvent_step_calibration(sol_prob, step_prob, beta=solvent_step_beta)

        all_car_true.append(y_car); all_car_prob.append(car_prob)
        all_step_true.append(y_step); all_step_prob.append(step_prob)
        all_sol_true.append(y_sol); all_sol_prob.append(sol_prob)
        all_has_info.append(has_info)

    car_true = np.concatenate(all_car_true, axis=0)
    car_prob = np.concatenate(all_car_prob, axis=0)
    step_true = np.concatenate(all_step_true, axis=0)
    step_prob = np.concatenate(all_step_prob, axis=0)
    sol_true = np.concatenate(all_sol_true, axis=0)
    sol_prob = np.concatenate(all_sol_prob, axis=0)
    has_info_all = np.concatenate(all_has_info, axis=0)
    return {
        "car_true": car_true,
        "car_prob": car_prob,
        "step_true": step_true,
        "step_prob": step_prob,
        "sol_true": sol_true,
        "sol_prob": sol_prob,
        "has_info": has_info_all,
    }


def tune_thresholds_on_val(
    model: SPEBaselineModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    init_thresholds: Dict[str, float],
    solvent_step_beta: float = 0.0,
    grid_min: float = 0.1,
    grid_max: float = 0.9,
    grid_num: int = 17,
) -> Dict[str, float]:
    """Tune per-task thresholds on validation set by maximizing micro-F1."""
    t = collect_eval_tensors(
        model=model,
        loader=val_loader,
        device=device,
        solvent_step_beta=solvent_step_beta,
    )
    grid = np.linspace(grid_min, grid_max, num=grid_num, dtype=np.float64)

    best = dict(init_thresholds)

    # Cartridge
    best_f1 = -1.0
    for thr in grid:
        pred = sigmoid_to_pred(t["car_prob"], float(thr))
        f1 = safe_get(multilabel_stats(t["car_true"], pred), "micro_f1", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best["cartridge"] = float(thr)

    # Step / Solvent: only has_spe_info == 1
    mask = t["has_info"] == 1
    if mask.sum() > 0:
        best_f1 = -1.0
        for thr in grid:
            pred = sigmoid_to_pred(t["step_prob"][mask], float(thr))
            f1 = safe_get(multilabel_stats(t["step_true"][mask], pred), "micro_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best["step"] = float(thr)

        n, a, b = t["sol_true"].shape
        sol_true_flat = t["sol_true"].reshape(n, a * b)[mask]
        sol_prob_flat = t["sol_prob"].reshape(n, a * b)[mask]
        best_f1 = -1.0
        for thr in grid:
            pred = sigmoid_to_pred(sol_prob_flat, float(thr))
            f1 = safe_get(multilabel_stats(sol_true_flat, pred), "micro_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best["solvent"] = float(thr)

    return best


def tune_solvent_step_beta_on_val(
    model: SPEBaselineModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    init_thresholds: Dict[str, float],
    beta_grid_min: float = 0.0,
    beta_grid_max: float = 2.0,
    beta_grid_num: int = 9,
    thr_grid_min: float = 0.1,
    thr_grid_max: float = 0.9,
    thr_grid_num: int = 17,
    topk_choices: Optional[List[int]] = None,
    gate_by_step: bool = False,
) -> Tuple[float, float, int, float]:
    """Tune solvent (beta, threshold, topk) by validation micro-F1."""
    beta_grid = np.linspace(beta_grid_min, beta_grid_max, num=beta_grid_num, dtype=np.float64)
    thr_grid = np.linspace(thr_grid_min, thr_grid_max, num=thr_grid_num, dtype=np.float64)
    if topk_choices is None or len(topk_choices) == 0:
        topk_choices = [0]

    best_beta = float(beta_grid[0]) if beta_grid.size > 0 else 0.0
    best_thr = float(init_thresholds["solvent"])
    best_topk = int(topk_choices[0])
    best_f1 = -1.0

    # use current step threshold to gate solvent predictions
    step_thr_arr = np.full((len(STEP_NAMES),), float(init_thresholds["step"]), dtype=np.float32)
    for beta in beta_grid:
        t = collect_eval_tensors(
            model=model,
            loader=val_loader,
            device=device,
            solvent_step_beta=0.0,
        )
        mask = t["has_info"] == 1
        if mask.sum() <= 0:
            continue
        step_prob_m = t["step_prob"][mask]
        sol_prob_m = t["sol_prob"][mask]
        sol_true_m = t["sol_true"][mask]
        for thr in thr_grid:
            sol_thr_arr = np.full((sol_true_m.shape[1], sol_true_m.shape[2]), float(thr), dtype=np.float32)
            for topk in topk_choices:
                sol_pred_m, _ = build_solvent_pred(
                    sol_prob=sol_prob_m,
                    step_prob=step_prob_m,
                    step_thr_arr=step_thr_arr,
                    sol_thr_arr=sol_thr_arr,
                    beta=float(beta),
                    topk_per_step=int(topk),
                    gate_by_step=gate_by_step,
                )
                n, a, b = sol_true_m.shape
                sol_true_flat = sol_true_m.reshape(n, a * b)
                sol_pred_flat = sol_pred_m.reshape(n, a * b)
                f1 = safe_get(multilabel_stats(sol_true_flat, sol_pred_flat), "micro_f1", 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_beta = float(beta)
                    best_thr = float(thr)
                    best_topk = int(topk)

    return best_beta, best_thr, best_topk, best_f1


def _best_thr_for_binary_label(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray,
    fallback: float,
    min_pos: int = 3,
) -> float:
    """Find threshold maximizing F1 for one binary label."""
    y_true = y_true.astype(np.int32)
    pos = int(y_true.sum())
    n = int(y_true.shape[0])
    if n == 0 or pos < min_pos or pos == n:
        return float(fallback)

    best_thr = float(fallback)
    best_f1 = -1.0
    for thr in grid:
        pred = (y_prob >= float(thr)).astype(np.int32)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def tune_thresholds_per_label_on_val(
    model: SPEBaselineModel,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    init_thresholds: Dict[str, float],
    solvent_step_beta: float = 0.0,
    grid_min: float = 0.1,
    grid_max: float = 0.9,
    grid_num: int = 21,
    min_pos_per_label: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Tune per-label thresholds on validation set by label-wise F1 maximization.
    Returns arrays:
      - cartridge: (C,)
      - step: (5,)
      - solvent: (5, S)
    """
    t = collect_eval_tensors(
        model=model,
        loader=val_loader,
        device=device,
        solvent_step_beta=solvent_step_beta,
    )
    grid = np.linspace(grid_min, grid_max, num=grid_num, dtype=np.float64)

    # Cartridge
    n_car = t["car_true"].shape[1]
    car_thr = np.full((n_car,), float(init_thresholds["cartridge"]), dtype=np.float32)
    for k in range(n_car):
        car_thr[k] = _best_thr_for_binary_label(
            y_true=t["car_true"][:, k],
            y_prob=t["car_prob"][:, k],
            grid=grid,
            fallback=float(init_thresholds["cartridge"]),
            min_pos=min_pos_per_label,
        )

    # Step / Solvent only on has_spe_info == 1
    mask = t["has_info"] == 1
    n_steps = t["step_true"].shape[1]
    step_thr = np.full((n_steps,), float(init_thresholds["step"]), dtype=np.float32)
    sol_thr = np.full((t["sol_true"].shape[1], t["sol_true"].shape[2]), float(init_thresholds["solvent"]), dtype=np.float32)
    if mask.sum() > 0:
        step_true_m = t["step_true"][mask]
        step_prob_m = t["step_prob"][mask]
        for k in range(n_steps):
            step_thr[k] = _best_thr_for_binary_label(
                y_true=step_true_m[:, k],
                y_prob=step_prob_m[:, k],
                grid=grid,
                fallback=float(init_thresholds["step"]),
                min_pos=min_pos_per_label,
            )

        sol_true_m = t["sol_true"][mask]
        sol_prob_m = t["sol_prob"][mask]
        for j in range(sol_true_m.shape[1]):
            for s in range(sol_true_m.shape[2]):
                sol_thr[j, s] = _best_thr_for_binary_label(
                    y_true=sol_true_m[:, j, s],
                    y_prob=sol_prob_m[:, j, s],
                    grid=grid,
                    fallback=float(init_thresholds["solvent"]),
                    min_pos=min_pos_per_label,
                )

    return {
        "cartridge": car_thr,
        "step": step_thr,
        "solvent": sol_thr,
    }


def build_selection_score(
    metrics: Dict[str, Any],
    w_cartridge: float = 1.0,
    w_step: float = 1.0,
    w_solvent: float = 1.0,
    w_ratio: float = 0.25,
) -> float:
    """Higher is better. Uses F1 and ratio-within-0.10 as a bounded score."""
    car = safe_get(metrics.get("cartridge", {}), "micro_f1", 0.0)
    step = safe_get(metrics.get("step", {}), "micro_f1", 0.0)
    solvent = safe_get(metrics.get("solvent", {}), "micro_f1", 0.0)
    ratio = safe_get(metrics.get("ratio", {}), "within_0.10", 0.0)
    return (
        w_cartridge * car +
        w_step * step +
        w_solvent * solvent +
        w_ratio * ratio
    )


def classification_loss_elements(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Element-wise multilabel loss (BCE or focal-BCE)."""
    bce_elem = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if loss_type == "bce":
        return bce_elem
    # Focal modulation on top of BCEWithLogits (pos_weight still respected).
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(focal_gamma)
    return bce_elem * focal_factor


def compute_ratio_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 0.05,
    mix_mae_weight: float = 0.3,
) -> torch.Tensor:
    """Ratio regression loss: mse | huber | mse_mae."""
    if loss_type == "huber":
        return F.huber_loss(y_pred, y_true, reduction="mean", delta=huber_delta)
    if loss_type == "mse_mae":
        w = float(min(max(mix_mae_weight, 0.0), 1.0))
        mse = F.mse_loss(y_pred, y_true, reduction="mean")
        mae = F.l1_loss(y_pred, y_true, reduction="mean")
        return (1.0 - w) * mse + w * mae
    return F.mse_loss(y_pred, y_true, reduction="mean")


def _is_finite_scalar_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


@torch.no_grad()
def eval_loss(
    model: SPEBaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pos_weights: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    loss_cfg: Dict[str, Any],
    smooth_topk: int,
) -> Dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "cartridge": 0.0, "step": 0.0, "solvent": 0.0, "ratio": 0.0, "smooth": 0.0, "gate": 0.0}
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_car = batch["y_cartridge"].to(device)
        y_step = batch["y_step"].to(device)
        y_sol = batch["y_solvent"].to(device)
        has_info = batch["has_spe_info"].to(device).view(-1, 1)  # (B,1)
        ratio_pairs = batch["ratio_pairs"]

        out = model(x)
        h = out["h"]
        car_logits = out["cartridge_logits"]
        step_logits = out["step_logits"]
        sol_logits = out["solvent_logits"]

        loss_car_elem = classification_loss_elements(
            logits=car_logits,
            targets=y_car,
            pos_weight=pos_weights["cartridge"],
            loss_type=loss_cfg["cartridge"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
        loss_car = loss_car_elem.mean()

        loss_step_elem = classification_loss_elements(
            logits=step_logits,
            targets=y_step,
            pos_weight=pos_weights["step"],
            loss_type=loss_cfg["step"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
        loss_step = (loss_step_elem * has_info).sum() / (has_info.sum() * y_step.size(1) + 1e-6)

        has_info_3d = has_info.view(-1, 1, 1)
        step_weight = (y_step * loss_cfg["solvent_pos_step_boost"] + (1.0 - y_step) * loss_cfg["solvent_neg_step_weight"]).unsqueeze(-1)
        loss_sol_elem = classification_loss_elements(
            logits=sol_logits,
            targets=y_sol,
            pos_weight=pos_weights["solvent"],
            loss_type=loss_cfg["solvent"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
        sol_weight = has_info_3d * step_weight
        loss_sol = (loss_sol_elem * sol_weight).sum() / (sol_weight.sum() * y_sol.size(2) + 1e-6)

        # ratio regression loss only on (step, solvent) pairs with known numeric ratio
        sample_ids, step_ids, solvent_ids, y_ratio = [], [], [], []
        for i in range(len(ratio_pairs)):
            if has_info[i].item() != 1.0:
                continue
            for sj, sid, r in ratio_pairs[i]:
                rv = float(r)
                if not np.isfinite(rv):
                    continue
                sample_ids.append(i)
                step_ids.append(sj)
                solvent_ids.append(sid)
                y_ratio.append(rv)

        if len(sample_ids) > 0:
            sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
            step_ids_t = torch.tensor(step_ids, device=device, dtype=torch.long)
            solvent_ids_t = torch.tensor(solvent_ids, device=device, dtype=torch.long)
            y_ratio_t = torch.tensor(np.asarray(y_ratio, dtype=np.float32), device=device, dtype=torch.float32)
            h_pairs = h[sample_ids_t]
            ratio_pred = model.ratio_pred(h_pairs, step_ids_t, solvent_ids_t)
            loss_ratio = compute_ratio_loss(
                y_pred=ratio_pred,
                y_true=y_ratio_t,
                loss_type=loss_cfg["ratio_loss"],
                huber_delta=loss_cfg["ratio_huber_delta"],
                mix_mae_weight=loss_cfg["ratio_mix_mae_weight"],
            )
        else:
            loss_ratio = torch.tensor(0.0, device=device)

        loss_smooth = batch_graph_smoothness(x=x, h=h, topk=smooth_topk)

        if hasattr(model, "get_solvent_gate_alpha"):
            gate_alpha = model.get_solvent_gate_alpha()
            loss_gate = ((gate_alpha - 1.0) ** 2).mean()
        else:
            loss_gate = torch.tensor(0.0, device=device)

        loss = (
            lambdas["cartridge"] * loss_car +
            lambdas["step"] * loss_step +
            lambdas["solvent"] * loss_sol +
            lambdas["ratio"] * loss_ratio +
            lambdas["smooth"] * loss_smooth +
            lambdas["gate"] * loss_gate
        )

        if not all(
            _is_finite_scalar_tensor(v)
            for v in (loss_car, loss_step, loss_sol, loss_ratio, loss_smooth, loss_gate, loss)
        ):
            continue

        total["loss"] += float(loss.item())
        total["cartridge"] += float(loss_car.item())
        total["step"] += float(loss_step.item())
        total["solvent"] += float(loss_sol.item())
        total["ratio"] += float(loss_ratio.item())
        total["smooth"] += float(loss_smooth.item())
        total["gate"] += float(loss_gate.item())
        n_batches += 1

    if n_batches == 0:
        for k in total:
            total[k] = float("nan")
    else:
        for k in total:
            total[k] /= n_batches
    return total


def train_one_epoch(
    model: SPEBaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    pos_weights: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    loss_cfg: Dict[str, Any],
    smooth_topk: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total = {"loss": 0.0, "cartridge": 0.0, "step": 0.0, "solvent": 0.0, "ratio": 0.0, "smooth": 0.0, "gate": 0.0}
    n_batches = 0

    non_finite_skipped = 0
    non_finite_grad_skipped = 0
    for batch in loader:
        x = batch["x"].to(device)
        y_car = batch["y_cartridge"].to(device)
        y_step = batch["y_step"].to(device)
        y_sol = batch["y_solvent"].to(device)
        has_info = batch["has_spe_info"].to(device).view(-1, 1)  # (B,1)
        ratio_pairs = batch["ratio_pairs"]

        out = model(x)
        h = out["h"]
        car_logits = out["cartridge_logits"]
        step_logits = out["step_logits"]
        sol_logits = out["solvent_logits"]

        loss_car_elem = classification_loss_elements(
            logits=car_logits,
            targets=y_car,
            pos_weight=pos_weights["cartridge"],
            loss_type=loss_cfg["cartridge"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
        loss_car = loss_car_elem.mean()

        loss_step_elem = classification_loss_elements(
            logits=step_logits,
            targets=y_step,
            pos_weight=pos_weights["step"],
            loss_type=loss_cfg["step"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
        loss_step = (loss_step_elem * has_info).sum() / (has_info.sum() * y_step.size(1) + 1e-6)

        has_info_3d = has_info.view(-1, 1, 1)
        step_weight = (y_step * loss_cfg["solvent_pos_step_boost"] + (1.0 - y_step) * loss_cfg["solvent_neg_step_weight"]).unsqueeze(-1)
        loss_sol_elem = classification_loss_elements(
            logits=sol_logits,
            targets=y_sol,
            pos_weight=pos_weights["solvent"],
            loss_type=loss_cfg["solvent"],
            focal_gamma=loss_cfg["focal_gamma"],
        )
        sol_weight = has_info_3d * step_weight
        loss_sol = (loss_sol_elem * sol_weight).sum() / (sol_weight.sum() * y_sol.size(2) + 1e-6)

        # ratio regression loss only on (step, solvent) pairs with known numeric ratio
        sample_ids, step_ids, solvent_ids, y_ratio = [], [], [], []
        for i in range(len(ratio_pairs)):
            if has_info[i].item() != 1.0:
                continue
            for sj, sid, r in ratio_pairs[i]:
                rv = float(r)
                if not np.isfinite(rv):
                    continue
                sample_ids.append(i)
                step_ids.append(sj)
                solvent_ids.append(sid)
                y_ratio.append(rv)

        if len(sample_ids) > 0:
            sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
            step_ids_t = torch.tensor(step_ids, device=device, dtype=torch.long)
            solvent_ids_t = torch.tensor(solvent_ids, device=device, dtype=torch.long)
            y_ratio_t = torch.tensor(np.asarray(y_ratio, dtype=np.float32), device=device, dtype=torch.float32)
            h_pairs = h[sample_ids_t]
            ratio_pred = model.ratio_pred(h_pairs, step_ids_t, solvent_ids_t)
            loss_ratio = compute_ratio_loss(
                y_pred=ratio_pred,
                y_true=y_ratio_t,
                loss_type=loss_cfg["ratio_loss"],
                huber_delta=loss_cfg["ratio_huber_delta"],
                mix_mae_weight=loss_cfg["ratio_mix_mae_weight"],
            )
        else:
            loss_ratio = torch.tensor(0.0, device=device)

        loss_smooth = batch_graph_smoothness(x=x, h=h, topk=smooth_topk)

        if hasattr(model, "get_solvent_gate_alpha"):
            gate_alpha = model.get_solvent_gate_alpha()
            loss_gate = ((gate_alpha - 1.0) ** 2).mean()
        else:
            loss_gate = torch.tensor(0.0, device=device)

        loss = (
            lambdas["cartridge"] * loss_car +
            lambdas["step"] * loss_step +
            lambdas["solvent"] * loss_sol +
            lambdas["ratio"] * loss_ratio +
            lambdas["smooth"] * loss_smooth +
            lambdas["gate"] * loss_gate
        )

        if not all(
            _is_finite_scalar_tensor(v)
            for v in (loss_car, loss_step, loss_sol, loss_ratio, loss_smooth, loss_gate, loss)
        ):
            non_finite_skipped += 1
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_is_finite = True
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                grad_is_finite = False
                break
        if not grad_is_finite:
            non_finite_grad_skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total["loss"] += float(loss.item())
        total["cartridge"] += float(loss_car.item())
        total["step"] += float(loss_step.item())
        total["solvent"] += float(loss_sol.item())
        total["ratio"] += float(loss_ratio.item())
        total["smooth"] += float(loss_smooth.item())
        total["gate"] += float(loss_gate.item())
        n_batches += 1

    if n_batches == 0:
        for k in total:
            total[k] = float("nan")
    else:
        for k in total:
            total[k] /= n_batches
    if non_finite_skipped > 0:
        print(f"[train_one_epoch] skipped non-finite batches: {non_finite_skipped}")
    if non_finite_grad_skipped > 0:
        print(f"[train_one_epoch] skipped non-finite gradients: {non_finite_grad_skipped}")
    return total

@torch.no_grad()
def run_eval(
    model: SPEBaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    thresholds: Dict[str, float],
    per_label_thresholds: Optional[Dict[str, np.ndarray]],
    solvent_vocab: List[str],
    cartridge_cols: List[str],
    solvent_step_beta: float = 0.0,
    solvent_topk_per_step: int = 0,
    solvent_gate_by_step: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute test metrics + per-sample lightweight prediction objects."""
    model.eval()

    all_car_true, all_car_prob = [], []
    all_step_true, all_step_prob = [], []
    all_sol_true, all_sol_prob = [], []
    all_has_info = []
    all_cids: List[str] = []
    all_method_lists: List[List[str]] = []
    ratio_true_vals: List[float] = []
    ratio_pred_vals: List[float] = []
    ratio_pair_records: List[Tuple[int, int, int, float, float]] = []

    for batch in loader:
        sample_offset = len(all_cids)
        x = batch["x"].to(device)
        y_car = batch["y_cartridge"].cpu().numpy()
        y_step = batch["y_step"].cpu().numpy()
        y_sol = batch["y_solvent"].cpu().numpy()
        has_info = batch["has_spe_info"].cpu().numpy().astype(np.int32)

        out = model(x)
        car_prob = torch.sigmoid(out["cartridge_logits"]).cpu().numpy()
        step_prob = torch.sigmoid(out["step_logits"]).cpu().numpy()
        sol_prob = torch.sigmoid(out["solvent_logits"]).cpu().numpy()

        all_car_true.append(y_car); all_car_prob.append(car_prob)
        all_step_true.append(y_step); all_step_prob.append(step_prob)
        all_sol_true.append(y_sol); all_sol_prob.append(sol_prob)
        all_has_info.append(has_info)
        all_cids.extend(batch["cids"])
        all_method_lists.extend(batch["method_ids"])

        # ratio regression: evaluate only on (step, solvent) pairs with known numeric ratio
        h = out["h"].detach()
        ratio_pairs = batch["ratio_pairs"]
        sample_ids, step_ids, solvent_ids, y_ratio = [], [], [], []
        for i in range(len(ratio_pairs)):
            if has_info[i] != 1:
                continue
            for sj, sid, r in ratio_pairs[i]:
                sample_ids.append(i)
                step_ids.append(sj)
                solvent_ids.append(sid)
                y_ratio.append(float(r))
        if len(sample_ids) > 0:
            sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
            step_ids_t = torch.tensor(step_ids, device=device, dtype=torch.long)
            solvent_ids_t = torch.tensor(solvent_ids, device=device, dtype=torch.long)
            h_pairs = h[sample_ids_t]
            ratio_pred = model.ratio_pred(h_pairs, step_ids_t, solvent_ids_t).cpu().numpy().tolist()
            for k, rp in enumerate(ratio_pred):
                sid_global = int(sample_offset + int(sample_ids[k]))
                st = int(step_ids[k])
                so = int(solvent_ids[k])
                yt = float(y_ratio[k])
                yp = float(rp)
                ratio_pair_records.append((sid_global, st, so, yt, yp))
                ratio_true_vals.append(yt)
                ratio_pred_vals.append(yp)

    car_true = np.concatenate(all_car_true, axis=0)
    car_prob = np.concatenate(all_car_prob, axis=0)
    has_info_all = np.concatenate(all_has_info, axis=0)
    step_true = np.concatenate(all_step_true, axis=0)
    step_prob = np.concatenate(all_step_prob, axis=0)
    sol_true = np.concatenate(all_sol_true, axis=0)
    sol_prob = np.concatenate(all_sol_prob, axis=0)

    if per_label_thresholds is not None:
        car_thr_arr = np.asarray(per_label_thresholds["cartridge"], dtype=np.float32)  # (C,)
        step_thr_arr = np.asarray(per_label_thresholds["step"], dtype=np.float32)       # (5,)
        sol_thr_arr = np.asarray(per_label_thresholds["solvent"], dtype=np.float32)     # (5,S)
    else:
        car_thr_arr = np.full((car_prob.shape[1],), float(thresholds["cartridge"]), dtype=np.float32)
        step_thr_arr = np.full((step_prob.shape[1],), float(thresholds["step"]), dtype=np.float32)
        sol_thr_arr = np.full((sol_prob.shape[1], sol_prob.shape[2]), float(thresholds["solvent"]), dtype=np.float32)

    car_pred = (car_prob >= car_thr_arr[None, :]).astype(np.int32)
    step_pred = (step_prob >= step_thr_arr[None, :]).astype(np.int32)
    sol_pred, sol_prob_adj = build_solvent_pred(
        sol_prob=sol_prob,
        step_prob=step_prob,
        step_thr_arr=step_thr_arr,
        sol_thr_arr=sol_thr_arr,
        beta=solvent_step_beta,
        topk_per_step=solvent_topk_per_step,
        gate_by_step=solvent_gate_by_step,
    )

    metrics: Dict[str, Any] = {}
    metrics["cartridge"] = multilabel_stats(car_true, car_pred)

    mask = has_info_all == 1
    if mask.sum() > 0:
        metrics["step"] = multilabel_stats(step_true[mask], step_pred[mask])
        n, a, b = sol_true.shape
        sol_true_flat = sol_true.reshape(n, a * b)
        sol_pred_flat = sol_pred.reshape(n, a * b)
        metrics["solvent"] = multilabel_stats(sol_true_flat[mask], sol_pred_flat[mask])
    else:
        metrics["step"] = {}; metrics["solvent"] = {}

    metrics["ratio"] = regression_stats(np.asarray(ratio_true_vals, dtype=np.float32), np.asarray(ratio_pred_vals, dtype=np.float32))
    metrics["overall"] = compute_overall_metrics(
        car_true=car_true,
        car_pred=car_pred,
        car_prob=car_prob,
        step_true=step_true,
        step_pred=step_pred,
        sol_true=sol_true,
        sol_pred=sol_pred,
        step_prob=step_prob,
        sol_prob_adj=sol_prob_adj,
        has_info_all=has_info_all,
        ratio_pair_records=ratio_pair_records,
        ratio_tau=0.10,
    )

    # Per-sample objects (initially without ratio; will be attached later)
    per_sample: List[Dict[str, Any]] = []
    for i in range(car_true.shape[0]):
        car_probs_i = car_prob[i]
        car_order = np.argsort(-car_probs_i)
        car_candidates = [
            {"name": cartridge_cols[idx], "prob": float(car_probs_i[idx])}
            for idx in car_order[:min(20, len(car_probs_i))]
            if car_probs_i[idx] >= float(car_thr_arr[idx])
        ]
        if len(car_candidates) == 0:
            top = int(car_order[0])
            car_candidates = [{"name": cartridge_cols[top], "prob": float(car_probs_i[top])}]
        car_top = int(car_order[0])

        step_probs_i = step_prob[i]
        steps_present = [STEP_NAMES[j] for j in range(len(STEP_NAMES)) if step_probs_i[j] >= float(step_thr_arr[j])]
        if len(steps_present) == 0:
            steps_present = [STEP_NAMES[int(step_probs_i.argmax())]]

        sol_probs_i = sol_prob_adj[i]  # (5,S)
        solvents_by_step: Dict[str, List[Dict[str, float]]] = {}
        for j, step_name in enumerate(STEP_NAMES):
            if step_name not in steps_present:
                continue
            s_probs = sol_probs_i[j]
            s_order = np.argsort(-s_probs)
            sols = []
            max_keep = solvent_topk_per_step if int(solvent_topk_per_step) > 0 else 3
            for sid in s_order[:min(30, len(s_probs))]:
                if s_probs[sid] >= float(sol_thr_arr[j, sid]):
                    sols.append({"name": solvent_vocab[sid], "prob": float(s_probs[sid])})
                if len(sols) >= max_keep:
                    break
            if len(sols) == 0:
                sid0 = int(s_order[0])
                sols = [{"name": solvent_vocab[sid0], "prob": float(s_probs[sid0])}]
            solvents_by_step[step_name] = sols

        per_sample.append({
            "cid": all_cids[i],
            "has_spe_info": bool(has_info_all[i] == 1),
            "predicted": {
                "cartridge_candidates": car_candidates,
                "cartridge_top1": {"name": cartridge_cols[car_top], "prob": float(car_probs_i[car_top])},
                "steps_present": steps_present,
                "solvents_by_step": solvents_by_step,
            },
            "ground_truth": {
                "cartridge_multihot": [cartridge_cols[k] for k in np.where(car_true[i] == 1)[0].tolist()],
                "method_ids": all_method_lists[i],
            }
        })

    return metrics, per_sample


@torch.no_grad()
def attach_predicted_ratios(
    model: SPEBaselineModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    per_sample: List[Dict[str, Any]],
    solvent2id: Dict[str, int],
) -> None:
    """Add predicted numeric ratio to each (step, solvent) in the predicted scheme."""
    model.eval()
    idx = 0
    for batch in loader:
        x = batch["x"].to(device)
        out = model(x)
        h = out["h"]  # (B,H)
        bsz = x.size(0)

        for bi in range(bsz):
            item = per_sample[idx]
            pred = item["predicted"]
            enriched = {}
            for step_name, sols in pred.get("solvents_by_step", {}).items():
                step_id = STEP_NAMES.index(step_name)
                enriched_sols = []
                for s in sols:
                    sol_name = s["name"]
                    sid = solvent2id.get(sol_name)
                    if sid is None:
                        enriched_sols.append({**s, "ratio_pred": None})
                        continue
                    h_pair = h[bi:bi+1]
                    step_ids = torch.tensor([step_id], device=device, dtype=torch.long)
                    solvent_ids = torch.tensor([sid], device=device, dtype=torch.long)
                    r = model.ratio_pred(h_pair, step_ids, solvent_ids).cpu().numpy()[0]
                    enriched_sols.append({**s, "ratio_pred": float(r)})
                enriched[step_name] = enriched_sols
            pred["solvents_by_step"] = enriched
            idx += 1

def generate_detailed_report(
    per_sample: List[Dict[str, Any]],
    output_dir: str,
    data_dir: str,
    unk_solvent_token: str,
) -> None:
    """Write JSONL + Markdown report for every test pollutant."""
    ensure_dir(output_dir)

    spe_path = os.path.join(data_dir, "spe_solvent_ratio.csv")
    needed_methods = set()
    for item in per_sample:
        needed_methods.update(item["ground_truth"]["method_ids"])

    method_step_map = build_method_step_map_subset_from_csv(
        spe_csv_path=spe_path,
        method_id_set=needed_methods,
        unk_solvent_token=unk_solvent_token,
    )

    rows_jsonl = []
    md = []
    md.append("# Test Set Predictions vs Ground Truth\n")
    md.append("本报告对测试集中**每个污染物**输出：模型预测的一套方案，以及该污染物对应的所有真实方法（能在 `spe_solvent_ratio.csv` 找到的部分）。\n")
    md.append("说明：\n")
    md.append("- `missing_methods` 表示在 `processed_molecular_data.csv` 中出现、但在 `spe_solvent_ratio.csv` 中缺失的方法。\n")
    md.append("- `ratio` 为连续数值（0-1），`null` 表示未知/缺失。\n")
    md.append("\n---\n")

    for item in per_sample:
        cid = item["cid"]
        pred = item["predicted"]
        gt = item["ground_truth"]
        method_ids = gt["method_ids"]

        found, missing = [], []
        for m in method_ids:
            (found if m in method_step_map else missing).append(m)

        true_methods_detail = []
        for m in found:
            step_detail = {}
            for step in STEP_NAMES:
                pairs = method_step_map[m].get(step, [])
                if len(pairs) == 0:
                    continue
                step_detail[step] = [
                    {
                        "solvent": sol,
                        "ratio_raw": ratio_raw,
                        "ratio_value": ratio_val,
                    }
                    for (sol, ratio_raw, ratio_val) in pairs
                ]
            true_methods_detail.append({"CasMp": m, "steps": step_detail})

        row = {
            "CID": cid,
            "predicted_scheme": pred,
            "ground_truth": {
                "cartridge_multihot": gt["cartridge_multihot"],
                "methods_found_in_spe": true_methods_detail,
                "missing_methods": missing,
            }
        }
        rows_jsonl.append(row)

        md.append(f"## CID {cid}\n")
        md.append("**Predicted scheme**\n")
        md.append("```json\n" + json_dumps(pred) + "\n```\n")
        md.append("**Ground truth (Cartridge multihot, from processed file)**\n")
        md.append("```json\n" + json_dumps(gt["cartridge_multihot"]) + "\n```\n")
        md.append("**Ground truth methods (from spe_solvent_ratio.csv)**\n")
        md.append("```json\n" + json_dumps(true_methods_detail) + "\n```\n")
        if len(missing) > 0:
            md.append("**Missing methods in spe_solvent_ratio.csv**\n")
            md.append("```json\n" + json_dumps(missing) + "\n```\n")
        md.append("\n---\n")

    save_jsonl(os.path.join(output_dir, "test_predictions.jsonl"), rows_jsonl)
    with open(os.path.join(output_dir, "test_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def main():
    p = argparse.ArgumentParser(description="SPE Baseline (方案一): MLP multi-task + within-batch graph smoothness")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="output")
    p.add_argument("--cache_path", type=str, default="data/cache/preprocessed.pkl")
    p.add_argument("--require_spe_info", action="store_true", help="只保留至少有一个方法能在 spe_solvent_ratio.csv 中找到的污染物")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--enc_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)

    # graph smoothness
    p.add_argument("--smooth_topk", type=int, default=8)
    p.add_argument("--lambda_smooth", type=float, default=0.05)
    p.add_argument("--lambda_gate", type=float, default=0.0, help="约束solvent gate接近1.0的正则权重")

    # task weights
    p.add_argument("--lambda_cartridge", type=float, default=1.0)
    p.add_argument("--lambda_step", type=float, default=1.0)
    p.add_argument("--lambda_solvent", type=float, default=1.0)
    p.add_argument("--lambda_ratio", type=float, default=1.0)
    p.add_argument("--unk_solvent_token", type=str, default="__UNK__")

    # losses
    p.add_argument("--loss_cartridge", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--loss_step", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--loss_solvent", type=str, default="bce", choices=["bce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--ratio_loss", type=str, default="mse", choices=["mse", "huber", "mse_mae"])
    p.add_argument("--ratio_huber_delta", type=float, default=0.05)
    p.add_argument("--ratio_mix_mae_weight", type=float, default=0.3)
    p.add_argument("--solvent_pos_step_boost", type=float, default=2.0, help="solvent loss中，真实存在step(y_step=1)的权重增强")
    p.add_argument("--solvent_neg_step_weight", type=float, default=0.5, help="solvent loss中，真实不存在step(y_step=0)的权重")

    # split
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # thresholds
    p.add_argument("--thr_cartridge", type=float, default=0.5)
    p.add_argument("--thr_step", type=float, default=0.5)
    p.add_argument("--thr_solvent", type=float, default=0.5)
    p.add_argument("--solvent_step_beta", type=float, default=0.0, help="solvent概率按step置信度校准: p_sol * p_step^beta")
    p.add_argument("--auto_tune_solvent_step_beta", action="store_true", help="训练后在验证集自动搜索solvent_step_beta")
    p.add_argument("--solvent_topk_per_step", type=int, default=0, help="solvent后处理: 每个step最多保留top-k个溶剂; 0表示不限制")
    p.add_argument("--solvent_topk_choices", type=str, default="0,2,3", help="自动调参时top-k候选，逗号分隔")
    p.add_argument("--beta_grid_min", type=float, default=0.0)
    p.add_argument("--beta_grid_max", type=float, default=2.0)
    p.add_argument("--beta_grid_num", type=int, default=9)
    # ratio is a regression target, no probability threshold is used.

    # threshold tuning on validation
    p.add_argument("--auto_tune_thresholds", action="store_true", help="训练后在验证集自动搜索分类阈值")
    p.add_argument("--thr_grid_min", type=float, default=0.1)
    p.add_argument("--thr_grid_max", type=float, default=0.9)
    p.add_argument("--thr_grid_num", type=int, default=17)
    p.add_argument("--auto_tune_thresholds_per_label", action="store_true", help="训练后在验证集按标签搜索阈值")
    p.add_argument("--thr_min_pos_per_label", type=int, default=3, help="按标签调阈值时每标签最少正样本数")

    # checkpoint selection
    p.add_argument("--select_by", type=str, default="score", choices=["score", "loss"], help="best checkpoint 选择依据")
    p.add_argument("--w_cartridge", type=float, default=1.0, help="score 模式下 cartridge micro_f1 权重")
    p.add_argument("--w_step", type=float, default=1.0, help="score 模式下 step micro_f1 权重")
    p.add_argument("--w_solvent", type=float, default=1.0, help="score 模式下 solvent micro_f1 权重")
    p.add_argument("--w_ratio", type=float, default=0.25, help="score 模式下 ratio within_0.10 权重")

    # clip pos_weight
    p.add_argument("--car_posw_min", type=float, default=1.0)
    p.add_argument("--car_posw_max", type=float, default=50.0)
    p.add_argument("--step_posw_min", type=float, default=1.0)
    p.add_argument("--step_posw_max", type=float, default=50.0)
    p.add_argument("--sol_posw_min", type=float, default=1.0)
    p.add_argument("--sol_posw_max", type=float, default=50.0)

    args = p.parse_args()
    if (
        (args.loss_cartridge == "focal" or args.loss_step == "focal" or args.loss_solvent == "focal")
        and args.focal_gamma < 1.0
    ):
        raise ValueError(
            f"focal_gamma={args.focal_gamma} is unstable (<1.0). "
            "Please set --focal_gamma >= 1.0 (recommended: 1.0~2.0)."
        )
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.dirname(args.cache_path))

    ds, artifacts = build_datasets(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        require_spe_info=args.require_spe_info,
        unk_solvent_token=args.unk_solvent_token,
    )

    train_idx, val_idx, test_idx = split_indices(len(ds), seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # # 测试用
    # train_mask = (ds.has_spe_info[train_idx] == 1)
    # val_mask   = (ds.has_spe_info[val_idx] == 1)

    # y_train = ds.y_solvent[train_idx][train_mask]   # (Ntr, 5, S)
    # y_val   = ds.y_solvent[val_idx][val_mask]       # (Nva, 5, S)

    # pos_tr = y_train.sum(axis=0)  # (5,S)
    # pos_va = y_val.sum(axis=0)    # (5,S)

    # bad = (pos_tr == 0) & (pos_va > 0)

    # print("Total step-solvent pairs:", pos_tr.size)
    # print("Train pos==0 pairs:", int((pos_tr==0).sum()))
    # print("Unseen in train but appear in val:", int(bad.sum()))

    # step_ids, sol_ids = np.where(bad)
    # for j, s in list(zip(step_ids, sol_ids))[:30]:
    #     print(STEP_NAMES[j], artifacts.solvent_vocab[s], "val_count=", int(pos_va[j,s]))


    # standardize
    x_train = ds.xs[train_idx]
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    standardize_inplace(ds.xs, mean, std)

    # pos_weight
    car_pos_weight = compute_pos_weight(ds.y_cartridge[train_idx])

    has_info_train = ds.has_spe_info[train_idx] == 1.0
    y_step_train = ds.y_step[train_idx][has_info_train]
    y_sol_train = ds.y_solvent[train_idx][has_info_train]
    step_pos_weight = compute_pos_weight(y_step_train) if y_step_train.shape[0] > 0 else np.ones((len(STEP_NAMES),), dtype=np.float32)
    sol_pos_weight = compute_pos_weight_3d(y_sol_train) if y_sol_train.shape[0] > 0 else np.ones((len(STEP_NAMES), len(artifacts.solvent_vocab)), dtype=np.float32)

    # clip pos_weight to avoid unstable extremely large class weights
    car_pos_weight = np.clip(car_pos_weight, args.car_posw_min, args.car_posw_max)
    step_pos_weight = np.clip(step_pos_weight, args.step_posw_min, args.step_posw_max)
    sol_pos_weight = np.clip(sol_pos_weight, args.sol_posw_min, args.sol_posw_max)

    train_loader, val_loader, test_loader = build_loaders(ds, train_idx, val_idx, test_idx, batch_size=args.batch_size, num_workers=0)

    device = torch.device(args.device)
    model = SPEBaselineModel(
        input_dim=len(artifacts.descriptor_cols),
        n_cartridge=len(artifacts.cartridge_cols),
        n_steps=len(STEP_NAMES),
        n_solvent=len(artifacts.solvent_vocab),
        hidden_dim=args.hidden_dim,
        enc_layers=args.enc_layers,
        dropout=args.dropout,
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_weights_t = {
        "cartridge": torch.tensor(car_pos_weight, device=device, dtype=torch.float32),
        "step": torch.tensor(step_pos_weight, device=device, dtype=torch.float32),
        "solvent": torch.tensor(sol_pos_weight, device=device, dtype=torch.float32),
    }
    lambdas = {
        "cartridge": args.lambda_cartridge,
        "step": args.lambda_step,
        "solvent": args.lambda_solvent,
        "ratio": args.lambda_ratio,
        "smooth": args.lambda_smooth,
        "gate": args.lambda_gate,
    }
    loss_cfg = {
        "cartridge": args.loss_cartridge,
        "step": args.loss_step,
        "solvent": args.loss_solvent,
        "focal_gamma": args.focal_gamma,
        "ratio_loss": args.ratio_loss,
        "ratio_huber_delta": args.ratio_huber_delta,
        "ratio_mix_mae_weight": args.ratio_mix_mae_weight,
        "solvent_pos_step_boost": args.solvent_pos_step_boost,
        "solvent_neg_step_weight": args.solvent_neg_step_weight,
    }
    thresholds = {
        "cartridge": args.thr_cartridge,
        "step": args.thr_step,
        "solvent": args.thr_solvent,
    }
    solvent_step_beta = float(args.solvent_step_beta)
    solvent_topk_per_step = int(max(0, args.solvent_topk_per_step))

    best_val = float("inf")
    best_score = float("-inf")
    best_path = os.path.join(args.output_dir, "best_model.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, device, opt, pos_weights_t, lambdas, loss_cfg, args.smooth_topk, args.grad_clip)
        va = eval_loss(model, val_loader, device, pos_weights_t, lambdas, loss_cfg, args.smooth_topk)
        val_metrics_for_select, _ = run_eval(
            model=model,
            loader=val_loader,
            device=device,
            thresholds=thresholds,
            per_label_thresholds=None,
            solvent_vocab=artifacts.solvent_vocab,
            cartridge_cols=artifacts.cartridge_cols,
            solvent_step_beta=solvent_step_beta,
            solvent_topk_per_step=solvent_topk_per_step,
        )
        val_score = build_selection_score(
            metrics=val_metrics_for_select,
            w_cartridge=args.w_cartridge,
            w_step=args.w_step,
            w_solvent=args.w_solvent,
            w_ratio=args.w_ratio,
        )

        history.append({"epoch": epoch, "train": tr, "val": va, "val_metrics": val_metrics_for_select, "val_score": val_score})
        print(f"[Epoch {epoch:03d}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_score={val_score:.4f}")

        improved = (va["loss"] < best_val) if args.select_by == "loss" else (val_score > best_score)
        if improved:
            best_val = min(best_val, va["loss"])
            best_score = max(best_score, val_score)
            torch.save(
                {"model_state": model.state_dict(), "mean": mean, "std": std, "artifacts": artifacts.__dict__, "args": vars(args)},
                best_path,
            )

    save_json(os.path.join(args.output_dir, "train_history.json"), history)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    if args.auto_tune_thresholds:
        tuned_thresholds = tune_thresholds_on_val(
            model=model,
            val_loader=val_loader,
            device=device,
            init_thresholds=thresholds,
            solvent_step_beta=solvent_step_beta,
            grid_min=args.thr_grid_min,
            grid_max=args.thr_grid_max,
            grid_num=args.thr_grid_num,
        )
        thresholds = tuned_thresholds
        print(f"[Threshold tuning] cartridge={thresholds['cartridge']:.3f}, step={thresholds['step']:.3f}, solvent={thresholds['solvent']:.3f}")
        save_json(os.path.join(args.output_dir, "tuned_thresholds.json"), thresholds)

    if args.auto_tune_solvent_step_beta:
        tuned_beta, tuned_sol_thr, tuned_topk, tuned_beta_f1 = tune_solvent_step_beta_on_val(
            model=model,
            val_loader=val_loader,
            device=device,
            init_thresholds=thresholds,
            beta_grid_min=args.beta_grid_min,
            beta_grid_max=args.beta_grid_max,
            beta_grid_num=args.beta_grid_num,
            thr_grid_min=args.thr_grid_min,
            thr_grid_max=args.thr_grid_max,
            thr_grid_num=args.thr_grid_num,
            topk_choices=parse_int_list_csv(args.solvent_topk_choices),
        )
        solvent_step_beta = float(tuned_beta)
        thresholds["solvent"] = float(tuned_sol_thr)
        solvent_topk_per_step = int(max(0, tuned_topk))
        print(
            f"[Solvent postproc tuning] beta={solvent_step_beta:.3f}, "
            f"solvent_thr={thresholds['solvent']:.3f}, topk={solvent_topk_per_step}, "
            f"val_solvent_f1={tuned_beta_f1:.4f}"
        )
        save_json(
            os.path.join(args.output_dir, "tuned_solvent_beta.json"),
            {
                "solvent_step_beta": solvent_step_beta,
                "solvent_threshold": thresholds["solvent"],
                "solvent_topk_per_step": solvent_topk_per_step,
                "val_solvent_micro_f1": tuned_beta_f1,
            },
        )

    per_label_thresholds: Optional[Dict[str, np.ndarray]] = None
    if args.auto_tune_thresholds_per_label:
        per_label_thresholds = tune_thresholds_per_label_on_val(
            model=model,
            val_loader=val_loader,
            device=device,
            init_thresholds=thresholds,
            solvent_step_beta=solvent_step_beta,
            grid_min=args.thr_grid_min,
            grid_max=args.thr_grid_max,
            grid_num=args.thr_grid_num,
            min_pos_per_label=args.thr_min_pos_per_label,
        )
        thr_obj = {
            "cartridge": per_label_thresholds["cartridge"].tolist(),
            "step": per_label_thresholds["step"].tolist(),
            "solvent": per_label_thresholds["solvent"].tolist(),
        }
        save_json(os.path.join(args.output_dir, "tuned_thresholds_per_label.json"), thr_obj)
        print("[Per-label threshold tuning] saved tuned_thresholds_per_label.json")

    metrics, per_sample = run_eval(
        model=model,
        loader=test_loader,
        device=device,
        thresholds=thresholds,
        per_label_thresholds=per_label_thresholds,
        solvent_vocab=artifacts.solvent_vocab,
        cartridge_cols=artifacts.cartridge_cols,
        solvent_step_beta=solvent_step_beta,
        solvent_topk_per_step=solvent_topk_per_step,
    )

    save_json(os.path.join(args.output_dir, "test_metrics.json"), {
        "metrics": metrics,
        "thresholds": thresholds,
        "solvent_step_beta": solvent_step_beta,
        "solvent_step_beta_tuned": bool(args.auto_tune_solvent_step_beta),
        "solvent_topk_per_step": solvent_topk_per_step,
        "per_label_thresholds_enabled": bool(args.auto_tune_thresholds_per_label),
        "loss_config": loss_cfg,
        "select_by": args.select_by,
        "best_val_loss": best_val,
        "best_val_score": best_score,
        "splits": {"n_total": len(ds), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))},
        "note": "Step/solvent/ratio 指标仅对 has_spe_info=1 的样本统计；ratio 只在存在数值标注的 (step, solvent) pair 上计算。",
    })

    # fill ratio predictions in the predicted scheme
    attach_predicted_ratios(model, test_loader, device, per_sample, artifacts.solvent2id)

    # write per-sample comparisons (all test pollutants)
    generate_detailed_report(per_sample, args.output_dir, args.data_dir, args.unk_solvent_token)

    print("Done.")
    print(f"- Best checkpoint: {best_path}")
    print(f"- Metrics: {os.path.join(args.output_dir, 'test_metrics.json')}")
    print(f"- Predictions: {os.path.join(args.output_dir, 'test_predictions.jsonl')}")
    print(f"- Report: {os.path.join(args.output_dir, 'test_report.md')}")


if __name__ == "__main__":
    main()
