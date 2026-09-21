import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from utils.metrics import multilabel_stats
from utils.parsing import STEP_NAMES


def _row_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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
    step_prob: np.ndarray,
    sol_true: np.ndarray,
    sol_pred: np.ndarray,
    sol_prob: np.ndarray,
    has_info: np.ndarray,
    ratio_pair_records: List[Tuple[int, int, int, float, float]],
    ratio_tau: float = 0.10,
) -> Dict[str, float]:
    n = car_true.shape[0]

    ratio_ok_per_sample = np.ones((n,), dtype=np.float64)
    ratio_score_per_sample = np.ones((n,), dtype=np.float64)
    pair_abs_err: Dict[Tuple[int, int, int], List[float]] = {}
    sample_abs_err: Dict[int, List[float]] = {}
    for sid, st, so, y_t, y_p in ratio_pair_records:
        e = abs(float(y_p) - float(y_t))
        pair_abs_err.setdefault((int(sid), int(st), int(so)), []).append(e)
        sample_abs_err.setdefault(int(sid), []).append(e)
    for sid, errs in sample_abs_err.items():
        arr = np.asarray(errs, dtype=np.float64)
        ratio_ok_per_sample[sid] = float(np.all(arr <= ratio_tau))
        ratio_score_per_sample[sid] = float(np.mean(np.maximum(0.0, 1.0 - arr / ratio_tau)))

    car_exact = np.all(car_true == car_pred, axis=1)
    step_exact = np.all(step_true == step_pred, axis=1)
    sol_exact = np.all(sol_true == sol_pred, axis=(1, 2))
    strict = np.ones((n,), dtype=bool)
    strict &= car_exact
    strict[has_info] &= step_exact[has_info]
    strict[has_info] &= sol_exact[has_info]
    strict[has_info] &= (ratio_ok_per_sample[has_info] >= 0.5)
    strict_end_to_end_acc = float(np.mean(strict.astype(np.float64)))

    s_car = _row_jaccard(car_true, car_pred)
    s_step = _row_jaccard(step_true, step_pred)
    s_sol = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        if not has_info[i]:
            s_sol[i] = 1.0
            continue
        vals = []
        for j in range(sol_true.shape[1]):
            yt = sol_true[i, j]
            yp = sol_pred[i, j]
            inter = float(np.logical_and(yt == 1, yp == 1).sum())
            uni = float(np.logical_or(yt == 1, yp == 1).sum())
            vals.append(1.0 if uni == 0 else inter / uni)
        s_sol[i] = float(np.mean(vals)) if vals else 1.0
    scheme_similarity_score = float(
        np.mean(
            0.2 * s_car
            + 0.2 * np.where(has_info, s_step, 1.0)
            + 0.4 * np.where(has_info, s_sol, 1.0)
            + 0.2 * np.where(has_info, ratio_score_per_sample, 1.0)
        )
    )

    hc = np.ones((n,), dtype=np.float64)
    for i in range(n):
        if not has_info[i]:
            hc[i] = 1.0
            continue
        vals = []
        for j in range(step_true.shape[1]):
            has_sol_pred = bool(np.any(sol_pred[i, j] == 1))
            if step_true[i, j] == 0:
                vals.append(1.0 if not has_sol_pred else 0.0)
            else:
                vals.append(1.0 if has_sol_pred else 0.0)
        hc[i] = float(np.mean(vals)) if vals else 1.0
    hierarchical_consistency_acc = float(np.mean(hc))

    top1_car = np.argmax(car_prob, axis=1)
    top1_step = np.argmax(step_prob, axis=1)
    hit = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        ok = bool(car_true[i, top1_car[i]] == 1)
        if has_info[i]:
            st = int(top1_step[i])
            so = int(np.argmax(sol_prob[i, st]))
            ok = ok and bool(step_true[i, st] == 1) and bool(sol_true[i, st, so] == 1)
            errs = pair_abs_err.get((int(i), st, so), None)
            if errs is not None and len(errs) > 0:
                ok = ok and bool(np.mean(np.asarray(errs, dtype=np.float64)) <= ratio_tau)
        hit[i] = 1.0 if ok else 0.0
    method_hit_at_1 = float(np.mean(hit))

    car_f1 = float(multilabel_stats(car_true, car_pred)["micro_f1"])
    step_f1 = float(multilabel_stats(step_true[has_info], step_pred[has_info])["micro_f1"]) if has_info.any() else 0.0
    n2, a2, b2 = sol_true.shape
    sol_true_flat = sol_true.reshape(n2, a2 * b2)
    sol_pred_flat = sol_pred.reshape(n2, a2 * b2)
    sol_f1 = float(multilabel_stats(sol_true_flat[has_info], sol_pred_flat[has_info])["micro_f1"]) if has_info.any() else 0.0
    ratio_within = float(
        np.mean(np.asarray([e <= ratio_tau for errs in sample_abs_err.values() for e in errs], dtype=np.float64))
    ) if len(sample_abs_err) > 0 else 0.0
    global_utility_score = float(100.0 * (0.25 * car_f1 + 0.25 * step_f1 + 0.35 * sol_f1 + 0.15 * ratio_within))

    return {
        "strict_end_to_end_acc": strict_end_to_end_acc,
        "scheme_similarity_score": scheme_similarity_score,
        "hierarchical_consistency_acc": hierarchical_consistency_acc,
        "method_hit_at_1": method_hit_at_1,
        "global_utility_score": global_utility_score,
    }


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def compute_for_model(base_output_dir: str, model_name: str) -> Dict[str, float]:
    car_dir = os.path.join(base_output_dir, f"{model_name}_Cartridge")
    step_dir = os.path.join(base_output_dir, f"{model_name}_Step")
    sol_dir = os.path.join(base_output_dir, f"{model_name}_Solvent")
    ratio_dir = os.path.join(base_output_dir, f"{model_name}_Ratio")

    car = _load_npz(os.path.join(car_dir, "predictions_full.npz"))
    step = _load_npz(os.path.join(step_dir, "predictions_full.npz"))
    sol = _load_npz(os.path.join(sol_dir, "predictions_full.npz"))
    ratio = _load_npz(os.path.join(ratio_dir, "predictions_full.npz"))

    test_indices = car["sample_indices"].astype(np.int32)
    n = test_indices.shape[0]
    idx2row = {int(sid): i for i, sid in enumerate(test_indices.tolist())}

    car_true = car["y_true"].astype(np.uint8)
    car_pred = car["y_pred"].astype(np.uint8)
    car_prob = car["y_score"].astype(np.float32)

    step_true = np.zeros((n, len(STEP_NAMES)), dtype=np.uint8)
    step_pred = np.zeros((n, len(STEP_NAMES)), dtype=np.uint8)
    step_prob = np.zeros((n, len(STEP_NAMES)), dtype=np.float32)
    has_info = np.zeros((n,), dtype=bool)
    for k, sid in enumerate(step["sample_indices"].astype(np.int32).tolist()):
        if sid not in idx2row:
            continue
        i = idx2row[sid]
        step_true[i] = step["y_true"][k].astype(np.uint8)
        step_pred[i] = step["y_pred"][k].astype(np.uint8)
        step_prob[i] = step["y_score"][k].astype(np.float32)
        has_info[i] = True

    n_steps = len(STEP_NAMES)
    flat_dim = int(sol["y_true"].shape[1])
    if flat_dim % n_steps != 0:
        raise ValueError(f"Solvent flat dim {flat_dim} not divisible by n_steps={n_steps}")
    n_solvent = flat_dim // n_steps
    sol_true = np.zeros((n, n_steps, n_solvent), dtype=np.uint8)
    sol_pred = np.zeros((n, n_steps, n_solvent), dtype=np.uint8)
    sol_prob = np.zeros((n, n_steps, n_solvent), dtype=np.float32)
    for k, sid in enumerate(sol["sample_indices"].astype(np.int32).tolist()):
        if sid not in idx2row:
            continue
        i = idx2row[sid]
        sol_true[i] = sol["y_true"][k].reshape(n_steps, n_solvent).astype(np.uint8)
        sol_pred[i] = sol["y_pred"][k].reshape(n_steps, n_solvent).astype(np.uint8)
        sol_prob[i] = sol["y_score"][k].reshape(n_steps, n_solvent).astype(np.float32)

    ratio_pair_records: List[Tuple[int, int, int, float, float]] = []
    rsid = ratio["sample_indices"].astype(np.int32)
    rst = ratio["step_ids"].astype(np.int32)
    rso = ratio["solvent_ids"].astype(np.int32)
    ryt = ratio["y_true"].astype(np.float32)
    ryp = ratio["y_pred"].astype(np.float32)
    for k in range(rsid.shape[0]):
        sid = int(rsid[k])
        if sid not in idx2row:
            continue
        ratio_pair_records.append((idx2row[sid], int(rst[k]), int(rso[k]), float(ryt[k]), float(ryp[k])))

    overall = compute_overall_metrics(
        car_true=car_true,
        car_pred=car_pred,
        car_prob=car_prob,
        step_true=step_true,
        step_pred=step_pred,
        step_prob=step_prob,
        sol_true=sol_true,
        sol_pred=sol_pred,
        sol_prob=sol_prob,
        has_info=has_info,
        ratio_pair_records=ratio_pair_records,
        ratio_tau=0.10,
    )
    return overall


def main() -> None:
    p = argparse.ArgumentParser(description="Compute strict 5 overall metrics for baselines")
    p.add_argument("--output_dir", type=str, default="output/baseline_output")
    p.add_argument(
        "--models",
        type=str,
        default="DecisionTree,RandomForest,SVM,XGBoost,LightGBM,MLP",
        help="Comma-separated model names",
    )
    args = p.parse_args()

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    summary: Dict[str, Any] = {}

    for m in models:
        overall = compute_for_model(args.output_dir, m)
        summary[m] = overall

        # Write a standalone file
        out_path = os.path.join(args.output_dir, f"{m}_overall_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": m, "overall": overall}, f, ensure_ascii=False, indent=2)

        # Inject into each task metrics.json for convenience
        for task in ["Cartridge", "Step", "Solvent", "Ratio"]:
            mp = os.path.join(args.output_dir, f"{m}_{task}", "metrics.json")
            if not os.path.exists(mp):
                continue
            with open(mp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "metrics" in obj and isinstance(obj["metrics"], dict):
                obj["metrics"]["overall"] = overall
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        print(f"[DONE] overall metrics for {m}")

    summary_path = os.path.join(args.output_dir, "baseline_overall_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[DONE] summary -> {summary_path}")


if __name__ == "__main__":
    main()

