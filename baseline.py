"""Traditional ML baselines (non end-to-end multi-task).

Usage examples:
  python baseline.py DecisionTree Cartridge
  python baseline.py RandomForest Step
  python baseline.py SVM Solvent
  python baseline.py XGBoost Cartridge
  python baseline.py LightGBM Step
  python baseline.py MLP Solvent
  python baseline.py RandomForest Ratio

Notes:
  - Traditional models are trained **per subtask**.
  - For multi-label classification (Cartridge/Step/Solvent), we use One-vs-Rest.
  - For Ratio (continuous regression in [0,1]), we train a regressor on (pollutant, step, solvent) pairs
    with known numeric ratio labels.

Outputs are written to: output/baseline_output/
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils.dataloader import build_datasets, split_indices
from utils.io import ensure_dir, save_json, save_jsonl
from utils.metrics import multilabel_stats, sigmoid_to_pred, regression_stats
from utils.parsing import STEP_NAMES
from utils.seed import set_seed


SUPPORTED_MODELS = ["DecisionTree", "RandomForest", "SVM", "MLP", "XGBoost", "LightGBM"]


def _import_xgboost():
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("XGBoost 未安装。请先执行: pip install xgboost") from e
    return xgb


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("LightGBM 未安装。请先执行: pip install lightgbm") from e
    return lgb


def _standardize_desc(xs: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize descriptor features using train split statistics."""
    mean = xs[train_idx].mean(axis=0)
    std = xs[train_idx].std(axis=0) + 1e-8
    xs_std = (xs - mean[None, :]) / std[None, :]
    return xs_std, mean, std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _get_clf_and_score_fn(model_name: str, seed: int):
    """Return (estimator, score_fn) for multi-label classification.

    score_fn(X) should return score/probabilities in [0,1] with shape (N, L).
    """
    name = model_name.lower()

    if name == "decisiontree":
        base = DecisionTreeClassifier(random_state=seed)
        clf = OneVsRestClassifier(base)
        def score_fn(m, X):
            # OneVsRest + predict_proba -> (N,L)
            return m.predict_proba(X)
        return clf, score_fn

    if name == "randomforest":
        base = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        )
        clf = OneVsRestClassifier(base)
        def score_fn(m, X):
            return m.predict_proba(X)
        return clf, score_fn

    if name == "svm":
        # Linear SVM is used for scalability in multi-label settings.
        base = LinearSVC(random_state=seed)
        clf = OneVsRestClassifier(base)

        def score_fn(m, X):
            # decision_function -> (-inf, +inf). Map to (0,1) via sigmoid.
            scores = m.decision_function(X)
            return _sigmoid(scores)

        return clf, score_fn

    if name == "mlp":
        base = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            # IMPORTANT:
            # In One-vs-Rest multi-label training, some labels are extremely rare.
            # sklearn MLPClassifier with early_stopping=True performs an internal
            # stratified split and can crash when the positive class has only 1 sample.
            # Disable early_stopping to keep training robust on sparse labels.
            early_stopping=False,
            random_state=seed,
        )
        clf = OneVsRestClassifier(base)
        def score_fn(m, X):
            return m.predict_proba(X)
        return clf, score_fn

    if name == "xgboost":
        xgb = _import_xgboost()
        base = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
        clf = OneVsRestClassifier(base)
        def score_fn(m, X):
            return m.predict_proba(X)
        return clf, score_fn

    if name == "lightgbm":
        lgb = _import_lightgbm()
        base = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        clf = OneVsRestClassifier(base)
        def score_fn(m, X):
            return m.predict_proba(X)
        return clf, score_fn

    raise ValueError(f"Unknown model: {model_name}. Choose from: {', '.join(SUPPORTED_MODELS)}")


def _get_regressor(model_name: str, seed: int):
    name = model_name.lower()
    if name == "decisiontree":
        return DecisionTreeRegressor(random_state=seed)
    if name == "randomforest":
        return RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "svm":
        # LinearSVR is much faster than RBF SVR on mid/large datasets.
        return LinearSVR(random_state=seed)
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            # Keep behavior consistent with classifier branch and avoid unstable
            # internal validation split on small subsets.
            early_stopping=False,
            random_state=seed,
        )
    if name == "xgboost":
        xgb = _import_xgboost()
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
    if name == "lightgbm":
        lgb = _import_lightgbm()
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {model_name}. Choose from: {', '.join(SUPPORTED_MODELS)}")


def _onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _save_multilabel_npz(
    path: str,
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        sample_indices=sample_indices.astype(np.int32),
        y_true=y_true.astype(np.uint8),
        y_pred=y_pred.astype(np.uint8),
        y_score=y_score.astype(np.float32),
    )


def _save_ratio_npz(
    path: str,
    sample_indices: np.ndarray,
    step_ids: np.ndarray,
    solvent_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        sample_indices=sample_indices.astype(np.int32),
        step_ids=step_ids.astype(np.int32),
        solvent_ids=solvent_ids.astype(np.int32),
        y_true=y_true.astype(np.float32),
        y_pred=y_pred.astype(np.float32),
    )


def build_ratio_pair_dataset(
    xs_desc: np.ndarray,
    ratio_pairs_list: List[List[Tuple[int, int, float]]],
    indices: np.ndarray,
    n_steps: int,
    n_solvent: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Create a (pollutant, step, solvent) pair-level regression dataset.

    X = [desc(26), step_onehot(5), solvent_onehot(S)]
    y = ratio in [0,1]

    Returns:
      X_pair: (P, 26+5+S)
      y_pair: (P,)
      meta: list of dicts per pair for saving predictions
    """
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta: List[Dict[str, Any]] = []
    for i in indices.tolist():
        pairs = ratio_pairs_list[i]
        if not pairs:
            continue
        desc = xs_desc[i]
        for step_id, sid, r in pairs:
            feat = np.concatenate([desc, _onehot(step_id, n_steps), _onehot(sid, n_solvent)], axis=0)
            X_list.append(feat.astype(np.float32))
            y_list.append(float(r))
            meta.append({"sample_index": int(i), "step_id": int(step_id), "solvent_id": int(sid), "y_true": float(r)})
    if len(X_list) == 0:
        return np.zeros((0, xs_desc.shape[1] + n_steps + n_solvent), dtype=np.float32), np.zeros((0,), dtype=np.float32), meta
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=np.float32), meta


def main():
    p = argparse.ArgumentParser(description="Traditional ML baselines for SPE subtasks")
    p.add_argument("model", type=str, help="DecisionTree | RandomForest | SVM | MLP | XGBoost | LightGBM")
    p.add_argument("task", type=str, help="Cartridge | Step | Solvent | Ratio")

    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--cache_path", type=str, default="data/cache/preprocessed.pkl")
    p.add_argument("--output_dir", type=str, default="output/baseline_output")
    p.add_argument("--require_spe_info", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for multilabel classification")

    # 与 spe/ 的步骤3 使用完全相同的官能团特征，用来做同输入的公平对比。
    # 默认关闭，所以不带这个开关时行为与历史 baseline 一字不差。
    p.add_argument("--use_functional_groups", action="store_true",
                   help="把官能团特征拼到描述符后面（输入 21 -> 96 维）")
    p.add_argument("--fg_transform", type=str, default="log1p", choices=["log1p", "raw", "binary"])
    p.add_argument("--fg_min_pos", type=int, default=20)

    args = p.parse_args()
    set_seed(args.seed)

    model_name = args.model
    task_name = args.task.lower()

    ensure_dir(args.output_dir)
    run_dir = os.path.join(args.output_dir, f"{model_name}_{args.task}")
    ensure_dir(run_dir)

    # Load dataset (same preprocessing as deep model)
    ds, artifacts = build_datasets(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        require_spe_info=args.require_spe_info,
        unk_solvent_token="__UNK__",
    )

    train_idx, val_idx, test_idx = split_indices(len(ds), seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    n_fg = 0
    if args.use_functional_groups:
        # 复用 spe.features，保证与多任务模型拿到的是同一份特征矩阵：
        # 同样的低频过滤（只在训练集上统计）、同样的 log1p 变换、同样的列顺序。
        from spe.features import build_functional_group_features

        fg = build_functional_group_features(
            cids=ds.cids,
            data_dir=args.data_dir,
            transform=args.fg_transform,
            min_pos=args.fg_min_pos,
            fit_indices=train_idx,
        )
        ds.xs = np.concatenate([ds.xs, fg.matrix], axis=1).astype(np.float32)
        n_fg = int(fg.matrix.shape[1])
        print(f"[FG] 输入维度 {ds.xs.shape[1]}（{ds.xs.shape[1] - n_fg} 维描述符 + {n_fg} 维官能团，"
              f"变换={args.fg_transform}，未关联样本 {fg.n_missing} 个）")

    xs_std, mean, std = _standardize_desc(ds.xs, train_idx)

    out_obj: Dict[str, Any] = {
        "model": model_name,
        "task": args.task,
        "splits": {"n_total": len(ds), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx))},
        "threshold": args.threshold,
        "input_dim": int(ds.xs.shape[1]),
        "n_functional_groups": n_fg,
    }

    if task_name == "cartridge":
        Xtr, Xte = xs_std[train_idx], xs_std[test_idx]
        ytr, yte = ds.y_cartridge[train_idx], ds.y_cartridge[test_idx]
        clf, score_fn = _get_clf_and_score_fn(model_name, args.seed)
        clf.fit(Xtr, ytr)
        scores = score_fn(clf, Xte)
        y_pred = sigmoid_to_pred(scores, args.threshold)
        metrics = multilabel_stats(yte, y_pred)
        out_obj["metrics"] = metrics

        # per-sample predictions
        rows = []
        for k, i in enumerate(test_idx.tolist()):
            rows.append({
                "cid": ds.cids[i],
                "y_true": [artifacts.cartridge_cols[j] for j in np.where(yte[k] == 1)[0].tolist()],
                "y_pred": [artifacts.cartridge_cols[j] for j in np.where(y_pred[k] == 1)[0].tolist()],
            })
        save_jsonl(os.path.join(run_dir, "predictions.jsonl"), rows)
        _save_multilabel_npz(
            path=os.path.join(run_dir, "predictions_full.npz"),
            sample_indices=test_idx,
            y_true=yte,
            y_pred=y_pred,
            y_score=scores,
        )
        save_json(os.path.join(run_dir, "metrics.json"), out_obj)
        print(f"[DONE] {model_name} on Cartridge -> {run_dir}")
        return

    if task_name == "step":
        # Only evaluate on has_spe_info == 1
        mask_tr = (ds.has_spe_info[train_idx] == 1)
        mask_te = (ds.has_spe_info[test_idx] == 1)
        tr_ids = train_idx[mask_tr]
        te_ids = test_idx[mask_te]

        Xtr, Xte = xs_std[tr_ids], xs_std[te_ids]
        ytr, yte = ds.y_step[tr_ids], ds.y_step[te_ids]

        clf, score_fn = _get_clf_and_score_fn(model_name, args.seed)
        clf.fit(Xtr, ytr)
        scores = score_fn(clf, Xte)
        y_pred = sigmoid_to_pred(scores, args.threshold)
        metrics = multilabel_stats(yte, y_pred)
        out_obj["metrics"] = metrics
        out_obj["n_eval_samples"] = int(len(te_ids))

        rows = []
        for k, i in enumerate(te_ids.tolist()):
            rows.append({
                "cid": ds.cids[i],
                "y_true": [STEP_NAMES[j] for j in np.where(yte[k] == 1)[0].tolist()],
                "y_pred": [STEP_NAMES[j] for j in np.where(y_pred[k] == 1)[0].tolist()],
            })
        save_jsonl(os.path.join(run_dir, "predictions.jsonl"), rows)
        _save_multilabel_npz(
            path=os.path.join(run_dir, "predictions_full.npz"),
            sample_indices=te_ids,
            y_true=yte,
            y_pred=y_pred,
            y_score=scores,
        )
        save_json(os.path.join(run_dir, "metrics.json"), out_obj)
        print(f"[DONE] {model_name} on Step -> {run_dir}")
        return

    if task_name == "solvent":
        # Only evaluate on has_spe_info == 1
        mask_tr = (ds.has_spe_info[train_idx] == 1)
        mask_te = (ds.has_spe_info[test_idx] == 1)
        tr_ids = train_idx[mask_tr]
        te_ids = test_idx[mask_te]

        Xtr, Xte = xs_std[tr_ids], xs_std[te_ids]
        ytr = ds.y_solvent[tr_ids].reshape(len(tr_ids), -1)
        yte = ds.y_solvent[te_ids].reshape(len(te_ids), -1)

        clf, score_fn = _get_clf_and_score_fn(model_name, args.seed)
        clf.fit(Xtr, ytr)
        scores = score_fn(clf, Xte)
        y_pred = sigmoid_to_pred(scores, args.threshold)
        metrics = multilabel_stats(yte, y_pred)
        out_obj["metrics"] = metrics
        out_obj["n_eval_samples"] = int(len(te_ids))
        out_obj["n_labels"] = int(yte.shape[1])

        # For solvent, writing full label list can be huge; we write only counts + top hits per sample.
        rows = []
        n_steps = len(STEP_NAMES)
        n_sol = len(artifacts.solvent_vocab)
        for k, i in enumerate(te_ids.tolist()):
            true_idx = np.where(yte[k] == 1)[0].tolist()
            pred_idx = np.where(y_pred[k] == 1)[0].tolist()

            def decode(flat_idx_list: List[int]) -> List[Dict[str, Any]]:
                out = []
                for fid in flat_idx_list:
                    step_id = fid // n_sol
                    sol_id = fid % n_sol
                    out.append({"step": STEP_NAMES[step_id], "solvent": artifacts.solvent_vocab[sol_id]})
                return out

            rows.append({
                "cid": ds.cids[i],
                "n_true": int(len(true_idx)),
                "n_pred": int(len(pred_idx)),
                "true": decode(true_idx[:50]),
                "pred": decode(pred_idx[:50]),
            })
        save_jsonl(os.path.join(run_dir, "predictions.jsonl"), rows)
        _save_multilabel_npz(
            path=os.path.join(run_dir, "predictions_full.npz"),
            sample_indices=te_ids,
            y_true=yte,
            y_pred=y_pred,
            y_score=scores,
        )
        save_json(os.path.join(run_dir, "metrics.json"), out_obj)
        print(f"[DONE] {model_name} on Solvent -> {run_dir}")
        return

    if task_name == "ratio":
        # Ratio is defined only on (step, solvent) pairs with numeric labels.
        n_steps = len(STEP_NAMES)
        n_sol = len(artifacts.solvent_vocab)

        Xtr_pair, ytr_pair, _ = build_ratio_pair_dataset(xs_std, ds.ratio_pairs, train_idx, n_steps, n_sol)
        Xte_pair, yte_pair, meta_te = build_ratio_pair_dataset(xs_std, ds.ratio_pairs, test_idx, n_steps, n_sol)

        if Xtr_pair.shape[0] == 0 or Xte_pair.shape[0] == 0:
            out_obj["metrics"] = {"n": 0, "note": "No ratio pairs available in train/test."}
            save_json(os.path.join(run_dir, "metrics.json"), out_obj)
            print(f"[WARN] No ratio pairs available -> {run_dir}")
            return

        # Standardize pair features (helps LinearSVR)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(Xtr_pair)
        Xte_s = scaler.transform(Xte_pair)

        reg = _get_regressor(model_name, args.seed)
        reg.fit(Xtr_s, ytr_pair)
        y_pred = reg.predict(Xte_s)
        # Clamp to [0,1] to match label domain
        y_pred = np.clip(y_pred, 0.0, 1.0)

        out_obj["metrics"] = regression_stats(yte_pair, y_pred)
        out_obj["metrics"]["n_pairs_eval"] = int(len(yte_pair))

        rows = []
        for k in range(len(yte_pair)):
            m = meta_te[k]
            rows.append({
                "sample_index": m["sample_index"],
                "cid": ds.cids[m["sample_index"]],
                "step": STEP_NAMES[m["step_id"]],
                "solvent": artifacts.solvent_vocab[m["solvent_id"]],
                "y_true": float(yte_pair[k]),
                "y_pred": float(y_pred[k]),
            })
        save_jsonl(os.path.join(run_dir, "predictions.jsonl"), rows)
        _save_ratio_npz(
            path=os.path.join(run_dir, "predictions_full.npz"),
            sample_indices=np.asarray([m["sample_index"] for m in meta_te], dtype=np.int32),
            step_ids=np.asarray([m["step_id"] for m in meta_te], dtype=np.int32),
            solvent_ids=np.asarray([m["solvent_id"] for m in meta_te], dtype=np.int32),
            y_true=yte_pair,
            y_pred=y_pred,
        )
        save_json(os.path.join(run_dir, "metrics.json"), out_obj)
        print(f"[DONE] {model_name} on Ratio -> {run_dir}")
        return

    raise ValueError(f"Unknown task: {args.task}. Choose from: Cartridge, Step, Solvent, Ratio")


if __name__ == "__main__":
    main()
