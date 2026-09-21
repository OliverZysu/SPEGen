"""
步骤 1 的离线工具：对**已有**的实验和 baseline 重算干净口径指标，不需要重新训练。

两种输入：

    # 历史 MTL 实验（会加载 best_model.pt 重跑一遍测试集推理）
    python -m spe.tools.recompute_metrics --run_dir output_mtl_v22_s2_car_strict_push

    # baseline（直接读 predictions_full.npz 里已落盘的概率）
    python -m spe.tools.recompute_metrics --baseline_dir output/baseline_output

MTL 分支会顺带做一个**自检**：重算出来的旧口径指标必须和该目录里 `test_metrics.json`
记录的数值一致（误差 < 1e-6）。这一步是为了证明干净口径与旧口径的差异来自口径本身，
而不是重算过程接错了线。

结果写到各自目录下的 `test_metrics_clean.json`，不覆盖任何既有文件。
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import main as legacy
from model import SPEBaselineModel
from utils.dataloader import build_datasets, build_loaders, split_indices
from utils.io import save_json
from utils.parsing import STEP_NAMES

from ..evaluate import collect_predictions
from ..metrics_strict import compare_legacy_vs_clean, compute_overall_metrics_v2

LEGACY_KEYS = (
    "strict_end_to_end_acc",
    "scheme_similarity_score",
    "hierarchical_consistency_acc",
    "method_hit_at_1",
    "global_utility_score",
)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_settings(run_dir: str, ckpt_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    还原该 run 评估测试集时用的解码设置。

    优先读 `test_metrics.json`（那是评估当时真正生效的值），缺失则退回 checkpoint 里的 args。
    """
    tm = _read_json(os.path.join(run_dir, "test_metrics.json")) or {}
    return {
        "thresholds": tm.get("thresholds") or {
            "cartridge": ckpt_args.get("thr_cartridge", 0.5),
            "step": ckpt_args.get("thr_step", 0.5),
            "solvent": ckpt_args.get("thr_solvent", 0.5),
        },
        "solvent_step_beta": float(
            tm.get("solvent_step_beta", ckpt_args.get("solvent_step_beta", 0.0))
        ),
        "solvent_topk_per_step": int(
            tm.get("solvent_topk_per_step", ckpt_args.get("solvent_topk_per_step", 0))
        ),
        "solvent_gate_by_step": bool(
            tm.get("solvent_gate_by_step", ckpt_args.get("solvent_gate_by_step", False))
        ),
        "stored_overall": (tm.get("metrics", {}) or {}).get("overall"),
    }


def recompute_run(run_dir: str, device_str: str = "cpu") -> Dict[str, Any]:
    """加载一个历史 MTL 实验的 checkpoint，重跑测试集并给出双口径指标。"""
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{run_dir} 里没有 best_model.pt")

    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args: Dict[str, Any] = ckpt.get("args", {}) or {}
    settings = _run_settings(run_dir, ckpt_args)

    # 走 legacy 数据路径：用 main.standardize_inplace 而不是 spe.data 的安全版本，
    # 保证重算结果与当年训练时逐位一致。
    ds, artifacts = build_datasets(
        data_dir=ckpt_args.get("data_dir", "data"),
        cache_path=ckpt_args.get("cache_path", "data/cache/dataset_cache.pkl"),
        require_spe_info=bool(ckpt_args.get("require_spe_info", False)),
        unk_solvent_token=ckpt_args.get("unk_solvent_token", "__UNK__"),
    )
    train_idx, val_idx, test_idx = split_indices(
        len(ds),
        seed=int(ckpt_args.get("seed", 42)),
        train_ratio=float(ckpt_args.get("train_ratio", 0.8)),
        val_ratio=float(ckpt_args.get("val_ratio", 0.1)),
    )
    mean = ckpt.get("mean")
    std = ckpt.get("std")
    if mean is None or std is None:
        x_train = ds.xs[train_idx]
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    legacy.standardize_inplace(ds.xs, np.asarray(mean), np.asarray(std))

    _, _, test_loader = build_loaders(
        ds, train_idx, val_idx, test_idx,
        batch_size=int(ckpt_args.get("batch_size", 64)), num_workers=0,
    )

    model = SPEBaselineModel(
        input_dim=ds.xs.shape[1],
        n_cartridge=len(artifacts.cartridge_cols),
        n_steps=len(STEP_NAMES),
        n_solvent=len(artifacts.solvent_vocab),
        hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
        enc_layers=int(ckpt_args.get("enc_layers", 2)),
        dropout=float(ckpt_args.get("dropout", 0.2)),
        step_emb_dim=int(ckpt_args.get("step_emb_dim", 32)),
        solvent_emb_dim=int(ckpt_args.get("solvent_emb_dim", 64)),
        conc_hidden_dim=int(ckpt_args.get("conc_hidden_dim", 256)),
    ).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)

    bundle = collect_predictions(
        model, test_loader, device, solvent_step_beta=settings["solvent_step_beta"]
    )

    # 还原当年的阈值：优先用 per-label 文件
    per_label = _read_json(os.path.join(run_dir, "tuned_thresholds_per_label.json"))
    thr = settings["thresholds"]
    if per_label is not None:
        car_thr = np.asarray(per_label["cartridge"], dtype=np.float32)
        step_thr = np.asarray(per_label["step"], dtype=np.float32)
        sol_thr = np.asarray(per_label["solvent"], dtype=np.float32)
    else:
        car_thr = np.full((bundle.car_prob.shape[1],), float(thr["cartridge"]), dtype=np.float32)
        step_thr = np.full((bundle.step_prob.shape[1],), float(thr["step"]), dtype=np.float32)
        sol_thr = np.full(bundle.sol_prob.shape[1:], float(thr["solvent"]), dtype=np.float32)

    car_pred = (bundle.car_prob >= car_thr[None, :]).astype(np.int32)
    step_pred = (bundle.step_prob >= step_thr[None, :]).astype(np.int32)
    # bundle.sol_prob 已经乘过 beta，等价于 main.build_solvent_pred 内部的校准结果
    sol_pred = (bundle.sol_prob >= sol_thr[None, :, :]).astype(np.int32)
    if settings["solvent_gate_by_step"]:
        sol_pred = sol_pred * step_pred[:, :, None]
    sol_pred = legacy._apply_solvent_topk_mask(
        sol_pred, bundle.sol_prob, topk_per_step=settings["solvent_topk_per_step"]
    )

    result = compute_overall_metrics_v2(
        car_true=bundle.car_true, car_pred=car_pred, car_prob=bundle.car_prob,
        step_true=bundle.step_true, step_pred=step_pred,
        sol_true=bundle.sol_true, sol_pred=sol_pred,
        step_prob=bundle.step_prob, sol_prob_adj=bundle.sol_prob,
        has_info_all=bundle.has_info, ratio_pair_records=bundle.ratio_records,
    )

    # 自检：重算的旧口径应与当年落盘的数值一致
    check: Dict[str, Any] = {"available": False}
    stored = settings["stored_overall"]
    if isinstance(stored, dict):
        diffs = {
            k: abs(float(result["legacy"][k]) - float(stored[k]))
            for k in LEGACY_KEYS
            if k in stored
        }
        check = {
            "available": True,
            "max_abs_diff": max(diffs.values()) if diffs else 0.0,
            "per_metric_abs_diff": diffs,
            "passed": bool(diffs) and max(diffs.values()) < 1e-6,
        }
    result["reproduction_check"] = check
    result["settings"] = {k: v for k, v in settings.items() if k != "stored_overall"}
    return result


def recompute_baseline(base_dir: str, model_name: str) -> Dict[str, Any]:
    """
    从 baseline 已落盘的 `predictions_full.npz` 重算双口径指标。

    数组拼装方式与 `compute_baseline_overall.py:141-215` 一致：以 Cartridge 的
    `sample_indices` 为基准行序，Step/Solvent 按样本 id 对齐（缺失的即 has_info=0）。
    """
    def load(task: str) -> Dict[str, np.ndarray]:
        path = os.path.join(base_dir, f"{model_name}_{task}", "predictions_full.npz")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    car, step, sol, ratio = load("Cartridge"), load("Step"), load("Solvent"), load("Ratio")

    test_indices = car["sample_indices"].astype(np.int32)
    n = int(test_indices.shape[0])
    idx2row = {int(sid): i for i, sid in enumerate(test_indices.tolist())}
    n_steps = len(STEP_NAMES)

    step_true = np.zeros((n, n_steps), dtype=np.uint8)
    step_pred = np.zeros((n, n_steps), dtype=np.uint8)
    step_prob = np.zeros((n, n_steps), dtype=np.float32)
    has_info = np.zeros((n,), dtype=np.int32)
    for k, sid in enumerate(step["sample_indices"].astype(np.int32).tolist()):
        i = idx2row.get(sid)
        if i is None:
            continue
        step_true[i] = step["y_true"][k]
        step_pred[i] = step["y_pred"][k]
        step_prob[i] = step["y_score"][k]
        has_info[i] = 1

    n_solvent = int(sol["y_true"].shape[1]) // n_steps
    sol_true = np.zeros((n, n_steps, n_solvent), dtype=np.uint8)
    sol_pred = np.zeros((n, n_steps, n_solvent), dtype=np.uint8)
    sol_prob = np.zeros((n, n_steps, n_solvent), dtype=np.float32)
    for k, sid in enumerate(sol["sample_indices"].astype(np.int32).tolist()):
        i = idx2row.get(sid)
        if i is None:
            continue
        sol_true[i] = sol["y_true"][k].reshape(n_steps, n_solvent)
        sol_pred[i] = sol["y_pred"][k].reshape(n_steps, n_solvent)
        sol_prob[i] = sol["y_score"][k].reshape(n_steps, n_solvent)

    records: List[Tuple[int, int, int, float, float]] = []
    for k in range(int(ratio["sample_indices"].shape[0])):
        i = idx2row.get(int(ratio["sample_indices"][k]))
        if i is None:
            continue
        records.append((i, int(ratio["step_ids"][k]), int(ratio["solvent_ids"][k]),
                        float(ratio["y_true"][k]), float(ratio["y_pred"][k])))

    return compute_overall_metrics_v2(
        car_true=car["y_true"], car_pred=car["y_pred"], car_prob=car["y_score"],
        step_true=step_true, step_pred=step_pred,
        sol_true=sol_true, sol_pred=sol_pred,
        step_prob=step_prob, sol_prob_adj=sol_prob,
        has_info_all=has_info, ratio_pair_records=records,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="对已有实验/baseline 重算干净口径端到端指标")
    p.add_argument("--run_dir", type=str, action="append", default=None,
                   help="历史 MTL 实验目录，可重复传多次")
    p.add_argument("--baseline_dir", type=str, default=None,
                   help="baseline 输出根目录，例如 output/baseline_output")
    p.add_argument("--baseline_models", type=str,
                   default="DecisionTree,RandomForest,SVM,XGBoost,LightGBM,MLP")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    if not args.run_dir and not args.baseline_dir:
        p.error("至少要给 --run_dir 或 --baseline_dir 之一")

    for run_dir in args.run_dir or []:
        print(f"\n{'=' * 70}\n{run_dir}\n{'=' * 70}", flush=True)
        res = recompute_run(run_dir, device_str=args.device)
        print(compare_legacy_vs_clean(res), flush=True)
        chk = res["reproduction_check"]
        if chk["available"]:
            status = "通过" if chk["passed"] else "不一致"
            print(f"\n旧口径复现自检：{status}（最大绝对误差 {chk['max_abs_diff']:.2e}）", flush=True)
        else:
            print("\n旧口径复现自检：跳过（该目录没有可比对的 test_metrics.json）", flush=True)
        save_json(os.path.join(run_dir, "test_metrics_clean.json"), res)
        print(f"已写入 {os.path.join(run_dir, 'test_metrics_clean.json')}", flush=True)

    if args.baseline_dir:
        models = [m.strip() for m in args.baseline_models.split(",") if m.strip()]
        summary: Dict[str, Any] = {}
        for m in models:
            car_npz = os.path.join(args.baseline_dir, f"{m}_Cartridge", "predictions_full.npz")
            if not os.path.exists(car_npz):
                print(f"跳过 {m}：找不到 {car_npz}", flush=True)
                continue
            print(f"\n{'=' * 70}\nbaseline: {m}\n{'=' * 70}", flush=True)
            res = recompute_baseline(args.baseline_dir, m)
            print(compare_legacy_vs_clean(res), flush=True)
            summary[m] = res
        out_path = os.path.join(args.baseline_dir, "overall_metrics_clean.json")
        save_json(out_path, summary)
        print(f"\n已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
