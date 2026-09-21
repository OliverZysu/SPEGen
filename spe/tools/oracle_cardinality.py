"""
基数上限诊断：如果集合大小预测头做到完美，指标能涨到哪里？

    python -m spe.tools.oracle_cardinality --run_dir output_mtl_v6_F_full_seed42

对同一个 checkpoint 用五种解码方式评估同一个测试集：

1. `阈值`         —— 纯逐标签阈值，不用任何集合大小信息
2. `预测基数`     —— 用模型自己回归出来的集合大小（即训练时实际采用的方式）
3. `分位校准基数` —— 把预测个数做分位映射，使其**边缘分布**与验证集真实分布一致，
                    但**逐样本的相对排序保持不变**
4. `真实基数`     —— 用测试集的真实集合大小（oracle，实际不可得）
5. `真实基数+完美排序` —— 进一步假设成员排序也完美，给出理论天花板

这五档构成一条递进的归因链：

- 2 → 3 的差距 = 单纯把个数的**尺度/分布**校准对能拿到多少（不需要模型变强，纯后处理）
- 3 → 4 的差距 = 剩下的部分必须靠提升**逐样本个数判别力**（相关系数）才能拿到
- 4 → 5 的差距 = 候选成员的**排序质量**还欠多少

它回答的核心问题是：步骤 2 收益有限，是因为集合大小头不准（值得继续投入），
还是因为候选排序本身不行（那么个数再准也救不回来）。
"""

import argparse
import copy
from typing import Any, Dict, List, Optional

import numpy as np

from ..decode import DecodeConfig
from ..evaluate import PredictionBundle, evaluate
from ..metrics_strict import compute_overall_metrics_v2
from .loadrun import collect_split, load_run


def _summ(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Car F1": metrics["cartridge"]["micro_f1"],
        "Car EM": metrics["cartridge"]["exact_match_rate"],
        "Sol F1": metrics["solvent"]["micro_f1"],
        "Sol EM": metrics["solvent"]["exact_match_rate"],
        "链路": metrics["overall_clean"]["chain_acc_no_ratio"],
        "StrictAcc(有信息)": metrics["overall_clean"]["strict_acc_info_only"],
        "SSS": metrics["overall"]["scheme_similarity_score"],
        "HC": metrics["overall"]["hierarchical_consistency_acc"],
        "GUS": metrics["overall"]["global_utility_score"],
        "Car集合大小": metrics["pred_set_size"]["cartridge_pred_mean"],
        "Sol集合大小": metrics["pred_set_size"]["solvent_pred_mean"],
    }


def _with_counts(
    bundle: PredictionBundle,
    car_count: Optional[np.ndarray],
    sol_count: Optional[np.ndarray],
) -> PredictionBundle:
    """浅拷贝一个 bundle，只替换集合大小，用来做 oracle 对照。"""
    b = copy.copy(bundle)
    b.car_count_pred = car_count
    b.sol_count_pred = sol_count
    return b


def quantile_calibrate(
    pred_test: np.ndarray,
    pred_fit: np.ndarray,
    true_fit: np.ndarray,
) -> np.ndarray:
    """
    分位映射：保持 `pred_test` 的相对排序不变，把它的边缘分布拉到 `true_fit` 的分布上。

    做法是先算每个测试样本在 `pred_fit`（验证集预测值）里的百分位，
    再取 `true_fit`（验证集真实个数）在同一百分位上的取值。

    这样"预测个数偏小 / 方差偏小"这类系统性尺度问题会被完全修正，
    而"哪个样本的集合更大"这个判别问题完全没有改善 —— 正好把两者分开。
    """
    pred_fit_sorted = np.sort(np.asarray(pred_fit, dtype=np.float64).reshape(-1))
    true_fit_sorted = np.sort(np.asarray(true_fit, dtype=np.float64).reshape(-1))
    if pred_fit_sorted.size == 0 or true_fit_sorted.size == 0:
        return np.asarray(pred_test, dtype=np.float64)

    flat = np.asarray(pred_test, dtype=np.float64).reshape(-1)
    # searchsorted 给出的是排名，除以总数得到百分位
    ranks = np.searchsorted(pred_fit_sorted, flat, side="left").astype(np.float64)
    pct = np.clip(ranks / max(1.0, float(pred_fit_sorted.size - 1)), 0.0, 1.0)
    idx = np.clip(
        np.round(pct * (true_fit_sorted.size - 1)).astype(np.int64), 0, true_fit_sorted.size - 1
    )
    return true_fit_sorted[idx].reshape(np.asarray(pred_test).shape)


def perfect_ranking_ceiling(bundle: PredictionBundle) -> Dict[str, float]:
    """
    天花板：用真实集合大小 k，且假设排序完美（即 top-k 恰好就是真实标签）。

    这等价于直接把预测置为真实标签，所以 Car EM / Sol EM 必然是 1.0。
    列出来是为了给"排序还有多少空间"一个参照上界，而不是一个可追求的目标。
    """
    car_pred = bundle.car_true.astype(np.int32)
    sol_pred = bundle.sol_true.astype(np.int32)
    step_pred = bundle.step_true.astype(np.int32)
    v2 = compute_overall_metrics_v2(
        car_true=bundle.car_true, car_pred=car_pred, car_prob=bundle.car_prob,
        step_true=bundle.step_true, step_pred=step_pred,
        sol_true=bundle.sol_true, sol_pred=sol_pred,
        step_prob=bundle.step_prob, sol_prob_adj=bundle.sol_prob,
        has_info_all=bundle.has_info, ratio_pair_records=bundle.ratio_records,
    )
    mask = bundle.has_info.astype(np.int32) == 1
    return {
        "Car F1": 1.0, "Car EM": 1.0, "Sol F1": 1.0, "Sol EM": 1.0,
        "链路": v2["clean"]["chain_acc_no_ratio"],
        "StrictAcc(有信息)": v2["clean"]["strict_acc_info_only"],
        "SSS": v2["legacy"]["scheme_similarity_score"],
        "HC": v2["legacy"]["hierarchical_consistency_acc"],
        "GUS": v2["legacy"]["global_utility_score"],
        "Car集合大小": float(car_pred.sum(axis=1).mean()),
        "Sol集合大小": float(sol_pred[mask].sum(axis=(1, 2)).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="集合大小预测的上限诊断")
    p.add_argument("--run_dir", type=str, action="append", required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--k_scale", type=float, default=1.0,
                   help="oracle 之外的两档也统一用这个缩放，默认 1.0 便于对齐")
    p.add_argument("--json", type=str, default=None,
                   help="把结果写成 JSON，供论文表格直接读取，避免手抄数字")
    args = p.parse_args()

    payload: Dict[str, Any] = {}

    for run_dir in args.run_dir:
        loaded = load_run(run_dir, device_str=args.device)
        bundle = collect_split(loaded, "test", device_str=args.device)
        thr = loaded.per_label_thresholds
        init = loaded.init_thresholds

        car_true_cnt = bundle.car_true.sum(axis=1).astype(np.float64)
        sol_true_cnt = bundle.sol_true.sum(axis=2).astype(np.float64)

        rows: List[tuple] = []
        rows.append(("阈值", evaluate(
            bundle, init, per_label_thresholds=thr,
            cartridge_decode=DecodeConfig(mode="threshold"),
            solvent_decode=DecodeConfig(mode="threshold"),
            solvent_gate_by_step=True,
        )))

        if bundle.has_counts:
            rows.append(("预测基数", evaluate(
                bundle, init, per_label_thresholds=thr,
                cartridge_decode=DecodeConfig(mode="clamp", k_scale=args.k_scale, k_min=1),
                solvent_decode=DecodeConfig(mode="clamp", k_scale=args.k_scale, k_min=0),
                solvent_gate_by_step=True,
            )))

            # 分位映射的参数只在验证集上拟合，避免用到测试集的真实个数
            val = collect_split(loaded, "val", device_str=args.device)
            vm = val.has_info.astype(np.int32) == 1
            cal_bundle = _with_counts(
                bundle,
                quantile_calibrate(
                    bundle.car_count_pred, val.car_count_pred, val.car_true.sum(axis=1)
                ),
                quantile_calibrate(
                    bundle.sol_count_pred,
                    val.sol_count_pred[vm].reshape(-1),
                    val.sol_true.sum(axis=2)[vm].reshape(-1),
                ),
            )
            rows.append(("分位校准基数", evaluate(
                cal_bundle, init, per_label_thresholds=thr,
                cartridge_decode=DecodeConfig(mode="cardinality", k_scale=1.0, k_min=1),
                solvent_decode=DecodeConfig(mode="cardinality", k_scale=1.0, k_min=0),
                solvent_gate_by_step=True,
            )))

        oracle_bundle = _with_counts(bundle, car_true_cnt, sol_true_cnt)
        rows.append(("真实基数(oracle)", evaluate(
            oracle_bundle, init, per_label_thresholds=thr,
            cartridge_decode=DecodeConfig(mode="cardinality", k_scale=1.0, k_min=1),
            solvent_decode=DecodeConfig(mode="cardinality", k_scale=1.0, k_min=0),
            solvent_gate_by_step=True,
        )))

        print("\n" + "=" * 100)
        print(run_dir)
        print("=" * 100)

        summaries = [(name, _summ(m)) for name, m in rows]
        summaries.append(("真实基数+完美排序", perfect_ranking_ceiling(bundle)))

        keys = list(summaries[0][1].keys())
        w = 20
        print(f"{'指标':<20}" + "".join(f"{n:>{w}}" for n, _ in summaries))
        print("-" * (20 + w * len(summaries)))
        for k in keys:
            cells = ""
            for _, s in summaries:
                v = s.get(k)
                nd = 2 if ("集合大小" in k or k == "GUS") else 4
                cells += f"{v:>{w}.{nd}f}" if v is not None else f"{'-':>{w}}"
            print(f"{k:<20}{cells}")

        print(
            "\n读法：'预测基数' 与 '真实基数(oracle)' 的差距 = 把集合大小头做准还能拿到的收益；"
            "\n      '真实基数(oracle)' 与 '真实基数+完美排序' 的差距 = 候选排序本身还欠多少。"
        )

        payload[run_dir] = {name: dict(summary) for name, summary in summaries}

    if args.json:
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
