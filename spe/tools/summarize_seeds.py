"""
多 seed 汇总：把若干个 run 目录的 `test_metrics.json` 聚成 `均值 ± 标准差`。

    python -m spe.tools.summarize_seeds --run_dir output_mtl_v6_s1_seed42 \
        --run_dir output_mtl_v6_s1_seed7 --run_dir output_mtl_v6_s1_seed2024

为什么需要：现在所有结论都建立在单次 seed=42 的运行上。v21_s2 与 v22_s2 的差距
（SSS 0.4924 vs 0.4917、GUS 59.96 vs 59.79）只有小数点后第三位，很可能落在随机波动内 ——
不给出标准差就无法判断哪些改进是真的。建议每个配置至少 3 个 seed。

本工具同时兼容 v6 的新格式（含 `overall_clean`）和历史格式（只有 `overall`）。

主指标是"链路"（`chain_acc_no_ratio`）。`Hit@1 子集` 的分母随模型变化，所以一并打印分母，
不要单看它的比值 —— 详见 `spe/metrics_strict.py` 的说明。
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.io import save_json

# (展示名, 在 test_metrics.json 里的取值路径)
METRIC_PATHS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Car F1", ("cartridge", "micro_f1")),
    ("Car EM", ("cartridge", "exact_match_rate")),
    ("Step F1", ("step", "micro_f1")),
    ("Sol F1", ("solvent", "micro_f1")),
    ("Sol EM", ("solvent", "exact_match_rate")),
    ("Ratio MAE", ("ratio", "mae")),
    ("Ratio<=0.10", ("ratio", "within_0.10")),
    ("StrictAcc", ("overall", "strict_end_to_end_acc")),
    ("SSS", ("overall", "scheme_similarity_score")),
    ("HC", ("overall", "hierarchical_consistency_acc")),
    ("链路(主指标)", ("overall_clean", "chain_acc_no_ratio")),
    ("Hit@1 最严", ("overall_clean", "hit_at_1_strict")),
    ("Hit@1 旧口径", ("overall", "method_hit_at_1")),
    ("Hit@1 子集(慎用)", ("overall_clean", "hit_at_1_ratio_subset")),
    ("  └子集分母", ("hit1_diagnostics", "n_ratio_subset")),
    ("GUS", ("overall", "global_utility_score")),
    ("Car 预测集合大小", ("pred_set_size", "cartridge_pred_mean")),
    ("Sol 预测集合大小", ("pred_set_size", "solvent_pred_mean")),
]


def _dig(obj: Any, path: Tuple[str, ...]) -> Optional[float]:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, (int, float)) and np.isfinite(float(cur)):
        return float(cur)
    return None


def load_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "test_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # v6 与历史格式都把子任务指标放在 "metrics" 下
    return obj.get("metrics", obj)


def main() -> None:
    p = argparse.ArgumentParser(description="多 seed 结果汇总（均值 ± 标准差）")
    p.add_argument("--run_dir", type=str, action="append", required=True,
                   help="run 目录，可重复传多次")
    p.add_argument("--label", type=str, default=None, help="这一组配置的名字，用于输出标题")
    p.add_argument("--out_json", type=str, default=None, help="可选：把汇总结果写到该路径")
    args = p.parse_args()

    loaded: Dict[str, Dict[str, Any]] = {}
    for d in args.run_dir:
        m = load_metrics(d)
        if m is None:
            print(f"跳过 {d}：找不到 test_metrics.json", flush=True)
            continue
        loaded[d] = m
    if not loaded:
        raise SystemExit("没有任何可用的 run 目录")

    label = args.label or f"{len(loaded)} 个 seed"
    print(f"\n配置：{label}    有效 run 数：{len(loaded)}")
    for d in loaded:
        print(f"  - {d}")
    print()
    header = f"{'指标':<20}{'均值':>10}{'标准差':>10}{'最小':>10}{'最大':>10}   各 run 取值"
    print(header)
    print("-" * len(header))

    summary: Dict[str, Any] = {"runs": list(loaded.keys()), "label": label, "metrics": {}}
    for name, path in METRIC_PATHS:
        vals = [_dig(m, path) for m in loaded.values()]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        # 单个 run 时标准差没有意义，用 ddof=1 需要至少 2 个样本
        std = float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")
        summary["metrics"][name] = {
            "mean": float(arr.mean()), "std": std,
            "min": float(arr.min()), "max": float(arr.max()),
            "n": int(arr.size), "values": [float(v) for v in arr],
        }
        std_txt = "     n/a" if arr.size < 2 else f"{std:>10.4f}"
        detail = ", ".join(f"{v:.4f}" for v in arr)
        print(f"{name:<20}{arr.mean():>10.4f}{std_txt}{arr.min():>10.4f}{arr.max():>10.4f}   {detail}")

    if args.out_json:
        save_json(args.out_json, summary)
        print(f"\n已写入 {args.out_json}")


if __name__ == "__main__":
    main()
