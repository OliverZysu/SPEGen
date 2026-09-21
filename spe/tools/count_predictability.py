"""
集合大小到底可不可预测？

oracle 诊断（`spe.tools.oracle_cardinality`）说明：把集合大小预测准，Car F1 能从
0.48 涨到 0.65。这个脚本回答紧接着的问题 —— 这个上限能不能靠一个更好的
计数头拿到。

做法是绕开我们自己的多任务模型，直接用梯度提升树在同样的 21 维描述符上
专门回归集合大小。如果一个专门优化这一个目标的强模型都做不动，那么
多任务模型里的计数头也做不动，上限就是不可达的。

    python -m spe.tools.count_predictability --json output/count_predictability.json

报告 R^2（尺度是否对得上）和 Spearman（逐样本大小排序是否分得开）：
解码只用到 top-k 的 k，所以真正要紧的是后者。
"""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

from scipy import stats  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from spe.data import prepare_data  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="集合大小的可预测性上限")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prepared = prepare_data(
        data_dir="data", cache_path="data/cache/dataset_cache.pkl", seed=42,
        train_ratio=0.8, val_ratio=0.1, require_spe_info=False,
        unk_solvent_token="__UNK__", use_functional_groups=False,
        fg_transform="binary", fg_min_pos=10, decompose_solvent=False,
        feature_transform="rank_gauss",
    )
    ds = prepared.ds
    x = np.asarray(ds.xs, dtype=float)
    has_info = np.asarray(ds.has_spe_info).astype(bool)

    # Cartridge labels exist for every pollutant; solvent labels only for the
    # annotated subset, so the two targets use different index sets.
    targets = {
        "cartridge": (np.asarray(ds.y_cartridge).sum(axis=1).astype(float),
                      np.ones(len(x), dtype=bool)),
        "solvent": (np.asarray(ds.y_solvent).sum(axis=(1, 2)).astype(float),
                    has_info),
    }

    out: Dict[str, Dict[str, float]] = {}
    for name, (y, keep) in targets.items():
        train = np.asarray([i for i in prepared.train_idx if keep[i]])
        test = np.asarray([i for i in prepared.test_idx if keep[i]])
        model = HistGradientBoostingRegressor(random_state=args.seed)
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        out[name] = {
            "r2": float(r2_score(y[test], pred)),
            "spearman": float(stats.spearmanr(y[test], pred).statistic),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "true_mean": float(y[test].mean()),
            "true_sd": float(y[test].std()),
        }
        print(f"{name:<10} R2 = {out[name]['r2']:.3f}   "
              f"Spearman = {out[name]['spearman']:.3f}   "
              f"n_test = {out[name]['n_test']}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
