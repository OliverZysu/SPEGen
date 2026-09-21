"""
同一数据划分上的深度集成：把若干个只有初始化种子不同的模型的概率平均起来。

**为什么这不是作弊，而是把对比拉正**：RandomForest 是 300 棵树的投票，
XGBoost 是 400 轮提升，LightGBM 是 500 轮。拿**一个**神经网络去和 300 棵树的集体
比较，本来就不是同等的模型容量/方差水平。k 个网络的集成才是森林的自然对应物。
所以本工具报出的结果应当和单模型结果并列呈现，让读者看到两种口径。

严格保证的两件事：

1. 所有成员共用**同一个数据划分**（`--seed` 相同，只有 `--init_seed` 不同）。
   这一点会被硬校验：任何成员的 split 或标签不一致就直接报错。
   注意历史上那种"跑三个 --seed"的做法**不能**用来做集成，
   因为 seed 同时决定划分，三个 run 的测试集根本不是同一批样本。
2. 阈值与解码配置在**平均后的验证集概率**上重新调优，测试集不参与任何选择。

用法：
  python3 -m spe.tools.deep_ensemble \
      --run_dir output/ens/M_init1 --run_dir output/ens/M_init2 --run_dir output/ens/M_init3 \
      --out_json output/ens/ensemble_metrics.json
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

from utils.io import save_json

from ..decode import DecodeConfig, tune_cartridge_decode, tune_per_label_thresholds, tune_solvent_decode
from ..evaluate import PredictionBundle, evaluate, selection_score
from ..metrics_strict import compare_legacy_vs_clean
from .loadrun import collect_split, load_run

# 这些 args 决定了模型看到什么数据、算什么指标。成员之间必须完全一致，
# 否则平均出来的概率没有共同含义。init_seed 是唯一允许不同的。
MUST_MATCH = (
    "seed", "train_ratio", "val_ratio", "data_dir", "cache_path", "require_spe_info",
    "unk_solvent_token", "no_functional_groups", "fg_transform", "fg_min_pos",
    "no_decompose_solvent", "feature_transform", "bootstrap", "bootstrap_frac",
)


def _assert_compatible(all_args: List[Dict[str, object]], dirs: List[str]) -> None:
    ref = all_args[0]
    for a, d in zip(all_args[1:], dirs[1:]):
        for k in MUST_MATCH:
            if a.get(k) != ref.get(k):
                raise AssertionError(
                    f"{d} 的 {k}={a.get(k)!r} 与 {dirs[0]} 的 {ref.get(k)!r} 不一致。"
                    f"集成成员必须共用同一个数据划分与特征配置。"
                )
        if a.get("init_seed") == ref.get("init_seed") and a.get("seed") == ref.get("seed"):
            print(f"[WARN] {d} 与 {dirs[0]} 的 init_seed 相同，这两个成员大概率完全一样，"
                  f"集成不会带来任何多样性")


def _assert_same_labels(bundles: List[PredictionBundle], dirs: List[str], split: str) -> None:
    """标签逐位相同 —— 这是"同一批样本、同一顺序"最直接的证据。"""
    ref = bundles[0]
    for b, d in zip(bundles[1:], dirs[1:]):
        for name, x, y in (
            ("cartridge", ref.car_true, b.car_true),
            ("step", ref.step_true, b.step_true),
            ("solvent", ref.sol_true, b.sol_true),
            ("has_info", ref.has_info, b.has_info),
        ):
            if x.shape != y.shape or not np.array_equal(x, y):
                raise AssertionError(f"{split} 上 {d} 的 {name} 标签与 {dirs[0]} 不一致")
        if ref.cids != b.cids:
            raise AssertionError(f"{split} 上 {d} 的样本顺序与 {dirs[0]} 不一致")


def average_bundles(bundles: List[PredictionBundle]) -> PredictionBundle:
    """
    概率算术平均。个数预测也平均。

    ratio 按 (样本, step, solvent) 键对齐后平均，而不是按记录下标 ——
    虽然成员同划分时顺序本该一致，但按键对齐能让这个假设失效时立刻暴露。
    """
    ref = bundles[0]
    k = float(len(bundles))

    car_prob = np.mean([b.car_prob for b in bundles], axis=0)
    step_prob = np.mean([b.step_prob for b in bundles], axis=0)
    sol_prob = np.mean([b.sol_prob for b in bundles], axis=0)

    car_cnt = None
    sol_cnt = None
    if all(b.car_count_pred is not None for b in bundles):
        car_cnt = np.mean([b.car_count_pred for b in bundles], axis=0)
        sol_cnt = np.mean([b.sol_count_pred for b in bundles], axis=0)

    # 个数分布按成员平均。平均分布再取分位数，和"各成员各自取分位数再平均"不同，
    # 前者才是集成该有的行为：先合成一个更可信的分布，再从它上面读数。
    car_dist = None
    sol_dist = None
    if all(b.car_count_dist is not None for b in bundles):
        car_dist = np.mean([b.car_count_dist for b in bundles], axis=0)
        sol_dist = np.mean([b.sol_count_dist for b in bundles], axis=0)

    keys = [(r[0], r[1], r[2]) for r in ref.ratio_records]
    truth = {(r[0], r[1], r[2]): r[3] for r in ref.ratio_records}
    acc: Dict[tuple, float] = {kk: 0.0 for kk in keys}
    for b in bundles:
        seen = set()
        for sid, st, so, _y_t, y_p in b.ratio_records:
            kk = (sid, st, so)
            if kk not in acc:
                raise AssertionError(f"ratio 记录 {kk} 在成员之间不一致，无法对齐")
            # 同一个键可能有多条记录（同一 pair 出现多次），这里按出现次数平均
            acc[kk] += float(y_p)
            seen.add(kk)
        missing = set(acc) - seen
        if missing:
            raise AssertionError(f"某个成员缺失 {len(missing)} 条 ratio 记录，无法对齐")

    records = [(kk[0], kk[1], kk[2], truth[kk], acc[kk] / k) for kk in keys]
    return PredictionBundle(
        car_true=ref.car_true, car_prob=car_prob.astype(np.float32),
        step_true=ref.step_true, step_prob=step_prob.astype(np.float32),
        sol_true=ref.sol_true, sol_prob=sol_prob.astype(np.float32),
        has_info=ref.has_info,
        ratio_records=records,
        ratio_true=np.asarray([r[3] for r in records], dtype=np.float32),
        ratio_pred=np.asarray([r[4] for r in records], dtype=np.float32),
        cids=ref.cids, method_lists=ref.method_lists,
        car_count_pred=car_cnt, sol_count_pred=sol_cnt,
        car_count_dist=car_dist, sol_count_dist=sol_dist,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="同划分深度集成")
    p.add_argument("--run_dir", type=str, action="append", required=True)
    p.add_argument("--decode_objective", type=str, default=None,
                   help="解码调优目标，默认沿用第一个成员的设置")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    dirs: List[str] = list(args.run_dir)
    if len(dirs) < 2:
        raise SystemExit("集成至少需要 2 个成员")

    loaded = [load_run(d, device_str=args.device) for d in dirs]
    _assert_compatible([l.args for l in loaded], dirs)
    print(f"[OK] {len(dirs)} 个成员共用同一数据划分（seed={loaded[0].args['seed']}），"
          f"init_seed={[l.args.get('init_seed') for l in loaded]}")

    val_bundles = [collect_split(l, "val", args.device) for l in loaded]
    test_bundles = [collect_split(l, "test", args.device) for l in loaded]
    _assert_same_labels(val_bundles, dirs, "验证集")
    _assert_same_labels(test_bundles, dirs, "测试集")
    print("[OK] 标签与样本顺序逐位一致")

    val_avg = average_bundles(val_bundles)
    test_avg = average_bundles(test_bundles)

    a0 = loaded[0].args
    init_thr = {
        "cartridge": float(a0.get("thr_cartridge", 0.5)),
        "step": float(a0.get("thr_step", 0.5)),
        "solvent": float(a0.get("thr_solvent", 0.5)),
    }
    objective = args.decode_objective or str(a0.get("decode_objective", "em_jaccard"))
    quantiles: List[Optional[float]] = [None]
    if a0.get("tune_count_quantile"):
        quantiles += [float(q) for q in str(a0.get("count_quantiles", "")).split(",") if q.strip()]

    # 阈值与解码全部在**平均后的验证集**上重新调优
    per_label = tune_per_label_thresholds(
        car_true=val_avg.car_true, car_prob=val_avg.car_prob,
        step_true=val_avg.step_true, step_prob=val_avg.step_prob,
        sol_true=val_avg.sol_true, sol_prob=val_avg.sol_prob,
        has_info=val_avg.has_info, init=init_thr,
        grid_num=int(a0.get("thr_grid_num", 41)),
        min_pos=int(a0.get("thr_min_pos_per_label", 5)),
    )
    car_cfg, car_obj, _ = tune_cartridge_decode(
        car_true=val_avg.car_true, car_prob=val_avg.car_prob,
        car_thr=per_label["cartridge"], car_count_pred=val_avg.car_count_pred,
        objective=objective,
        car_count_dist=val_avg.car_count_dist, count_quantiles=quantiles,
    )
    val_step_pred = (val_avg.step_prob >= per_label["step"][None, :]).astype(np.int32)
    sol_cfg, sol_gate, sol_obj, _ = tune_solvent_decode(
        sol_true=val_avg.sol_true, sol_prob=val_avg.sol_prob,
        sol_thr=per_label["solvent"], sol_count_pred=val_avg.sol_count_pred,
        has_info=val_avg.has_info, step_pred=val_step_pred, objective=objective,
        sol_count_dist=val_avg.sol_count_dist, count_quantiles=quantiles,
    )
    print(f"解码调优（目标={objective}）: car={car_cfg.as_dict()} 目标值={car_obj:.4f}")
    print(f"                            sol={sol_cfg.as_dict()} gate={sol_gate} 目标值={sol_obj:.4f}")

    sel_weights = {
        "strict": float(a0.get("sel_w_strict", 3.0)), "sss": float(a0.get("sel_w_sss", 1.5)),
        "hc": float(a0.get("sel_w_hc", 1.2)), "chain": float(a0.get("sel_w_chain", 4.0)),
        "gus": float(a0.get("sel_w_gus", 1.0)), "car_top1": float(a0.get("sel_w_car_top1", 0.0)),
    }
    val_metrics = evaluate(val_avg, init_thr, per_label_thresholds=per_label,
                           cartridge_decode=car_cfg, solvent_decode=sol_cfg,
                           solvent_gate_by_step=sol_gate)
    test_metrics = evaluate(test_avg, init_thr, per_label_thresholds=per_label,
                            cartridge_decode=car_cfg, solvent_decode=sol_cfg,
                            solvent_gate_by_step=sol_gate)

    print("\n=== 集成后的测试集 ===")
    print(compare_legacy_vs_clean({
        "legacy": test_metrics["overall"], "clean": test_metrics["overall_clean"],
        "diagnostics": test_metrics["hit1_diagnostics"],
    }))
    print("\n预测集合大小：", test_metrics["pred_set_size"])

    # 各成员单独的表现，用来看集成到底涨了多少
    print("\n=== 各成员单独 vs 集成（链路正确率 / Car top-1 / GUS）===")
    for d, b in zip(dirs, test_bundles):
        m = evaluate(b, init_thr, per_label_thresholds=per_label,
                     cartridge_decode=car_cfg, solvent_decode=sol_cfg,
                     solvent_gate_by_step=sol_gate)
        print(f"  {os.path.basename(d):<28}"
              f"{m['overall_clean']['chain_acc_no_ratio']:.4f}  "
              f"{m['overall_clean']['cartridge_top1_acc']:.4f}  "
              f"{m['overall']['global_utility_score']:.2f}")
    print(f"  {'>>> 集成':<28}"
          f"{test_metrics['overall_clean']['chain_acc_no_ratio']:.4f}  "
          f"{test_metrics['overall_clean']['cartridge_top1_acc']:.4f}  "
          f"{test_metrics['overall']['global_utility_score']:.2f}")

    if args.out_json:
        save_json(args.out_json, {
            "members": dirs,
            "metrics": test_metrics,
            "val_metrics_tuned": val_metrics,
            "val_select_score_tuned": selection_score(val_metrics, sel_weights),
            "decode": {"objective": objective, "cartridge": car_cfg.as_dict(),
                       "solvent": {**sol_cfg.as_dict(), "gate_by_step": sol_gate}},
            "note": "同一数据划分（seed 固定）、仅初始化不同的深度集成；"
                    "阈值与解码在平均后的验证集上调优，测试集未参与任何选择。",
        })
        print(f"\n[SAVE] {args.out_json}")


if __name__ == "__main__":
    main()
