"""
步骤 1：修正端到端指标的口径泄漏。

`main.py` 里的 `compute_overall_metrics()` 有两处会让 Hit@1 / StrictAcc 虚高：

1. **无 ratio 标注免检**（`main.py:218-220`、`main_v4.py:158-161`）——
   top-1 链路选中的 `(step, solvent)` 若恰好没有 ratio 数值标注，就跳过 ratio 校验直接算命中。
   v22_s2 有 343/1145 个样本走这条免检通道，占其 429 次"命中"的 80%。
   于是"把 top-1 往标注稀疏的溶剂上偏移"就能免费涨分。

2. **无流程信息免检**（`main.py:214`）——
   `has_spe_info=0` 的样本（1145 中有 102 个）只要 cartridge top-1 命中就算整条链路命中，
   因为 step/solvent/ratio 三项根本不参与判定。

本模块给出与旧指标**并列输出**的干净版本，不改动旧函数，方便回归对照。
`compute_overall_metrics_v2()` 的入参与 `main.compute_overall_metrics()` 完全一致，
所以 MTL 模型和 `compute_baseline_overall.py` 里的 baseline 都能调用，保证同口径比较。
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from utils.metrics import multilabel_stats

# (sample_id, step_id, solvent_id, ratio_true, ratio_pred)
RatioRecord = Tuple[int, int, int, float, float]


def row_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """逐样本 Jaccard。与 `main._row_jaccard` 行为一致（union 为空时记 1.0）。"""
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    inter = np.logical_and(y_true == 1, y_pred == 1).sum(axis=1).astype(np.float64)
    uni = np.logical_or(y_true == 1, y_pred == 1).sum(axis=1).astype(np.float64)
    out = np.ones((y_true.shape[0],), dtype=np.float64)
    m = uni > 0
    out[m] = inter[m] / uni[m]
    return out


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size > 0 else float("nan")


def compute_overall_metrics_v2(
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
    ratio_pair_records: List[RatioRecord],
    ratio_tau: float = 0.10,
    w_car: float = 0.2,
    w_step: float = 0.2,
    w_sol: float = 0.4,
    w_ratio: float = 0.2,
) -> Dict[str, Any]:
    """
    同时计算旧口径（`legacy`）与干净口径（`clean`）的端到端指标。

    `legacy` 逐位复刻 `main.compute_overall_metrics()`，用于验证本模块接线正确 ——
    对同一个 run 调用本函数，`legacy` 各项应与该 run 的 `test_metrics.json` 完全一致。

    `clean` 里的 Hit@1 变体从宽到严依次是：

    - `hit_at_1_legacy`      : 旧口径，两处免检都保留（= `legacy.method_hit_at_1`）
    - `hit_at_1_info_only`   : 分母只算 has_info=1 的样本，去掉"无流程信息免检"
    - `hit_at_1_ratio_subset`: 只在 top-1 pair 确实有 ratio 标注的子集上算
    - `hit_at_1_strict`      : 分母为 has_info=1，缺 ratio 标注一律判失败（最保守）

    **跨模型比较请用 `chain_acc_no_ratio`（cartridge/step/solvent 三段 top-1 全对，
    分母恒为 has_info=1 的样本数）或 `hit_at_1_strict`。**

    `hit_at_1_ratio_subset` 不可跨模型比较：哪个 `(step, solvent)` 成为 top-1 由模型自己决定，
    所以它的分母 `n_ratio_subset` 是模型相关的（实测在 84~177 之间浮动）。
    分母越小该值越容易虚高 —— v16_s1 的 0.81 只建立在 84 个样本上，反而是最弱的模型。
    要引用这个值就必须同时给出 `n_ratio_subset`。
    """
    n = int(car_true.shape[0])
    has_info = has_info_all.astype(np.int32) == 1
    sol_pred = sol_pred.reshape(sol_true.shape)

    # ---- ratio 误差按样本 / 按 (样本, step, solvent) 归并 ----
    pair_abs_err: Dict[Tuple[int, int, int], List[float]] = {}
    sample_abs_err: Dict[int, List[float]] = {}
    for sid, st, so, y_t, y_p in ratio_pair_records:
        e = abs(float(y_p) - float(y_t))
        pair_abs_err.setdefault((int(sid), int(st), int(so)), []).append(e)
        sample_abs_err.setdefault(int(sid), []).append(e)

    ratio_ok_per_sample = np.ones((n,), dtype=np.float64)
    ratio_score_per_sample = np.ones((n,), dtype=np.float64)
    for sid, errs in sample_abs_err.items():
        errs_arr = np.asarray(errs, dtype=np.float64)
        ratio_ok_per_sample[sid] = float(np.all(errs_arr <= ratio_tau))
        ratio_score_per_sample[sid] = float(np.mean(np.maximum(0.0, 1.0 - errs_arr / ratio_tau)))

    # ---- 1) StrictAcc ----
    car_exact = np.all(car_true == car_pred, axis=1)
    step_exact = np.all(step_true == step_pred, axis=1)
    sol_exact = np.all(sol_true == sol_pred, axis=(1, 2))

    strict_ok = np.ones((n,), dtype=bool)
    strict_ok &= car_exact
    strict_ok[has_info] &= step_exact[has_info]
    strict_ok[has_info] &= sol_exact[has_info]
    strict_ok[has_info] &= ratio_ok_per_sample[has_info] >= 0.5
    strict_legacy = float(np.mean(strict_ok.astype(np.float64)))
    # 干净版：无流程信息的样本不该算"端到端全对"，从分母里剔除
    strict_info_only = _safe_mean(strict_ok[has_info].astype(np.float64))

    # ---- 2) SSS ----
    s_car = row_jaccard(car_true, car_pred)
    s_step = row_jaccard(step_true, step_pred)
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
        s_sol[i] = float(np.mean(step_scores)) if step_scores else 1.0

    sss_per_sample = (
        w_car * s_car
        + w_step * np.where(has_info, s_step, 1.0)
        + w_sol * np.where(has_info, s_sol, 1.0)
        + w_ratio * np.where(has_info, ratio_score_per_sample, 1.0)
    )
    sss_legacy = float(np.mean(sss_per_sample))
    sss_info_only = _safe_mean(sss_per_sample[has_info])

    # ---- 3) HC ----
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
        hc[i] = float(np.mean(step_scores)) if step_scores else 1.0
    hc_legacy = float(np.mean(hc))
    hc_info_only = _safe_mean(hc[has_info])

    # ---- 4) Hit@1（四个口径一次算完）----
    top1_car = np.argmax(car_prob, axis=1)
    top1_step = np.argmax(step_prob, axis=1)

    hit_legacy = np.zeros((n,), dtype=bool)
    # 链路前三段（cartridge -> step -> solvent）是否全对，与 ratio 无关
    chain_ok = np.zeros((n,), dtype=bool)
    # top-1 选中的 (step, solvent) 是否有 ratio 标注可校验
    ratio_available = np.zeros((n,), dtype=bool)
    ratio_ok = np.zeros((n,), dtype=bool)
    # 逐段 top-1 是否正确。分段记录是为了能把链路失败归因到具体某一段，
    # 也让 cartridge top-1 可以单独作为选模指标（做单任务诊断时用得上）。
    car_top1_ok = np.zeros((n,), dtype=bool)
    step_top1_ok = np.zeros((n,), dtype=bool)

    diag = {
        "cartridge_top1_wrong": 0,
        "step_top1_wrong": 0,
        "solvent_top1_wrong": 0,
        "ratio_top1_wrong": 0,
        "ratio_top1_not_available": 0,
        "no_info_auto_hit": 0,
    }

    for i in range(n):
        car_ok = bool(car_true[i, top1_car[i]] == 1)
        car_top1_ok[i] = car_ok
        if not car_ok:
            diag["cartridge_top1_wrong"] += 1
            continue

        if not has_info[i]:
            # 旧口径在这里直接记命中；干净口径把这些样本整体排除
            hit_legacy[i] = True
            diag["no_info_auto_hit"] += 1
            continue

        st = int(top1_step[i])
        step_top1_ok[i] = int(step_true[i, st]) == 1
        if not step_top1_ok[i]:
            diag["step_top1_wrong"] += 1
            continue

        so = int(np.argmax(sol_prob_adj[i, st]))
        if int(sol_true[i, st, so]) != 1:
            diag["solvent_top1_wrong"] += 1
            continue

        chain_ok[i] = True
        errs = pair_abs_err.get((int(i), st, so), None)
        if errs is None or len(errs) == 0:
            # 旧口径：跳过 ratio 校验并记命中
            diag["ratio_top1_not_available"] += 1
            hit_legacy[i] = True
            continue

        ratio_available[i] = True
        if float(np.mean(np.asarray(errs, dtype=np.float64))) <= ratio_tau:
            ratio_ok[i] = True
            hit_legacy[i] = True
        else:
            diag["ratio_top1_wrong"] += 1

    hit1_legacy = float(np.mean(hit_legacy.astype(np.float64)))
    hit1_info_only = _safe_mean(hit_legacy[has_info].astype(np.float64))
    # 最严：缺标注判失败
    hit1_strict = _safe_mean((chain_ok & ratio_ok)[has_info].astype(np.float64))
    # 最公平：只在可校验子集上算
    subset = has_info & ratio_available
    hit1_ratio_subset = _safe_mean((chain_ok & ratio_ok)[subset].astype(np.float64))
    chain_acc = _safe_mean(chain_ok[has_info].astype(np.float64))

    # ---- 5) GUS ----
    car_f1 = float(multilabel_stats(car_true, car_pred)["micro_f1"])
    if has_info.any():
        step_f1 = float(multilabel_stats(step_true[has_info], step_pred[has_info])["micro_f1"])
        n2, a2, b2 = sol_true.shape
        sol_f1 = float(
            multilabel_stats(
                sol_true.reshape(n2, a2 * b2)[has_info],
                sol_pred.reshape(n2, a2 * b2)[has_info],
            )["micro_f1"]
        )
    else:
        step_f1 = 0.0
        sol_f1 = 0.0

    all_errs = [e for errs in sample_abs_err.values() for e in errs]
    ratio_within = float(np.mean(np.asarray(all_errs) <= ratio_tau)) if all_errs else 0.0
    gus = float(100.0 * (0.25 * car_f1 + 0.25 * step_f1 + 0.35 * sol_f1 + 0.15 * ratio_within))

    return {
        "legacy": {
            "strict_end_to_end_acc": strict_legacy,
            "scheme_similarity_score": sss_legacy,
            "hierarchical_consistency_acc": hc_legacy,
            "method_hit_at_1": hit1_legacy,
            "global_utility_score": gus,
        },
        "clean": {
            "hit_at_1_legacy": hit1_legacy,
            "hit_at_1_info_only": hit1_info_only,
            "hit_at_1_ratio_subset": hit1_ratio_subset,
            "hit_at_1_strict": hit1_strict,
            "chain_acc_no_ratio": chain_acc,
            "strict_acc_info_only": strict_info_only,
            "scheme_similarity_info_only": sss_info_only,
            "hierarchical_consistency_info_only": hc_info_only,
            # 逐段 top-1，用来把链路失败归因到具体某一段。
            # `cartridge_top1_acc` 的分母是全部样本（cartridge 对无流程信息的样本也有标签），
            # 后两段的分母是"前面各段都对"的子集，所以是条件正确率。
            "cartridge_top1_acc": float(car_top1_ok.mean()),
            "step_top1_acc_given_car": (
                _safe_mean(step_top1_ok[has_info & car_top1_ok].astype(np.float64))
                if (has_info & car_top1_ok).any() else float("nan")
            ),
            "solvent_top1_acc_given_prefix": (
                _safe_mean(chain_ok[has_info & car_top1_ok & step_top1_ok].astype(np.float64))
                if (has_info & car_top1_ok & step_top1_ok).any() else float("nan")
            ),
        },
        "diagnostics": {
            "n_samples": n,
            "n_has_info": int(has_info.sum()),
            "n_no_info": int((~has_info).sum()),
            "n_ratio_subset": int(subset.sum()),
            "ratio_tau": float(ratio_tau),
            **diag,
        },
    }


def compare_legacy_vs_clean(result: Dict[str, Any]) -> str:
    """把 `compute_overall_metrics_v2()` 的结果渲染成一段可读的对照文本。"""
    lg = result["legacy"]
    cl = result["clean"]
    dg = result["diagnostics"]
    lines = [
        f"样本数 {dg['n_samples']}（有流程信息 {dg['n_has_info']}，无流程信息 {dg['n_no_info']}）",
        f"Hit@1 可校验子集大小 {dg['n_ratio_subset']}",
        "",
        f"Hit@1 旧口径（两处免检都在）      : {cl['hit_at_1_legacy']:.4f}",
        f"Hit@1 去掉无流程信息免检          : {cl['hit_at_1_info_only']:.4f}",
        f"Hit@1 仅可校验子集（推荐主指标）  : {cl['hit_at_1_ratio_subset']:.4f}",
        f"Hit@1 缺 ratio 标注判失败（最严） : {cl['hit_at_1_strict']:.4f}",
        f"前三段链路正确率（不看 ratio）    : {cl['chain_acc_no_ratio']:.4f}",
        "",
        f"StrictAcc 旧口径 / 仅有流程信息   : {lg['strict_end_to_end_acc']:.4f} / {cl['strict_acc_info_only']:.4f}",
        f"SSS 旧口径 / 仅有流程信息         : {lg['scheme_similarity_score']:.4f} / {cl['scheme_similarity_info_only']:.4f}",
        f"HC 旧口径 / 仅有流程信息          : {lg['hierarchical_consistency_acc']:.4f} / {cl['hierarchical_consistency_info_only']:.4f}",
        f"GUS                               : {lg['global_utility_score']:.4f}",
        "",
        "Hit@1 失败原因（对全部样本计数，互斥）：",
        f"  cartridge top-1 错     : {dg['cartridge_top1_wrong']}",
        f"  step top-1 错          : {dg['step_top1_wrong']}",
        f"  solvent top-1 错       : {dg['solvent_top1_wrong']}",
        f"  ratio 超差             : {dg['ratio_top1_wrong']}",
        f"  ratio 无标注(旧口径免检): {dg['ratio_top1_not_available']}",
        f"  无流程信息(旧口径免检)  : {dg['no_info_auto_hit']}",
    ]
    return "\n".join(lines)
