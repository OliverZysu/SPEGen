"""
步骤 2（解码侧）：基数感知解码，以及面向 exact-match / Jaccard 的调优。

为什么需要这个：最终模型 v22_s2 在测试集上每样本预测 5.97 个 cartridge，而真实平均只有
3.27 个（1.82 倍过预测）；solvent 是 24.36 vs 15.24（1.60 倍）。precision 0.358 / 0.366
而 recall 0.653 / 0.584。StrictAcc 要求集合完全一致，集合大 1.8 倍就几乎不可能全对。

历史上 v10~v22 一直用 **micro-F1** 作为阈值搜索目标（`main.tune_thresholds_per_label_on_val`
和 `main.tune_thresholds_on_val` 都是），而 F1 天然鼓励高召回，所以怎么调都压不下过预测。
v17 那四组"只调阈值"的实验全部退回 GUS 52 档，已经证明单靠阈值走不通。

本模块提供两件事：

1. **三种解码模式**（`DecodeMode`）—— 除了旧的逐标签阈值，加入"用预测个数取 top-k"
   和"阈值集合再按预测个数截断"。
2. **正确的调优目标** —— 可以直接对 exact-match 或 Jaccard 调优，而不是只能对 F1 调优。

一个重要性质：SSS 是各子任务得分的**线性加权和**（`w_car*s_car + w_step*s_step + ...`），
且 `s_car` 只取决于 cartridge 的解码方式。所以分任务独立调优就是全局最优，
不需要做联合网格搜索 —— 这让搜索空间从乘法降到加法。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.metrics import multilabel_stats

DecodeMode = str  # "threshold" | "cardinality" | "clamp"
VALID_MODES = ("threshold", "cardinality", "clamp")


@dataclass
class DecodeConfig:
    """一个任务（cartridge 或 solvent）的解码配置。"""

    mode: DecodeMode = "threshold"
    # 预测个数的缩放系数：k_allow = ceil(k_pred * k_scale)
    k_scale: float = 1.0
    # k 的下界。cartridge 用 1（每个样本总要推荐至少一种柱子）；
    # solvent 必须用 0，因为真实不存在的步骤就该一个溶剂都不出，这是 HC 指标的要求。
    k_min: int = 0
    k_max: int = 64
    # 全局阈值偏移，叠加在 per-label 阈值之上（正数=更严格）
    thr_shift: float = 0.0
    # 是否在预测集合为空时兜底取 top-1。
    # 默认 False 以对齐 `main.run_eval` 计算指标时的行为 —— 那里的空集合兜底只作用于
    # 展示用的 `car_candidates` / `solvents_by_step`，并不进入指标数组。
    ensure_nonempty: bool = False
    # 个数分类模式下，从预测的个数分布里取哪个分位数作为 k。None 表示不走分位数路径
    #（此时用模型自己给的标量个数，也就是众数或期望）。
    # 众数和期望都是塌缩的：众数恒为 1（真实分布里 508/1145 个样本只有 1 个 cartridge），
    # 期望被长尾拉到 4 左右。真正合适的取值在两者之间，所以让它在验证集上可搜。
    count_quantile: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "k_scale": self.k_scale,
            "k_min": self.k_min,
            "k_max": self.k_max,
            "thr_shift": self.thr_shift,
            "ensure_nonempty": self.ensure_nonempty,
            "count_quantile": self.count_quantile,
        }


def topk_mask(prob: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    """
    逐行取 top-k 的 0/1 掩码。prob: (M, L)，k_vec: (M,)。

    用秩比较实现，避免 Python 层循环。
    """
    m, l = prob.shape
    order = np.argsort(-prob, axis=1)
    ranks = np.empty((m, l), dtype=np.int32)
    np.put_along_axis(ranks, order, np.tile(np.arange(l, dtype=np.int32), (m, 1)), axis=1)
    k = np.clip(k_vec.astype(np.int32), 0, l)
    return (ranks < k[:, None]).astype(np.int32)


def counts_to_k(
    count_pred: np.ndarray,
    k_scale: float,
    k_min: int,
    k_max: int,
) -> np.ndarray:
    """把回归出来的连续个数变成整数 k，并夹到 [k_min, k_max]。"""
    k = np.ceil(np.asarray(count_pred, dtype=np.float64) * float(k_scale))
    return np.clip(k, int(k_min), int(k_max)).astype(np.int32)


def counts_from_dist(count_dist: np.ndarray, quantile: float) -> np.ndarray:
    """
    从个数分布里按分位数取个数。`count_dist` 最后一维是桶（桶下标即个数），其余维是样本轴。

    逐样本地在累积分布上找第一个超过 `quantile` 的桶。`quantile=0` 附近趋向最小的
    有概率的个数，`quantile=1` 附近趋向最大的，中间连续过渡 ——
    所以它把"众数"和"期望"这两种塌缩之间的整段区间都打开了，可以在验证集上搜。
    """
    d = np.asarray(count_dist, dtype=np.float64)
    cdf = np.cumsum(d, axis=-1)
    cdf = cdf / np.clip(cdf[..., -1:], 1e-12, None)
    return np.argmax(cdf >= float(quantile), axis=-1).astype(np.float64)


def decode_2d(
    prob: np.ndarray,
    thr_arr: np.ndarray,
    count_pred: Optional[np.ndarray],
    cfg: DecodeConfig,
) -> np.ndarray:
    """
    解码一个 (M, L) 的概率矩阵成 0/1 预测。

    - `threshold`   : 旧行为，逐标签过阈值
    - `cardinality` : 完全忽略阈值，直接取 top-k（k 来自集合大小预测头）
    - `clamp`       : 先过阈值，若集合比预测个数大则截断到 top-k（推荐）

    只有 `cfg.ensure_nonempty=True` 时才会对空集合兜底取 top-1，默认不兜底。
    """
    if cfg.mode not in VALID_MODES:
        raise ValueError(f"未知解码模式 {cfg.mode}，可选 {VALID_MODES}")

    m, l = prob.shape
    thr = np.asarray(thr_arr, dtype=np.float32)
    if thr.ndim == 0:
        thr = np.full((l,), float(thr), dtype=np.float32)
    thr = np.clip(thr + float(cfg.thr_shift), 0.0, 1.0)

    if cfg.mode == "cardinality":
        if count_pred is None:
            raise ValueError("cardinality 模式需要集合大小预测，请开启 --predict_cardinality")
        k = counts_to_k(count_pred, cfg.k_scale, cfg.k_min, cfg.k_max)
        pred = topk_mask(prob, k)
    else:
        pred = (prob >= thr[None, :]).astype(np.int32)
        if cfg.mode == "clamp":
            if count_pred is None:
                raise ValueError("clamp 模式需要集合大小预测，请开启 --predict_cardinality")
            k_allow = counts_to_k(count_pred, cfg.k_scale, cfg.k_min, cfg.k_max)
            rows = np.where(pred.sum(axis=1) > k_allow)[0]
            if rows.size > 0:
                pred[rows] = pred[rows] * topk_mask(prob[rows], k_allow[rows])

    if cfg.ensure_nonempty:
        idx = np.where(pred.sum(axis=1) == 0)[0]
        if idx.size > 0:
            pred[idx] = topk_mask(prob[idx], np.ones((idx.size,), dtype=np.int32))
    return pred


def decode_cartridge(
    car_prob: np.ndarray,
    car_thr: np.ndarray,
    car_count_pred: Optional[np.ndarray],
    cfg: DecodeConfig,
) -> np.ndarray:
    """(N, C) -> (N, C)"""
    return decode_2d(car_prob, car_thr, car_count_pred, cfg)


def decode_solvent(
    sol_prob: np.ndarray,
    sol_thr: np.ndarray,
    sol_count_pred: Optional[np.ndarray],
    cfg: DecodeConfig,
    step_pred: Optional[np.ndarray] = None,
    gate_by_step: bool = False,
) -> np.ndarray:
    """
    (N, 5, S) -> (N, 5, S)。把 (sample, step) 摊平成行后按 2D 解码。

    `gate_by_step=True` 时，预测为不存在的步骤会被清空溶剂 —— 这直接服务于 HC 指标
    （真实不存在的步骤不应预测任何溶剂）。step F1 已达 0.95，所以这个门控是可靠的。
    """
    n, n_steps, s = sol_prob.shape
    flat_prob = sol_prob.reshape(n * n_steps, s)
    thr = np.asarray(sol_thr, dtype=np.float32)
    flat_thr = thr.reshape(n_steps, s) if thr.ndim == 2 else np.full((n_steps, s), float(thr.reshape(-1)[0]), dtype=np.float32)

    flat_count = None
    if sol_count_pred is not None:
        flat_count = np.asarray(sol_count_pred, dtype=np.float64).reshape(n * n_steps)

    out = np.zeros((n * n_steps, s), dtype=np.int32)
    # 每个 step 有自己的阈值向量，所以按 step 分组解码
    for j in range(n_steps):
        rows = np.arange(j, n * n_steps, n_steps)
        out[rows] = decode_2d(
            flat_prob[rows],
            flat_thr[j],
            None if flat_count is None else flat_count[rows],
            cfg,
        )
    sol_pred = out.reshape(n, n_steps, s)

    if gate_by_step and step_pred is not None:
        sol_pred = sol_pred * step_pred.astype(np.int32)[:, :, None]
    return sol_pred


# ----------------------------------------------------------------------------
# per-label 阈值（在数组上做，不需要反复跑模型）
# ----------------------------------------------------------------------------

def _best_thr_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: np.ndarray,
    fallback: float,
    min_pos: int,
) -> float:
    """单个标签的最佳阈值，按该标签的二分类 F1 选。正例太少则直接用兜底阈值。"""
    pos = int((y_true == 1).sum())
    if pos < int(min_pos):
        return float(fallback)
    best_thr, best_f1 = float(fallback), -1.0
    for thr in grid:
        pred = y_prob >= thr
        tp = float(np.logical_and(pred, y_true == 1).sum())
        fp = float(np.logical_and(pred, y_true == 0).sum())
        fn = float(np.logical_and(~pred, y_true == 1).sum())
        f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def tune_per_label_thresholds(
    car_true: np.ndarray,
    car_prob: np.ndarray,
    step_true: np.ndarray,
    step_prob: np.ndarray,
    sol_true: np.ndarray,
    sol_prob: np.ndarray,
    has_info: np.ndarray,
    init: Dict[str, float],
    grid_min: float = 0.2,
    grid_max: float = 0.9,
    grid_num: int = 41,
    min_pos: int = 5,
) -> Dict[str, np.ndarray]:
    """
    逐标签阈值标定，等价于 `main.tune_thresholds_per_label_on_val`，
    但直接吃验证集数组，不用每次重新前向。

    这里仍然按**每个标签自己的 F1** 选阈值 —— 逐标签是一个独立的二分类操作点问题。
    集合大小的系统性偏差由 `DecodeConfig` 负责，两者分工不重叠。
    v15_e3 的消融证明了这一步不能省：关掉它 cartridge F1 会从 0.41 崩到 0.27。
    """
    grid = np.linspace(grid_min, grid_max, num=grid_num, dtype=np.float64)
    mask = has_info.astype(np.int32) == 1

    car_thr = np.full((car_true.shape[1],), float(init["cartridge"]), dtype=np.float32)
    for k in range(car_true.shape[1]):
        car_thr[k] = _best_thr_binary(car_true[:, k], car_prob[:, k], grid, init["cartridge"], min_pos)

    step_thr = np.full((step_true.shape[1],), float(init["step"]), dtype=np.float32)
    sol_thr = np.full(sol_true.shape[1:], float(init["solvent"]), dtype=np.float32)
    if mask.any():
        st_true, st_prob = step_true[mask], step_prob[mask]
        for k in range(step_true.shape[1]):
            step_thr[k] = _best_thr_binary(st_true[:, k], st_prob[:, k], grid, init["step"], min_pos)

        so_true, so_prob = sol_true[mask], sol_prob[mask]
        for j in range(so_true.shape[1]):
            for s in range(so_true.shape[2]):
                sol_thr[j, s] = _best_thr_binary(
                    so_true[:, j, s], so_prob[:, j, s], grid, init["solvent"], min_pos
                )

    return {"cartridge": car_thr, "step": step_thr, "solvent": sol_thr}


# ----------------------------------------------------------------------------
# 调优目标
# ----------------------------------------------------------------------------

def _per_sample_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    inter = np.logical_and(y_true == 1, y_pred == 1).sum(axis=1).astype(np.float64)
    uni = np.logical_or(y_true == 1, y_pred == 1).sum(axis=1).astype(np.float64)
    out = np.ones((y_true.shape[0],), dtype=np.float64)
    m = uni > 0
    out[m] = inter[m] / uni[m]
    return out


def objective_2d(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> float:
    """
    在一个 (M, L) 的多标签任务上计算调优目标。

    - `f1`          : micro-F1，等价于历史行为，用于消融对照
    - `jaccard`     : 逐样本 Jaccard 均值，SSS 的直接组成部分
    - `exact_match` : 集合完全一致的比例，StrictAcc 的直接组成部分
    - `em_jaccard`  : `0.5*exact_match + 0.5*jaccard`，兼顾严格与部分正确（推荐）
    """
    if kind == "f1":
        return float(multilabel_stats(y_true, y_pred)["micro_f1"])
    if kind == "jaccard":
        return float(_per_sample_jaccard(y_true, y_pred).mean())
    if kind == "exact_match":
        return float(np.all(y_true == y_pred, axis=1).mean())
    if kind == "em_jaccard":
        em = float(np.all(y_true == y_pred, axis=1).mean())
        jac = float(_per_sample_jaccard(y_true, y_pred).mean())
        return 0.5 * em + 0.5 * jac
    raise ValueError(f"未知调优目标 {kind}")


def _candidate_configs(
    modes: Sequence[str],
    k_scales: Sequence[float],
    thr_shifts: Sequence[float],
    k_max: int,
    k_min: int,
    has_counts: bool,
    count_quantiles: Sequence[Optional[float]] = (None,),
) -> List[DecodeConfig]:
    """
    枚举候选解码配置。

    `threshold` 模式下 `k_scale` 不起作用，`cardinality` 模式下 `thr_shift` 不起作用，
    所以对应维度只保留一个取值，避免生成大量完全等价的重复配置。
    `threshold` 模式完全不用个数，所以分位数维度也只保留 None。
    """
    out: List[DecodeConfig] = []
    for mode in modes:
        if mode in {"cardinality", "clamp"} and not has_counts:
            continue
        scales = [1.0] if mode == "threshold" else list(k_scales)
        shifts = [0.0] if mode == "cardinality" else list(thr_shifts)
        quants: List[Optional[float]] = [None] if mode == "threshold" else list(count_quantiles)
        for ks in scales:
            for sh in shifts:
                for q in quants:
                    out.append(
                        DecodeConfig(
                            mode=mode, k_scale=float(ks), k_min=int(k_min),
                            k_max=int(k_max), thr_shift=float(sh),
                            count_quantile=None if q is None else float(q),
                        )
                    )
    return out


def resolve_count_pred(
    cfg: DecodeConfig,
    count_pred: Optional[np.ndarray],
    count_dist: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """
    按配置决定用哪个个数：指定了分位数且有分布就走分位数，否则用模型给的标量个数。

    抽成独立函数是为了让训练时的调优和测试时的复现走完全同一条路径 ——
    否则很容易出现"验证集上按分位数搜到了配置，测试集上却用了标量个数"这种静默错位。
    """
    if cfg.count_quantile is not None and count_dist is not None:
        return counts_from_dist(count_dist, cfg.count_quantile)
    return count_pred


def tune_cartridge_decode(
    car_true: np.ndarray,
    car_prob: np.ndarray,
    car_thr: np.ndarray,
    car_count_pred: Optional[np.ndarray],
    objective: str = "em_jaccard",
    modes: Sequence[str] = ("threshold", "clamp", "cardinality"),
    k_scales: Sequence[float] = (0.8, 1.0, 1.2, 1.5),
    thr_shifts: Sequence[float] = (-0.05, 0.0, 0.05, 0.10),
    car_count_dist: Optional[np.ndarray] = None,
    count_quantiles: Sequence[Optional[float]] = (None,),
) -> Tuple[DecodeConfig, float, List[Dict[str, object]]]:
    """
    在验证集上为 cartridge 选解码配置。返回 (最佳配置, 最佳目标值, 全部试验记录)。

    `k_min=1`：每个样本至少要推荐一种 cartridge，预测 0 个没有实际意义。
    """
    cands = _candidate_configs(
        modes, k_scales, thr_shifts, car_true.shape[1], 1, car_count_pred is not None,
        count_quantiles=count_quantiles,
    )
    best_cfg, best_val = DecodeConfig(), -1.0
    trials: List[Dict[str, object]] = []
    for cfg in cands:
        cnt = resolve_count_pred(cfg, car_count_pred, car_count_dist)
        pred = decode_cartridge(car_prob, car_thr, cnt, cfg)
        val = objective_2d(car_true, pred, objective)
        trials.append({**cfg.as_dict(), "objective": objective, "value": val,
                       "mean_pred_size": float(pred.sum(axis=1).mean())})
        if val > best_val:
            best_val, best_cfg = val, cfg
    return best_cfg, best_val, trials


def tune_solvent_decode(
    sol_true: np.ndarray,
    sol_prob: np.ndarray,
    sol_thr: np.ndarray,
    sol_count_pred: Optional[np.ndarray],
    has_info: np.ndarray,
    step_pred: Optional[np.ndarray] = None,
    objective: str = "em_jaccard",
    modes: Sequence[str] = ("threshold", "clamp", "cardinality"),
    k_scales: Sequence[float] = (0.8, 1.0, 1.2, 1.5),
    thr_shifts: Sequence[float] = (-0.05, 0.0, 0.05, 0.10),
    gate_choices: Sequence[bool] = (False, True),
    sol_count_dist: Optional[np.ndarray] = None,
    count_quantiles: Sequence[Optional[float]] = (None,),
) -> Tuple[DecodeConfig, bool, float, List[Dict[str, object]]]:
    """
    在验证集上为 solvent 选解码配置。只在 `has_info == 1` 的样本上评估。

    额外搜索 `gate_by_step`，返回 (最佳配置, 是否门控, 最佳目标值, 全部试验记录)。

    `k_min=0`：真实不存在的步骤应当一个溶剂都不出，强行给下界 1 会制造大量假阳性并打崩 HC。
    """
    mask = has_info.astype(np.int32) == 1
    n, n_steps, s = sol_true.shape
    cands = _candidate_configs(
        modes, k_scales, thr_shifts, s, 0, sol_count_pred is not None,
        count_quantiles=count_quantiles,
    )

    best_cfg, best_gate, best_val = DecodeConfig(), False, -1.0
    trials: List[Dict[str, object]] = []
    for cfg in cands:
        cnt = resolve_count_pred(cfg, sol_count_pred, sol_count_dist)
        for gate in gate_choices:
            if gate and step_pred is None:
                continue
            pred = decode_solvent(sol_prob, sol_thr, cnt, cfg,
                                  step_pred=step_pred, gate_by_step=bool(gate))
            val = objective_2d(
                sol_true[mask].reshape(int(mask.sum()), n_steps * s),
                pred[mask].reshape(int(mask.sum()), n_steps * s),
                objective,
            )
            trials.append({**cfg.as_dict(), "gate_by_step": bool(gate), "objective": objective,
                           "value": val, "mean_pred_size": float(pred[mask].sum(axis=(1, 2)).mean())})
            if val > best_val:
                best_val, best_cfg, best_gate = val, cfg, bool(gate)
    return best_cfg, best_gate, best_val, trials
