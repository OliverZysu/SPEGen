"""
统一的推理与评估。

`collect_predictions()` 把模型在一个 loader 上的全部原始输出收集成 numpy 数组
（一次前向，之后所有解码配置的搜索都在数组上做，不再重复跑模型）——
这让 `spe/decode.py` 的网格搜索成本几乎可以忽略。

`evaluate()` 在给定解码配置下算出完整指标，**同时输出旧口径和干净口径**：
`metrics["overall"]` 与 `main.run_eval` 完全同义，可直接和历史 `test_metrics.json` 对比；
`metrics["overall_clean"]` 是 `spe/metrics_strict.py` 给出的修正版本。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.metrics import multilabel_stats, regression_stats

from .decode import DecodeConfig, decode_cartridge, decode_solvent, resolve_count_pred
from .metrics_strict import RatioRecord, compute_overall_metrics_v2
from .solvent_vocab import (
    SolventDecomposition,
    base_prob_to_combo_prior,
    blend_combo_with_prior,
)


@dataclass
class PredictionBundle:
    """模型在某个 split 上的全部原始输出。"""

    car_true: np.ndarray                       # (N, C)
    car_prob: np.ndarray                       # (N, C)
    step_true: np.ndarray                      # (N, 5)
    step_prob: np.ndarray                      # (N, 5)
    sol_true: np.ndarray                       # (N, 5, S)
    sol_prob: np.ndarray                       # (N, 5, S) 已含 beta 校准与基础先验融合
    has_info: np.ndarray                       # (N,)
    ratio_records: List[RatioRecord]
    ratio_true: np.ndarray
    ratio_pred: np.ndarray
    cids: List[str]
    method_lists: List[List[str]]
    car_count_pred: Optional[np.ndarray] = None    # (N,)
    sol_count_pred: Optional[np.ndarray] = None    # (N, 5)
    base_prob: Optional[np.ndarray] = None         # (N, 5, S_base)
    # 个数分类模式下的完整分布，(N, K) / (N, 5, K)。
    # 留着整个分布而不只是一个标量，是为了让"取哪个分位数"变成解码时可搜的超参，
    # 不必为每个候选分位数重训一遍。
    car_count_dist: Optional[np.ndarray] = None
    sol_count_dist: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return int(self.car_true.shape[0])

    @property
    def has_counts(self) -> bool:
        return self.car_count_pred is not None and self.sol_count_pred is not None


def apply_step_calibration(sol_prob: np.ndarray, step_prob: np.ndarray, beta: float) -> np.ndarray:
    """与 `main.apply_solvent_step_calibration` 同义：p(sol) * p(step)^beta。"""
    b = float(beta)
    if b <= 0.0:
        return sol_prob
    factor = np.power(np.clip(step_prob, 1e-6, 1.0), b).astype(np.float32)
    return (sol_prob * factor[:, :, None]).astype(np.float32)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    solvent_step_beta: float = 0.0,
    decomp: Optional[SolventDecomposition] = None,
    base_prior_weight: float = 0.0,
) -> PredictionBundle:
    """
    跑一遍前向，收集所有需要的数组。

    `base_prior_weight > 0` 且模型带基础溶剂头时，会把组合级概率与基础级先验做几何融合
    （步骤 4）。融合发生在这里，之后的解码和调优对此透明。
    """
    model.eval()

    car_true_l, car_prob_l = [], []
    step_true_l, step_prob_l = [], []
    sol_true_l, sol_prob_l = [], []
    base_prob_l: List[np.ndarray] = []
    car_cnt_l, sol_cnt_l = [], []
    car_dist_l: List[np.ndarray] = []
    sol_dist_l: List[np.ndarray] = []
    has_info_l = []
    cids: List[str] = []
    method_lists: List[List[str]] = []
    ratio_records: List[RatioRecord] = []
    ratio_true_l: List[float] = []
    ratio_pred_l: List[float] = []

    for batch in loader:
        sample_offset = len(cids)
        x = batch["x"].to(device)
        has_info = batch["has_spe_info"].cpu().numpy().astype(np.int32)

        out = model(x)
        car_prob_l.append(torch.sigmoid(out["cartridge_logits"]).cpu().numpy())
        step_prob_l.append(torch.sigmoid(out["step_logits"]).cpu().numpy())
        sol_prob_l.append(torch.sigmoid(out["solvent_logits"]).cpu().numpy())
        if "base_solvent_logits" in out:
            base_prob_l.append(torch.sigmoid(out["base_solvent_logits"]).cpu().numpy())
        if "cartridge_count" in out:
            car_cnt_l.append(out["cartridge_count"].cpu().numpy())
            sol_cnt_l.append(out["solvent_count"].cpu().numpy())
        if "cartridge_count_logits" in out:
            car_dist_l.append(torch.softmax(out["cartridge_count_logits"], dim=-1).cpu().numpy())
            sol_dist_l.append(torch.softmax(out["solvent_count_logits"], dim=-1).cpu().numpy())

        car_true_l.append(batch["y_cartridge"].cpu().numpy())
        step_true_l.append(batch["y_step"].cpu().numpy())
        sol_true_l.append(batch["y_solvent"].cpu().numpy())
        has_info_l.append(has_info)
        cids.extend(batch["cids"])
        method_lists.extend(batch["method_ids"])

        # ratio 只在有数值标注的 (step, solvent) pair 上评估
        h = out["h"].detach()
        sample_ids, step_ids, solvent_ids, y_ratio = [], [], [], []
        for i, pairs in enumerate(batch["ratio_pairs"]):
            if has_info[i] != 1 or pairs is None:
                continue
            for sj, sid, r in pairs:
                rv = float(r)
                if not np.isfinite(rv):
                    continue
                sample_ids.append(i)
                step_ids.append(int(sj))
                solvent_ids.append(int(sid))
                y_ratio.append(rv)
        if sample_ids:
            idx_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
            st_t = torch.tensor(step_ids, device=device, dtype=torch.long)
            so_t = torch.tensor(solvent_ids, device=device, dtype=torch.long)
            pred = model.ratio_pred(h[idx_t], st_t, so_t).cpu().numpy().tolist()
            for k, yp in enumerate(pred):
                ratio_records.append(
                    (sample_offset + sample_ids[k], step_ids[k], solvent_ids[k], y_ratio[k], float(yp))
                )
                ratio_true_l.append(y_ratio[k])
                ratio_pred_l.append(float(yp))

    step_prob = np.concatenate(step_prob_l, axis=0)
    sol_prob = apply_step_calibration(
        np.concatenate(sol_prob_l, axis=0), step_prob, solvent_step_beta
    )

    base_prob = np.concatenate(base_prob_l, axis=0) if base_prob_l else None
    if base_prob is not None and decomp is not None and base_prior_weight > 0.0:
        prior = base_prob_to_combo_prior(base_prob, decomp)
        sol_prob = blend_combo_with_prior(sol_prob, prior, weight=base_prior_weight)

    return PredictionBundle(
        car_true=np.concatenate(car_true_l, axis=0),
        car_prob=np.concatenate(car_prob_l, axis=0),
        step_true=np.concatenate(step_true_l, axis=0),
        step_prob=step_prob,
        sol_true=np.concatenate(sol_true_l, axis=0),
        sol_prob=sol_prob,
        has_info=np.concatenate(has_info_l, axis=0),
        ratio_records=ratio_records,
        ratio_true=np.asarray(ratio_true_l, dtype=np.float32),
        ratio_pred=np.asarray(ratio_pred_l, dtype=np.float32),
        cids=cids,
        method_lists=method_lists,
        car_count_pred=np.concatenate(car_cnt_l, axis=0) if car_cnt_l else None,
        sol_count_pred=np.concatenate(sol_cnt_l, axis=0) if sol_cnt_l else None,
        base_prob=base_prob,
        car_count_dist=np.concatenate(car_dist_l, axis=0) if car_dist_l else None,
        sol_count_dist=np.concatenate(sol_dist_l, axis=0) if sol_dist_l else None,
    )


def resolve_thresholds(
    bundle: PredictionBundle,
    thresholds: Dict[str, float],
    per_label: Optional[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把全局阈值或 per-label 阈值统一成 (C,) / (5,) / (5,S) 三个数组。"""
    if per_label is not None:
        return (
            np.asarray(per_label["cartridge"], dtype=np.float32),
            np.asarray(per_label["step"], dtype=np.float32),
            np.asarray(per_label["solvent"], dtype=np.float32),
        )
    return (
        np.full((bundle.car_prob.shape[1],), float(thresholds["cartridge"]), dtype=np.float32),
        np.full((bundle.step_prob.shape[1],), float(thresholds["step"]), dtype=np.float32),
        np.full(bundle.sol_prob.shape[1:], float(thresholds["solvent"]), dtype=np.float32),
    )


def evaluate(
    bundle: PredictionBundle,
    thresholds: Dict[str, float],
    per_label_thresholds: Optional[Dict[str, np.ndarray]] = None,
    cartridge_decode: Optional[DecodeConfig] = None,
    solvent_decode: Optional[DecodeConfig] = None,
    solvent_gate_by_step: bool = False,
    ratio_tau: float = 0.10,
) -> Dict[str, Any]:
    """
    在给定解码配置下算全套指标。

    `cartridge_decode` / `solvent_decode` 为 None 时退化成纯阈值解码，
    此时本函数的输出与 `main.run_eval` 的 `metrics` 在数值上应当一致
    （唯一差别是 solvent 的 top-k 剪枝改由 DecodeConfig 表达）。
    """
    car_cfg = cartridge_decode or DecodeConfig(mode="threshold")
    sol_cfg = solvent_decode or DecodeConfig(mode="threshold")
    car_thr, step_thr, sol_thr = resolve_thresholds(bundle, thresholds, per_label_thresholds)

    step_pred = (bundle.step_prob >= step_thr[None, :]).astype(np.int32)
    # 走 resolve_count_pred，保证这里用的个数与验证集调优时选中的那套完全一致
    car_pred = decode_cartridge(
        bundle.car_prob, car_thr,
        resolve_count_pred(car_cfg, bundle.car_count_pred, bundle.car_count_dist),
        car_cfg,
    )
    sol_pred = decode_solvent(
        bundle.sol_prob,
        sol_thr,
        resolve_count_pred(sol_cfg, bundle.sol_count_pred, bundle.sol_count_dist),
        sol_cfg,
        step_pred=step_pred,
        gate_by_step=solvent_gate_by_step,
    )

    mask = bundle.has_info.astype(np.int32) == 1
    n, n_steps, s = bundle.sol_true.shape

    metrics: Dict[str, Any] = {"cartridge": multilabel_stats(bundle.car_true, car_pred)}
    if mask.any():
        metrics["step"] = multilabel_stats(bundle.step_true[mask], step_pred[mask])
        metrics["solvent"] = multilabel_stats(
            bundle.sol_true.reshape(n, n_steps * s)[mask],
            sol_pred.reshape(n, n_steps * s)[mask],
        )
    else:
        metrics["step"], metrics["solvent"] = {}, {}
    metrics["ratio"] = regression_stats(bundle.ratio_true, bundle.ratio_pred)

    v2 = compute_overall_metrics_v2(
        car_true=bundle.car_true,
        car_pred=car_pred,
        car_prob=bundle.car_prob,
        step_true=bundle.step_true,
        step_pred=step_pred,
        sol_true=bundle.sol_true,
        sol_pred=sol_pred,
        step_prob=bundle.step_prob,
        sol_prob_adj=bundle.sol_prob,
        has_info_all=bundle.has_info,
        ratio_pair_records=bundle.ratio_records,
        ratio_tau=ratio_tau,
    )
    metrics["overall"] = v2["legacy"]
    metrics["overall_clean"] = v2["clean"]
    metrics["hit1_diagnostics"] = v2["diagnostics"]
    metrics["pred_set_size"] = {
        "cartridge_pred_mean": float(car_pred.sum(axis=1).mean()),
        "cartridge_true_mean": float(bundle.car_true.sum(axis=1).mean()),
        "solvent_pred_mean": float(sol_pred[mask].sum(axis=(1, 2)).mean()) if mask.any() else 0.0,
        "solvent_true_mean": float(bundle.sol_true[mask].sum(axis=(1, 2)).mean()) if mask.any() else 0.0,
    }
    return metrics


def selection_score(metrics: Dict[str, Any], weights: Dict[str, float]) -> float:
    """
    选模用的加权分。默认权重见 `spe/train.py`。

    与 `main_v4._overall_select_score` 的关键区别：链路项用 `chain_acc_no_ratio`
    而不是旧的 `method_hit_at_1`。

    旧口径不能用来选模 —— 它把"top-1 落在没有 ratio 标注的 pair 上"算作命中，
    于是"把 top-1 往标注稀疏处偏移"这种退化行为会被选模逻辑主动奖励。
    也不能用 `hit_at_1_ratio_subset`，因为它的分母随 checkpoint 变化，不同 epoch 之间不可比。
    `chain_acc_no_ratio` 的分母恒为 has_info=1 的样本数，跨 epoch、跨模型都稳定。
    """
    clean = metrics.get("overall_clean", {})
    legacy = metrics.get("overall", {})
    chain = clean.get("chain_acc_no_ratio")
    if chain is None or not np.isfinite(chain):
        chain = 0.0
    return (
        weights["strict"] * float(clean.get("strict_acc_info_only", 0.0) or 0.0)
        + weights["sss"] * float(legacy.get("scheme_similarity_score", 0.0))
        + weights["hc"] * float(legacy.get("hierarchical_consistency_acc", 0.0))
        + weights["chain"] * float(chain)
        + weights["gus"] * float(legacy.get("global_utility_score", 0.0)) / 100.0
        + weights.get("car_top1", 0.0) * float(clean.get("cartridge_top1_acc", 0.0) or 0.0)
    )
