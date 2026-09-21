"""
v6 训练入口：四步改进整合在一个脚本里，每一步都能单独开关，方便做消融。

运行方式（从仓库根目录）：

    python -m spe.train --output_dir output_mtl_v6_s1_full --epochs 60

四个开关（默认全开）：

    --no_functional_groups     关掉步骤 3（官能团特征）
    --no_decompose_solvent     关掉步骤 4（溶剂拆解）
    --no_cardinality           关掉步骤 2（集合大小预测与基数解码）
    --decode_objective f1      步骤 2 退化成按 F1 调优（等价于历史行为）

与 `main_v4.py` / `main_v5.py` 的关系：本脚本自带训练循环，不再用 monkey-patch
（`main_v5.py` 是靠替换 `base.train_one_epoch` 生效的）。排序辅助损失作为普通函数
从 `spe/losses.py` 调用，行为与 `main_v5` 一致，所以 v21/v22 的结果仍可复现。
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from main import (
    _is_finite_scalar_tensor,
    classification_loss_elements,
    compute_ratio_loss,
)
from model import batch_graph_smoothness
from utils.dataloader import build_loaders
from utils.io import ensure_dir, save_json
from utils.metrics import compute_pos_weight, compute_pos_weight_3d
from utils.seed import set_seed

from .data import PreparedData, describe, prepare_data
from .decode import (
    DecodeConfig,
    tune_cartridge_decode,
    tune_per_label_thresholds,
    tune_solvent_decode,
)
from .evaluate import collect_predictions, evaluate, selection_score
from .losses import (
    asymmetric_loss,
    base_solvent_loss,
    cardinality_loss,
    count_ce_loss,
    rank_losses,
    ratio_bucket_loss,
)
from .metrics_strict import compare_legacy_vs_clean
from .model import build_model
from .solvent_vocab import combo_labels_to_base


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPE v6：干净指标 + 基数感知解码 + 官能团特征 + 溶剂拆解",
    )
    # 路径与切分
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cache_path", type=str, default="data/cache/dataset_cache.pkl")
    p.add_argument("--seed", type=int, default=42,
                   help="决定**数据划分**的种子。要和基线可比就必须固定成 42")
    p.add_argument("--init_seed", type=int, default=None,
                   help="决定**参数初始化与训练随机性**的种子，默认跟随 --seed。"
                        "把它和 --seed 解耦有两个用处：(1) 在同一个划分上做深度集成，"
                        "(2) 单独测量初始化带来的方差 —— 之前用 --seed 跑多 seed 时，"
                        "划分方差和初始化方差是混在一起的，测试集本身都不同，"
                        "严格来说不能直接和只在 seed 42 划分上评过的基线比。")
    p.add_argument("--train_ratio", type=float, default=0.8)
    # 只换初始化的深度集成增益只有 +0.003～0.013；随机森林/XGBoost 的强度恰恰来自
    # 数据级 bagging，所以这里补上同一机制，用来做方法学上对等的比较。
    p.add_argument("--bootstrap", action="store_true",
                   help="对训练集做有放回重采样（验证/测试集不动），配合不同 --init_seed "
                        "构成 bagging 集成的成员")
    p.add_argument("--bootstrap_frac", type=float, default=1.0,
                   help="重采样条数相对原训练集的比例")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--require_spe_info", action="store_true")
    p.add_argument("--unk_solvent_token", type=str, default="__UNK__")

    # 训练
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)

    # 结构（默认值取自 v22_s2 的 checkpoint args）
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--enc_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--smooth_topk", type=int, default=8)

    # 四步开关
    p.add_argument("--no_functional_groups", action="store_true", help="关掉步骤 3")
    p.add_argument("--fg_transform", type=str, default="log1p", choices=["binary", "count", "log1p"])
    p.add_argument("--fg_min_pos", type=int, default=20)
    p.add_argument("--no_decompose_solvent", action="store_true", help="关掉步骤 4")
    p.add_argument("--feature_transform", type=str, default="zscore",
                   choices=["zscore", "rank_gauss", "log1p", "robust"],
                   help="输入特征的单调变换。描述符极度偏斜（最极端一维偏度 53.8、"
                        "峰度 3269），而 z-score 不改变分布形状；树模型对单调变换免疫，MLP 不是。"
                        "维度不变，参数只在训练集上拟合")
    p.add_argument("--base_prior_weight", type=float, default=0.3,
                   help="步骤 4：基础溶剂先验与组合概率的融合权重，0 表示不融合")
    p.add_argument("--no_cardinality", action="store_true", help="关掉步骤 2")

    # --- 以下改动都**不改变输入特征**，只改模型内部表示、损失形式或解码 ---

    # 集合大小：回归 vs 分桶分类。上限诊断表明个数估计是最大瓶颈
    #（换成真实个数后 GUS 57.7 -> 67.9），而回归头在长尾分布下必然回归到均值。
    p.add_argument("--count_mode", type=str, default="regression",
                   choices=["regression", "classification"])
    p.add_argument("--n_car_count_bins", type=int, default=16)
    p.add_argument("--n_sol_count_bins", type=int, default=12)
    p.add_argument("--count_reduce", type=str, default="argmax", choices=["argmax", "expectation"])
    p.add_argument("--count_label_smooth", type=float, default=0.1,
                   help="个数分类时分给相邻桶的目标概率，编码「个数是有序量」这个先验")
    # cartridge 排序损失的形式。权重从 0.5 加到 2.0 时 top-1 单调上升、3.0 回落，
    # 说明该换形式而不是继续加权重。默认 set_ce 保持历史行为。
    p.add_argument("--rank_loss", type=str, default="set_ce",
                   choices=["set_ce", "any_pos", "ranknet", "margin"],
                   help="set_ce 把目标质量均匀摊到所有正标签（平均 3.27 个），"
                        "要求它们全部排高；而 top-1 只要求其中一个排第一。"
                        "any_pos 用 -log(正例总概率) 严格对齐 top-1；"
                        "ranknet 成对逻辑损失；margin 是 top-1 铰链损失")
    p.add_argument("--tune_count_quantile", action="store_true",
                   help="个数分类时，在验证集上搜「取分布的哪个分位数作为 k」。"
                        "众数（argmax）恒为 1、期望被长尾拉到 4，两头都塌缩；"
                        "分位数把中间整段区间打开。默认关闭以保持历史行为不变。")
    p.add_argument("--count_quantiles", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8",
                   help="逗号分隔的候选分位数")

    # 数值特征嵌入：输入还是那 21 个描述符，只是每个标量先展开成向量再进 MLP。
    # 目标是补上 MLP 相对树模型在轴对齐分段关系上的劣势（我们 cartridge top-1 0.51 vs RF 0.59）。
    p.add_argument("--numeric_embed", type=str, default="none", choices=["none", "periodic", "ple"])
    p.add_argument("--numeric_embed_bins", type=int, default=24)
    p.add_argument("--numeric_embed_dim", type=int, default=16)
    p.add_argument("--numeric_embed_sigma", type=float, default=0.05)

    # ratio：连续回归 vs 分桶分类。回归收敛到条件均值，MAE 好但 ≤0.10 命中率吃亏。
    p.add_argument("--ratio_mode", type=str, default="regression", choices=["regression", "bucket"])
    p.add_argument("--n_ratio_bins", type=int, default=20)
    p.add_argument("--ratio_label_smooth", type=float, default=0.1)

    # 多任务干扰缓解：不确定性加权自动学习四个主任务的相对权重
    p.add_argument("--uncertainty_weighting", action="store_true")

    # 损失权重（cartridge/step/solvent/ratio/smooth 沿用 v22_s2 的量级）
    p.add_argument("--lambda_cartridge", type=float, default=1.0)
    p.add_argument("--lambda_step", type=float, default=1.0)
    p.add_argument("--lambda_solvent", type=float, default=1.0)
    p.add_argument("--lambda_ratio", type=float, default=1.0)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.0)
    p.add_argument("--lambda_rank_cartridge", type=float, default=0.10)
    p.add_argument("--lambda_rank_solvent", type=float, default=0.05)
    p.add_argument("--lambda_count_cartridge", type=float, default=0.30)
    p.add_argument("--lambda_count_solvent", type=float, default=0.30)
    p.add_argument("--lambda_base_solvent", type=float, default=0.50)

    # 损失形式（默认取 v22_s2 的配置）
    p.add_argument("--loss_cartridge", type=str, default="focal", choices=["bce", "focal", "asl"])
    p.add_argument("--loss_step", type=str, default="bce", choices=["bce", "focal", "asl"])
    p.add_argument("--loss_solvent", type=str, default="focal", choices=["bce", "focal", "asl"])
    p.add_argument("--focal_gamma", type=float, default=1.5)
    # 非对称损失（ASL）超参，只在 loss_* = asl 时生效
    p.add_argument("--asl_gamma_pos", type=float, default=0.0)
    p.add_argument("--asl_gamma_neg", type=float, default=4.0)
    p.add_argument("--asl_clip", type=float, default=0.05)
    p.add_argument("--ratio_loss", type=str, default="mse_mae", choices=["mse", "huber", "mse_mae"])
    p.add_argument("--ratio_huber_delta", type=float, default=0.05)
    p.add_argument("--ratio_mix_mae_weight", type=float, default=0.10)
    p.add_argument("--solvent_pos_step_boost", type=float, default=2.2)
    p.add_argument("--solvent_neg_step_weight", type=float, default=1.0)

    # pos_weight 截断
    p.add_argument("--car_posw_max", type=float, default=50.0)
    p.add_argument("--step_posw_max", type=float, default=50.0)
    p.add_argument("--sol_posw_max", type=float, default=50.0)
    p.add_argument("--base_posw_max", type=float, default=50.0)

    # 阈值与解码
    p.add_argument("--thr_cartridge", type=float, default=0.5)
    p.add_argument("--thr_step", type=float, default=0.5)
    p.add_argument("--thr_solvent", type=float, default=0.5)
    p.add_argument("--solvent_step_beta", type=float, default=0.0,
                   help="solvent 概率按 step 置信度校准的指数，v22_s2 用的是 0.0")
    p.add_argument("--thr_grid_num", type=int, default=41)
    p.add_argument("--thr_min_pos_per_label", type=int, default=5)
    p.add_argument("--decode_objective", type=str, default="em_jaccard",
                   choices=["f1", "jaccard", "exact_match", "em_jaccard"],
                   help="步骤 2 的解码调优目标。f1 等价于历史行为")

    # 选模权重。链路项用干净口径的 chain_acc_no_ratio，理由见 spe/evaluate.py:selection_score
    p.add_argument("--sel_w_strict", type=float, default=3.0)
    p.add_argument("--sel_w_sss", type=float, default=1.5)
    p.add_argument("--sel_w_hc", type=float, default=1.2)
    p.add_argument("--sel_w_chain", type=float, default=4.0)
    p.add_argument("--sel_w_gus", type=float, default=1.0)
    p.add_argument("--sel_w_car_top1", type=float, default=0.0,
                   help="cartridge top-1 正确率在选模分里的权重。"
                        "做单任务诊断（其余 lambda 置 0）时把它设成 1、其余 sel_w 置 0")
    p.add_argument("--select_every", type=int, default=1,
                   help="每隔多少个 epoch 做一次完整选模评估（评估比训练慢，可放大以省时间）")
    return p


# ---------------------------------------------------------------------------
# pos_weight
# ---------------------------------------------------------------------------

def compute_pos_weights(
    prepared: PreparedData,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ds, tr = prepared.ds, prepared.train_idx
    car_w = np.clip(compute_pos_weight(ds.y_cartridge[tr].astype(np.float32)), 1.0, args.car_posw_max)
    step_w = np.clip(compute_pos_weight(ds.y_step[tr].astype(np.float32)), 1.0, args.step_posw_max)
    sol_w = np.clip(compute_pos_weight_3d(ds.y_solvent[tr].astype(np.float32)), 1.0, args.sol_posw_max)

    out = {
        "cartridge": torch.tensor(car_w, dtype=torch.float32, device=device),
        "step": torch.tensor(step_w, dtype=torch.float32, device=device),
        "solvent": torch.tensor(sol_w, dtype=torch.float32, device=device),
    }
    if prepared.decomp is not None:
        y_base = combo_labels_to_base(ds.y_solvent[tr].astype(np.float32), prepared.decomp)
        base_w = np.clip(compute_pos_weight_3d(y_base), 1.0, args.base_posw_max)
        out["base_solvent"] = torch.tensor(base_w, dtype=torch.float32, device=device)
    return out


# ---------------------------------------------------------------------------
# 单个 epoch 的损失计算（训练与验证共用）
# ---------------------------------------------------------------------------

LOSS_KEYS = (
    "loss", "cartridge", "step", "solvent", "ratio", "smooth", "gate",
    "rank_cartridge", "rank_solvent", "count_cartridge", "count_solvent", "base_solvent",
)


def _compute_losses(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
    pos_weights: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    loss_cfg: Dict[str, Any],
    smooth_topk: int,
    membership: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = batch["x"].to(device)
    y_car = batch["y_cartridge"].to(device)
    y_step = batch["y_step"].to(device)
    y_sol = batch["y_solvent"].to(device)
    has_info = batch["has_spe_info"].to(device).view(-1, 1)

    out = model(x)
    h = out["h"]
    car_logits = out["cartridge_logits"]
    step_logits = out["step_logits"]
    sol_logits = out["solvent_logits"]

    zero = torch.zeros((), device=device)

    def _cls_elements(logits: torch.Tensor, targets: torch.Tensor, task: str) -> torch.Tensor:
        """按任务的损失形式返回逐元素损失。`asl` 走 spe/losses，其余沿用 main 的实现。"""
        if loss_cfg[task] == "asl":
            # ASL 自带针对负例的衰减机制，再叠加 pos_weight 会重复放大正例，所以不传
            return asymmetric_loss(
                logits, targets,
                gamma_pos=loss_cfg["asl_gamma_pos"],
                gamma_neg=loss_cfg["asl_gamma_neg"],
                clip=loss_cfg["asl_clip"],
            )
        return classification_loss_elements(
            logits=logits, targets=targets, pos_weight=pos_weights[task],
            loss_type=loss_cfg[task], focal_gamma=loss_cfg["focal_gamma"],
        )

    loss_car = _cls_elements(car_logits, y_car, "cartridge").mean()

    step_elem = _cls_elements(step_logits, y_step, "step")
    loss_step = (step_elem * has_info).sum() / (has_info.sum() * y_step.size(1) + 1e-6)

    step_weight = (
        y_step * loss_cfg["solvent_pos_step_boost"]
        + (1.0 - y_step) * loss_cfg["solvent_neg_step_weight"]
    ).unsqueeze(-1)
    sol_weight = has_info.view(-1, 1, 1) * step_weight
    sol_elem = _cls_elements(sol_logits, y_sol, "solvent")
    loss_sol = (sol_elem * sol_weight).sum() / (sol_weight.sum() * y_sol.size(2) + 1e-6)

    # ratio：只在有数值标注的 pair 上
    sample_ids, step_ids, solvent_ids, y_ratio = [], [], [], []
    for i, pairs in enumerate(batch["ratio_pairs"]):
        if has_info[i].item() != 1.0 or pairs is None:
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
        yr_t = torch.tensor(np.asarray(y_ratio, dtype=np.float32), device=device)
        if getattr(model, "ratio_mode", "regression") == "bucket":
            loss_ratio = ratio_bucket_loss(
                ratio_logits=model.ratio_bucket_logits(h[idx_t], st_t, so_t),
                y_ratio=yr_t, n_bins=model.n_ratio_bins,
                label_smooth_adjacent=loss_cfg["ratio_label_smooth"],
            )
        else:
            loss_ratio = compute_ratio_loss(
                y_pred=model.ratio_pred(h[idx_t], st_t, so_t), y_true=yr_t,
                loss_type=loss_cfg["ratio_loss"], huber_delta=loss_cfg["ratio_huber_delta"],
                mix_mae_weight=loss_cfg["ratio_mix_mae_weight"],
            )
    else:
        loss_ratio = zero

    loss_smooth = batch_graph_smoothness(x=x, h=h, topk=smooth_topk)
    loss_gate = ((model.solvent_step_gate_alpha - 1.0) ** 2).mean()

    rank = rank_losses(
        car_logits=car_logits, y_car=y_car, step_true=y_step,
        sol_logits=sol_logits, y_sol=y_sol, has_info=has_info,
        form=loss_cfg["rank_loss"],
    )

    if "cartridge_count_logits" in out:
        cnt = count_ce_loss(
            car_count_logits=out["cartridge_count_logits"], y_car=y_car,
            sol_count_logits=out["solvent_count_logits"], y_sol=y_sol, has_info=has_info,
            label_smooth_adjacent=loss_cfg["count_label_smooth"],
        )
    elif "cartridge_count" in out:
        cnt = cardinality_loss(
            car_count_pred=out["cartridge_count"], y_car=y_car,
            sol_count_pred=out["solvent_count"], y_sol=y_sol, has_info=has_info,
        )
    else:
        cnt = {"count_cartridge": zero, "count_solvent": zero}

    if "base_solvent_logits" in out and membership is not None:
        b, n_steps, n_combo = y_sol.shape
        y_base = (y_sol.reshape(b * n_steps, n_combo) @ membership).clamp(max=1.0)
        y_base = y_base.reshape(b, n_steps, -1)
        loss_base = base_solvent_loss(
            base_logits=out["base_solvent_logits"], y_base=y_base,
            step_true=y_step, has_info=has_info,
            pos_weight=pos_weights.get("base_solvent"),
            pos_step_boost=loss_cfg["solvent_pos_step_boost"],
            neg_step_weight=loss_cfg["solvent_neg_step_weight"],
        )
    else:
        loss_base = zero

    if getattr(model, "uncertainty_weighting", False):
        # Kendall 等人的同方差不确定性加权：L = sum_i [ exp(-s_i) * L_i + s_i ]。
        # s_i 是可学习的 log 方差，噪声大的任务会被自动压低权重，
        # 而 +s_i 这一项防止模型靠把所有 s_i 推到 +inf 来作弊。
        # 只作用于四个主任务；辅助损失（rank/count/base/smooth）仍用固定 lambda，
        # 否则它们会和主任务抢同一套权重，失去"辅助"的定位。
        s = model.task_log_var
        main = (
            torch.exp(-s[0]) * loss_car + s[0]
            + torch.exp(-s[1]) * loss_step + s[1]
            + torch.exp(-s[2]) * loss_sol + s[2]
            + torch.exp(-s[3]) * loss_ratio + s[3]
        )
    else:
        main = (
            lambdas["cartridge"] * loss_car
            + lambdas["step"] * loss_step
            + lambdas["solvent"] * loss_sol
            + lambdas["ratio"] * loss_ratio
        )

    total = (
        main
        + lambdas["smooth"] * loss_smooth
        + lambdas["gate"] * loss_gate
        + lambdas["rank_cartridge"] * rank["rank_cartridge"]
        + lambdas["rank_solvent"] * rank["rank_solvent"]
        + lambdas["count_cartridge"] * cnt["count_cartridge"]
        + lambdas["count_solvent"] * cnt["count_solvent"]
        + lambdas["base_solvent"] * loss_base
    )

    parts = {
        "loss": total, "cartridge": loss_car, "step": loss_step, "solvent": loss_sol,
        "ratio": loss_ratio, "smooth": loss_smooth, "gate": loss_gate,
        "rank_cartridge": rank["rank_cartridge"], "rank_solvent": rank["rank_solvent"],
        "count_cartridge": cnt["count_cartridge"], "count_solvent": cnt["count_solvent"],
        "base_solvent": loss_base,
    }
    if not all(_is_finite_scalar_tensor(v) for v in parts.values()):
        return total, {}
    return total, {k: float(v.item()) for k, v in parts.items()}


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    pos_weights: Dict[str, torch.Tensor],
    lambdas: Dict[str, float],
    loss_cfg: Dict[str, Any],
    smooth_topk: int,
    membership: Optional[torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """`optimizer is None` 时是验证，否则是训练。非有限损失的 batch 会被跳过。"""
    training = optimizer is not None
    model.train(training)
    totals = {k: 0.0 for k in LOSS_KEYS}
    n_ok = 0
    n_skipped = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            total, parts = _compute_losses(
                model, batch, device, pos_weights, lambdas, loss_cfg, smooth_topk, membership
            )
            if not parts:
                n_skipped += 1
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grads_ok = all(
                    torch.isfinite(p.grad).all().item()
                    for p in model.parameters() if p.grad is not None
                )
                if not grads_ok:
                    n_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
            for k, v in parts.items():
                totals[k] += v
            n_ok += 1

    if n_ok == 0:
        return {k: float("nan") for k in LOSS_KEYS}
    out = {k: v / n_ok for k, v in totals.items()}
    if n_skipped:
        out["skipped_batches"] = float(n_skipped)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dir(args.output_dir)
    if os.path.dirname(args.cache_path):
        ensure_dir(os.path.dirname(args.cache_path))
    set_seed(args.seed)
    device = torch.device(args.device)

    use_cardinality = not args.no_cardinality
    # prepare_data 里的 split_indices 用的是独立的 default_rng(seed)，不吃全局随机状态，
    # 所以下面重设种子不会改变数据划分。
    prepared = prepare_data(
        data_dir=args.data_dir,
        cache_path=args.cache_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        require_spe_info=args.require_spe_info,
        unk_solvent_token=args.unk_solvent_token,
        use_functional_groups=not args.no_functional_groups,
        fg_transform=args.fg_transform,
        fg_min_pos=args.fg_min_pos,
        decompose_solvent=not args.no_decompose_solvent,
        feature_transform=args.feature_transform,
    )
    print(describe(prepared), flush=True)

    init_seed = args.seed if args.init_seed is None else int(args.init_seed)
    if init_seed != args.seed:
        set_seed(init_seed)
        print(f"数据划分种子 {args.seed}，初始化/训练种子 {init_seed}（划分不受影响）", flush=True)

    # bagging：只对训练集做有放回重采样，验证集与测试集保持原样，
    # 所以各成员之间以及与基线之间的评测集完全一致。
    # 重采样由 init_seed 决定，因此深度集成的「同划分」前提不被破坏。
    train_idx = prepared.train_idx
    if args.bootstrap:
        rng = np.random.default_rng(init_seed)
        n_draw = max(1, int(round(args.bootstrap_frac * len(train_idx))))
        train_idx = rng.choice(train_idx, size=n_draw, replace=True)
        n_unique = len(np.unique(train_idx))
        print(
            f"bootstrap 重采样：抽 {n_draw} 条（原 {len(prepared.train_idx)} 条），"
            f"覆盖 {n_unique} 个不同样本（{n_unique / len(prepared.train_idx):.1%}）",
            flush=True,
        )

    train_loader, val_loader, test_loader = build_loaders(
        prepared.ds, train_idx, prepared.val_idx, prepared.test_idx,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = build_model(
        input_dim=prepared.input_dim,
        n_cartridge=len(prepared.artifacts.cartridge_cols),
        n_steps=prepared.n_steps,
        n_solvent=len(prepared.artifacts.solvent_vocab),
        hidden_dim=args.hidden_dim,
        enc_layers=args.enc_layers,
        dropout=args.dropout,
        n_base_solvent=prepared.n_base_solvent,
        predict_cardinality=use_cardinality,
        count_mode=args.count_mode,
        n_car_count_bins=args.n_car_count_bins,
        n_sol_count_bins=args.n_sol_count_bins,
        count_reduce=args.count_reduce,
        numeric_embed=args.numeric_embed,
        numeric_embed_bins=args.numeric_embed_bins,
        numeric_embed_dim=args.numeric_embed_dim,
        numeric_embed_sigma=args.numeric_embed_sigma,
        # ple 的分桶边界只从训练集取，且取的是标准化之后的取值（模型看到的就是这个尺度）
        x_train=prepared.ds.xs[prepared.train_idx],
        ratio_mode=args.ratio_mode,
        n_ratio_bins=args.n_ratio_bins,
        uncertainty_weighting=args.uncertainty_weighting,
    ).to(device)

    pos_weights = compute_pos_weights(prepared, args, device)
    membership = (
        torch.tensor(prepared.decomp.membership, dtype=torch.float32, device=device)
        if prepared.decomp is not None
        else None
    )

    lambdas = {
        "cartridge": args.lambda_cartridge, "step": args.lambda_step,
        "solvent": args.lambda_solvent, "ratio": args.lambda_ratio,
        "smooth": args.lambda_smooth, "gate": args.lambda_gate,
        "rank_cartridge": args.lambda_rank_cartridge, "rank_solvent": args.lambda_rank_solvent,
        "count_cartridge": args.lambda_count_cartridge if use_cardinality else 0.0,
        "count_solvent": args.lambda_count_solvent if use_cardinality else 0.0,
        "base_solvent": args.lambda_base_solvent if prepared.decomp is not None else 0.0,
    }
    loss_cfg = {
        "cartridge": args.loss_cartridge, "step": args.loss_step, "solvent": args.loss_solvent,
        "focal_gamma": args.focal_gamma, "ratio_loss": args.ratio_loss,
        "ratio_huber_delta": args.ratio_huber_delta,
        "ratio_mix_mae_weight": args.ratio_mix_mae_weight,
        "solvent_pos_step_boost": args.solvent_pos_step_boost,
        "solvent_neg_step_weight": args.solvent_neg_step_weight,
        "asl_gamma_pos": args.asl_gamma_pos,
        "asl_gamma_neg": args.asl_gamma_neg,
        "asl_clip": args.asl_clip,
        "count_label_smooth": args.count_label_smooth,
        "ratio_label_smooth": args.ratio_label_smooth,
        "rank_loss": args.rank_loss,
    }
    sel_weights = {
        "strict": args.sel_w_strict, "sss": args.sel_w_sss, "hc": args.sel_w_hc,
        "chain": args.sel_w_chain, "gus": args.sel_w_gus,
        "car_top1": args.sel_w_car_top1,
    }
    init_thr = {
        "cartridge": args.thr_cartridge, "step": args.thr_step, "solvent": args.thr_solvent,
    }

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[Dict[str, Any]] = []
    best_score = -float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, pos_weights, lambdas, loss_cfg,
                       args.smooth_topk, membership, optimizer=optimizer, grad_clip=args.grad_clip)
        va = run_epoch(model, val_loader, device, pos_weights, lambdas, loss_cfg,
                       args.smooth_topk, membership, optimizer=None)

        row: Dict[str, Any] = {"epoch": epoch, "train": tr, "val": va}

        if epoch % max(1, args.select_every) == 0 or epoch == args.epochs:
            val_bundle = collect_predictions(
                model, val_loader, device,
                solvent_step_beta=args.solvent_step_beta,
                decomp=prepared.decomp,
                base_prior_weight=args.base_prior_weight,
            )
            val_metrics = evaluate(val_bundle, init_thr)
            score = selection_score(val_metrics, sel_weights)
            row["val_select_score"] = score
            row["val_chain_acc"] = val_metrics["overall_clean"]["chain_acc_no_ratio"]
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(row)
        print(
            f"[epoch {epoch:>3}/{args.epochs}] train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f}"
            + (f" val_score={row['val_select_score']:.4f}" if "val_select_score" in row else "")
            + f" ({time.time() - t0:.0f}s)",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("没有任何 epoch 完成选模评估，请检查 --select_every 与 --epochs")
    model.load_state_dict(best_state)
    print(f"最佳 checkpoint 来自 epoch {best_epoch}，验证集选模分 {best_score:.4f}", flush=True)

    # --- 标定与解码调优：全部在验证集上做 ---
    val_bundle = collect_predictions(
        model, val_loader, device, solvent_step_beta=args.solvent_step_beta,
        decomp=prepared.decomp, base_prior_weight=args.base_prior_weight,
    )
    per_label = tune_per_label_thresholds(
        car_true=val_bundle.car_true, car_prob=val_bundle.car_prob,
        step_true=val_bundle.step_true, step_prob=val_bundle.step_prob,
        sol_true=val_bundle.sol_true, sol_prob=val_bundle.sol_prob,
        has_info=val_bundle.has_info, init=init_thr,
        grid_num=args.thr_grid_num, min_pos=args.thr_min_pos_per_label,
    )
    print("per-label 阈值标定完成", flush=True)

    quantiles: List[Optional[float]] = [None]
    if args.tune_count_quantile:
        quantiles += [float(q) for q in args.count_quantiles.split(",") if q.strip()]

    car_cfg, car_obj, car_trials = tune_cartridge_decode(
        car_true=val_bundle.car_true, car_prob=val_bundle.car_prob,
        car_thr=per_label["cartridge"], car_count_pred=val_bundle.car_count_pred,
        objective=args.decode_objective,
        car_count_dist=val_bundle.car_count_dist, count_quantiles=quantiles,
    )
    val_step_pred = (val_bundle.step_prob >= per_label["step"][None, :]).astype(np.int32)
    sol_cfg, sol_gate, sol_obj, sol_trials = tune_solvent_decode(
        sol_true=val_bundle.sol_true, sol_prob=val_bundle.sol_prob,
        sol_thr=per_label["solvent"], sol_count_pred=val_bundle.sol_count_pred,
        has_info=val_bundle.has_info, step_pred=val_step_pred,
        objective=args.decode_objective,
        sol_count_dist=val_bundle.sol_count_dist, count_quantiles=quantiles,
    )
    print(f"解码调优（目标={args.decode_objective}）:", flush=True)
    print(f"  cartridge -> {car_cfg.as_dict()}  目标值={car_obj:.4f}", flush=True)
    print(f"  solvent   -> {sol_cfg.as_dict()} gate={sol_gate}  目标值={sol_obj:.4f}", flush=True)

    # 用调优后的阈值与解码在**验证集**上再评一次。
    # 这是跨配置选模的唯一合法依据 —— 训练中的 val_select_score 用的是未调优的初始阈值，
    # 不能反映各配置在自己最优解码下的水平；而测试集绝不能参与任何配置选择。
    val_metrics_tuned = evaluate(
        val_bundle, init_thr, per_label_thresholds=per_label,
        cartridge_decode=car_cfg, solvent_decode=sol_cfg, solvent_gate_by_step=sol_gate,
    )
    val_score_tuned = selection_score(val_metrics_tuned, sel_weights)
    print(f"验证集（调优后）选模分 {val_score_tuned:.4f}，"
          f"链路正确率 {val_metrics_tuned['overall_clean']['chain_acc_no_ratio']:.4f}", flush=True)

    # --- 测试集评估 ---
    test_bundle = collect_predictions(
        model, test_loader, device, solvent_step_beta=args.solvent_step_beta,
        decomp=prepared.decomp, base_prior_weight=args.base_prior_weight,
    )
    test_metrics = evaluate(
        test_bundle, init_thr, per_label_thresholds=per_label,
        cartridge_decode=car_cfg, solvent_decode=sol_cfg, solvent_gate_by_step=sol_gate,
    )
    # 同时给出"纯阈值解码"的对照，隔离出基数解码的净贡献
    test_metrics_threshold_only = evaluate(
        test_bundle, init_thr, per_label_thresholds=per_label,
        cartridge_decode=DecodeConfig(mode="threshold"),
        solvent_decode=DecodeConfig(mode="threshold"),
        solvent_gate_by_step=False,
    )

    print("\n=== 测试集（基数感知解码）===", flush=True)
    print(compare_legacy_vs_clean({
        "legacy": test_metrics["overall"],
        "clean": test_metrics["overall_clean"],
        "diagnostics": test_metrics["hit1_diagnostics"],
    }), flush=True)
    print("\n预测集合大小：", test_metrics["pred_set_size"], flush=True)

    # --- 落盘 ---
    config = {
        "steps_enabled": {
            "step1_clean_metrics": True,
            "step2_cardinality_decode": use_cardinality,
            "step3_functional_groups": not args.no_functional_groups,
            "step4_solvent_decomposition": not args.no_decompose_solvent,
        },
        "args": vars(args),
        "lambdas": lambdas,
        "loss_config": loss_cfg,
        "selection_weights": sel_weights,
        "input_dim": prepared.input_dim,
        "n_base_solvent": prepared.n_base_solvent,
        "splits": {
            "n_total": len(prepared.ds),
            "n_train": int(len(prepared.train_idx)),
            "n_val": int(len(prepared.val_idx)),
            "n_test": int(len(prepared.test_idx)),
        },
    }
    save_json(os.path.join(args.output_dir, "config.json"), config)
    save_json(os.path.join(args.output_dir, "train_history.json"), history)
    save_json(
        os.path.join(args.output_dir, "decode_config.json"),
        {
            "objective": args.decode_objective,
            "cartridge": {**car_cfg.as_dict(), "val_objective": car_obj},
            "solvent": {**sol_cfg.as_dict(), "gate_by_step": sol_gate, "val_objective": sol_obj},
            "cartridge_trials": car_trials,
            "solvent_trials": sol_trials,
        },
    )
    save_json(
        os.path.join(args.output_dir, "tuned_thresholds_per_label.json"),
        {k: np.asarray(v).tolist() for k, v in per_label.items()},
    )
    save_json(
        os.path.join(args.output_dir, "test_metrics.json"),
        {
            "metrics": test_metrics,
            "metrics_threshold_only": test_metrics_threshold_only,
            "val_metrics_tuned": val_metrics_tuned,
            "val_select_score_tuned": val_score_tuned,
            "best_epoch": best_epoch,
            "best_val_select_score": best_score,
            "steps_enabled": config["steps_enabled"],
            "splits": config["splits"],
            "note": (
                "v6：metrics.overall 为旧口径（可与历史 test_metrics.json 直接对比）；"
                "metrics.overall_clean 为修正口径，建议以 hit_at_1_ratio_subset 为主指标。"
                "metrics_threshold_only 是关掉基数解码的对照。"
            ),
        },
    )
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "mean": prepared.mean,
            "std": prepared.std,
            "feature_names": prepared.feature_names,
            "cartridge_cols": prepared.artifacts.cartridge_cols,
            "solvent_vocab": prepared.artifacts.solvent_vocab,
            "base_solvent_vocab": prepared.decomp.base_vocab if prepared.decomp else None,
            "per_label_thresholds": {k: np.asarray(v).tolist() for k, v in per_label.items()},
            "decode": {"cartridge": car_cfg.as_dict(),
                       "solvent": {**sol_cfg.as_dict(), "gate_by_step": sol_gate}},
        },
        os.path.join(args.output_dir, "best_model.pt"),
    )
    print(f"\n全部结果已写入 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
