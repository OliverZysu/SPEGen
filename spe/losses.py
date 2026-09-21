"""
各辅助损失项。

- `set_rank_ce` / `rank_losses`：从 `main_v5.py` 抽出来的多正例集合级排序损失。
  这是历史上唯一带来量级提升的改动（Hit@1 0.12 -> 0.29/0.36），所以保留并沿用同样的实现。
  `main_v5.py` 是靠 monkey-patch `base.train_one_epoch` 生效的，这里改成正常的函数调用。
- `cardinality_loss`：步骤 2 的集合大小回归损失。
- `base_solvent_loss`：步骤 4 的基础溶剂多标签损失。
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def set_rank_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    多正例集合级交叉熵：把多标签目标归一化成一个分布，再和 log_softmax 做 CE。

    与逐标签 BCE 的区别在于它约束的是**标签之间的相对排序**，
    因此直接作用于 top-1 是否正确 —— 这正是 Hit@1 关心的。

    logits  : (N, C)
    targets : (N, C) 多热
    """
    pos_cnt = targets.sum(dim=1, keepdim=True)
    valid = (pos_cnt.squeeze(1) > 0).float()
    tgt_dist = targets / (pos_cnt + 1e-6)
    logp = F.log_softmax(logits, dim=1)
    row_loss = -(tgt_dist * logp).sum(dim=1)
    w = valid
    if sample_weight is not None:
        w = w * sample_weight.view(-1).float()
    return (row_loss * w).sum() / (w.sum() + 1e-6)


def any_positive_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    「任意一个正例排在最前」的交叉熵：-log( sum_{p in P} softmax(logits)_p )。

    与 `set_rank_ce` 的区别在目标分布。`set_rank_ce` 把质量均匀摊到所有正标签上，
    于是平均 3.27 个正例时，模型被要求把它们**全部**排高；而 top-1 准确率只要求
    **其中一个**排第一。这个失配会稀释 top-1 的梯度信号，也解释了为什么单纯加大
    `set_rank_ce` 的权重到 2.0 就见顶。

    本损失只要求正例集合的总概率质量尽可能大，模型可以自由地把注意力集中在
    最容易的那个正例上，与 top-1 指标的定义严格对齐。
    """
    pos_cnt = targets.sum(dim=1)
    valid = (pos_cnt > 0).float()
    logp = F.log_softmax(logits, dim=1)
    masked = logp.masked_fill(targets <= 0, float("-inf"))
    # 无正例的行整行是 -inf，logsumexp 会得到 -inf，先用 valid 把它们的损失置 0
    row_loss = -torch.logsumexp(masked, dim=1)
    row_loss = torch.where(valid > 0, row_loss, torch.zeros_like(row_loss))
    w = valid
    if sample_weight is not None:
        w = w * sample_weight.view(-1).float()
    return (row_loss * w).sum() / (w.sum() + 1e-6)


def pairwise_logistic(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RankNet 式成对逻辑损失：对每个(正例, 负例)对惩罚 softplus(-(s_pos - s_neg))。

    优化的是成对顺序而不是整体分布，对正例个数不敏感。
    """
    pos = targets.clamp(min=0)
    neg = 1.0 - pos
    diff = logits.unsqueeze(2) - logits.unsqueeze(1)          # (N, C_pos, C_neg)
    pair_mask = pos.unsqueeze(2) * neg.unsqueeze(1)
    per_row = (F.softplus(-diff) * pair_mask).sum(dim=(1, 2))
    n_pairs = pair_mask.sum(dim=(1, 2))
    row_loss = per_row / (n_pairs + 1e-6)
    w = (n_pairs > 0).float()
    if sample_weight is not None:
        w = w * sample_weight.view(-1).float()
    return (row_loss * w).sum() / (w.sum() + 1e-6)


def top1_margin(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    top-1 铰链损失：要求最高分的正例比最高分的负例至少高出 margin。

    这是 top-1 准确率最直接的凸替代，只作用于决定 top-1 的那两个标签。
    """
    pos = targets.clamp(min=0)
    has_pos = (pos.sum(dim=1) > 0) & ((1.0 - pos).sum(dim=1) > 0)
    max_pos = logits.masked_fill(pos <= 0, float("-inf")).max(dim=1).values
    max_neg = logits.masked_fill(pos > 0, float("-inf")).max(dim=1).values
    row_loss = F.relu(margin - (max_pos - max_neg))
    row_loss = torch.where(has_pos, row_loss, torch.zeros_like(row_loss))
    w = has_pos.float()
    if sample_weight is not None:
        w = w * sample_weight.view(-1).float()
    return (row_loss * w).sum() / (w.sum() + 1e-6)


RANK_LOSS_FORMS = ("set_ce", "any_pos", "ranknet", "margin")


def _rank_term(
    form: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if form == "set_ce":
        return set_rank_ce(logits, targets, sample_weight)
    if form == "any_pos":
        return any_positive_ce(logits, targets, sample_weight)
    if form == "ranknet":
        return pairwise_logistic(logits, targets, sample_weight)
    if form == "margin":
        return top1_margin(logits, targets, sample_weight=sample_weight)
    raise ValueError(f"未知的 rank_loss 形式: {form}，可选 {RANK_LOSS_FORMS}")


def rank_losses(
    car_logits: torch.Tensor,
    y_car: torch.Tensor,
    step_true: torch.Tensor,
    sol_logits: torch.Tensor,
    y_sol: torch.Tensor,
    has_info: torch.Tensor,
    form: str = "set_ce",
) -> Dict[str, torch.Tensor]:
    """
    cartridge 与 solvent 的排序损失。

    solvent 只在"该样本有流程信息、且该步骤真实存在"的 (sample, step) 上计算 ——
    对真实不存在的步骤做排序没有意义。

    `form` 只作用于 cartridge。solvent 一律沿用 `set_ce`：溶剂的 top-1 不是我们
    追的指标，换形式只会引入无关变量。
    """
    loss_rank_car = _rank_term(form, car_logits, y_car)

    b, n_steps, n_sol = y_sol.shape
    sol_flat_logits = sol_logits.reshape(b * n_steps, n_sol)
    sol_flat_true = y_sol.reshape(b * n_steps, n_sol)
    step_mask = step_true.reshape(b * n_steps)
    info_mask = has_info.view(-1, 1).repeat(1, n_steps).reshape(b * n_steps)
    loss_rank_sol = set_rank_ce(
        sol_flat_logits,
        sol_flat_true,
        sample_weight=(step_mask * info_mask).float(),
    )
    return {"rank_cartridge": loss_rank_car, "rank_solvent": loss_rank_sol}


def cardinality_loss(
    car_count_pred: torch.Tensor,
    y_car: torch.Tensor,
    sol_count_pred: torch.Tensor,
    y_sol: torch.Tensor,
    has_info: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    集合大小回归损失（步骤 2）。

    在 `log1p` 空间上用 smooth_l1：cartridge 个数分布长尾（多数 1~5，少数十几个），
    直接在原始尺度回归会被少数大值主导。

    solvent 个数按 (sample, step) 计算，只在有流程信息的样本上算。
    真实不存在的步骤目标个数为 0，这是有意义的监督信号（模型应学会该步不出溶剂）。
    """
    car_count_true = y_car.sum(dim=1)
    loss_car_cnt = F.smooth_l1_loss(
        torch.log1p(car_count_pred),
        torch.log1p(car_count_true),
        reduction="mean",
    )

    sol_count_true = y_sol.sum(dim=2)                      # (B, 5)
    mask = has_info.view(-1, 1).expand_as(sol_count_true).float()
    elem = F.smooth_l1_loss(
        torch.log1p(sol_count_pred),
        torch.log1p(sol_count_true),
        reduction="none",
    )
    loss_sol_cnt = (elem * mask).sum() / (mask.sum() + 1e-6)

    return {"count_cartridge": loss_car_cnt, "count_solvent": loss_sol_cnt}


def count_ce_loss(
    car_count_logits: torch.Tensor,
    y_car: torch.Tensor,
    sol_count_logits: torch.Tensor,
    y_sol: torch.Tensor,
    has_info: torch.Tensor,
    label_smooth_adjacent: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    把集合大小当**分类**而不是回归来学。

    换掉回归的理由：真实个数是长尾的（cartridge 1145 个样本里 508 个只有 1 个，
    尾巴延伸到 23），而 L1/L2 回归在长尾下的最优解是中位数/均值，
    所以回归头必然收敛到"什么样本都猜 2~3 个"，实测逐样本相关系数只有 0.27。
    分类不受这个约束 —— 它可以把概率质量放在 1 上，同时保留识别大集合的能力。

    `label_smooth_adjacent` 把一部分目标概率分给相邻的桶。个数是**有序**的，
    把 5 预测成 4 显然比预测成 12 要好，而普通交叉熵对这两种错误一视同仁；
    给相邻桶分一点质量相当于把这个序关系告诉损失函数。
    """

    def _ce(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
        k = logits.size(-1)
        tgt = target.clamp(0, k - 1).long()
        dist = F.one_hot(tgt, num_classes=k).float()
        if label_smooth_adjacent > 0.0:
            e = float(label_smooth_adjacent)
            left = F.pad(dist, (1, 0))[..., :-1]
            right = F.pad(dist, (0, 1))[..., 1:]
            dist = (1.0 - e) * dist + 0.5 * e * left + 0.5 * e * right
            dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        row = -(dist * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if weight is None:
            return row.mean()
        return (row * weight).sum() / (weight.sum() + 1e-6)

    loss_car = _ce(car_count_logits, y_car.sum(dim=1), None)

    sol_count_true = y_sol.sum(dim=2)                                    # (B, 5)
    b, n_steps = sol_count_true.shape
    mask = has_info.view(-1, 1).expand_as(sol_count_true).reshape(-1).float()
    loss_sol = _ce(
        sol_count_logits.reshape(b * n_steps, -1),
        sol_count_true.reshape(-1),
        mask,
    )
    return {"count_cartridge": loss_car, "count_solvent": loss_sol}


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """
    非对称损失（ASL），逐元素返回，形状与 `logits` 相同。

    专门针对极端稀疏的多标签：溶剂有 5*378=1890 个标签位，正例率只有 0.0075，
    也就是每个样本有 1875 个负例。对称的 BCE 会被负例的梯度淹没，
    加 pos_weight 只是放大正例，改变不了"绝大多数梯度来自已经预测得很好的简单负例"。

    ASL 做两件事：
    1. 对负例用更大的 focal 指数 `gamma_neg`，把已经压得很低的简单负例的梯度进一步衰减；
    2. `clip` 做概率平移，把预测概率低于 `clip` 的负例梯度直接置零 —— 这些位置已经足够好，
       继续压它们只会挤占正例的学习信号。
    """
    p = torch.sigmoid(logits)
    p_neg = (p - float(clip)).clamp(min=0.0) if clip > 0 else p

    loss_pos = targets * torch.log(p.clamp_min(1e-8)) * ((1.0 - p) ** float(gamma_pos))
    loss_neg = (1.0 - targets) * torch.log((1.0 - p_neg).clamp_min(1e-8)) * (p_neg ** float(gamma_neg))
    return -(loss_pos + loss_neg)


def ratio_bucket_loss(
    ratio_logits: torch.Tensor,
    y_ratio: torch.Tensor,
    n_bins: int,
    label_smooth_adjacent: float = 0.1,
) -> torch.Tensor:
    """
    把 ratio 当分桶分类学，目标是提升"误差 ≤ 0.10 的命中率"。

    MSE 回归收敛到条件均值，输出连续且居中，所以平均误差小，
    但不容易正好落进 ±0.10 这个窄窗口 —— 实测我们 MAE 赢 DecisionTree 0.024，
    却在 ≤0.10 命中率上输它 0.021。ratio 真实值高度集中在少数常见配比
    （50:50、95:5 之类），分类能把概率质量直接压在这些众数上。

    和 `count_ce_loss` 同理，用相邻桶平滑来编码"配比是有序量"这个先验。
    """
    edges = torch.clamp((y_ratio * n_bins).long(), 0, n_bins - 1)
    dist = F.one_hot(edges, num_classes=n_bins).float()
    if label_smooth_adjacent > 0.0:
        e = float(label_smooth_adjacent)
        left = F.pad(dist, (1, 0))[..., :-1]
        right = F.pad(dist, (0, 1))[..., 1:]
        dist = (1.0 - e) * dist + 0.5 * e * left + 0.5 * e * right
        dist = dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return -(dist * F.log_softmax(ratio_logits, dim=-1)).sum(dim=-1).mean()


def base_solvent_loss(
    base_logits: torch.Tensor,
    y_base: torch.Tensor,
    step_true: torch.Tensor,
    has_info: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    pos_step_boost: float = 2.2,
    neg_step_weight: float = 1.0,
) -> torch.Tensor:
    """
    基础溶剂多标签损失（步骤 4）。

    加权方式与 `main.py` 里 solvent 的处理保持一致：真实存在的步骤加权 `pos_step_boost`，
    不存在的步骤加权 `neg_step_weight`，整体再按 `has_info` 掩码。

    base_logits : (B, 5, S_base)
    y_base      : (B, 5, S_base)
    """
    elem = F.binary_cross_entropy_with_logits(
        base_logits, y_base, pos_weight=pos_weight, reduction="none"
    )
    step_weight = (step_true * pos_step_boost + (1.0 - step_true) * neg_step_weight).unsqueeze(-1)
    weight = has_info.view(-1, 1, 1) * step_weight
    return (elem * weight).sum() / (weight.sum() * y_base.size(2) + 1e-6)
