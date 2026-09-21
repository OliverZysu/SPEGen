"""
步骤 4：把溶剂组合词表拆成基础溶剂，缓解标签稀疏。

现状：词表共 378 项 = 79 个单一溶剂 + 299 个双组分组合（如 `dichloromethane+hexane`），
模型要在 5×378 = 1890 维上做多标签分类。后果是 **solvent 的 exact_match 在全部 28 组
历史实验里恒为 0.0** —— 从未有任何一个样本把 5 个步骤的溶剂集合全部预测对。

拆解后基础溶剂只有 81 个，标签空间压缩 4.7 倍。做法是两级预测：

- **基础级**（辅助任务）：预测每一步用到哪些基础溶剂，(5, 81) 多标签。
  `acetone+water` 会同时给 `acetone` 和 `water` 记正例，正例密度大幅提高。
- **组合级**（主任务，保持 378 维不变）：仍然输出原词表上的概率，但用基础级概率
  构造先验去重排序。指标维度不变，所以能和历史实验、baseline 直接对比。

组合先验用几何平均：`prior(a+b) = sqrt(p_base(a) * p_base(b))`。
一个组合只有在它的**每个**组分都被看好时才会被抬高，这正是我们想要的语义。
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class SolventDecomposition:
    """组合词表 -> 基础溶剂词表的映射。"""

    base_vocab: List[str]            # 长度 S_base
    base2id: Dict[str, int]
    combo_vocab: List[str]           # 长度 S_combo，等于原 solvent_vocab
    # (S_combo, S_base) 的 0/1 矩阵：membership[c, b] = 1 表示组合 c 含基础溶剂 b
    membership: np.ndarray
    # 每个组合的组分个数，(S_combo,)
    arity: np.ndarray

    @property
    def n_base(self) -> int:
        return len(self.base_vocab)

    @property
    def n_combo(self) -> int:
        return len(self.combo_vocab)


def decompose_vocab(
    solvent_vocab: Sequence[str],
    unk_token: str = "__UNK__",
    separator: str = "+",
) -> SolventDecomposition:
    """
    按 `separator` 拆分组合词表。`unk_token` 视为不可拆的独立基础溶剂。

    基础词表按字典序排序，保证同一份数据多次运行的 id 稳定。
    """
    combo_vocab = list(solvent_vocab)

    base_set = set()
    parts_per_combo: List[List[str]] = []
    for name in combo_vocab:
        if name == unk_token:
            parts = [unk_token]
        else:
            parts = [p.strip() for p in name.split(separator)]
            parts = [p for p in parts if p]
            if not parts:
                parts = [unk_token]
        parts_per_combo.append(parts)
        base_set.update(parts)

    base_vocab = sorted(base_set)
    base2id = {b: i for i, b in enumerate(base_vocab)}

    membership = np.zeros((len(combo_vocab), len(base_vocab)), dtype=np.float32)
    arity = np.zeros((len(combo_vocab),), dtype=np.float32)
    for c, parts in enumerate(parts_per_combo):
        uniq = sorted(set(parts))
        for p in uniq:
            membership[c, base2id[p]] = 1.0
        arity[c] = float(len(uniq))

    return SolventDecomposition(
        base_vocab=base_vocab,
        base2id=base2id,
        combo_vocab=combo_vocab,
        membership=membership,
        arity=arity,
    )


def combo_labels_to_base(
    y_solvent: np.ndarray,
    decomp: SolventDecomposition,
) -> np.ndarray:
    """
    把组合级标签 (N, 5, S_combo) 投影成基础级标签 (N, 5, S_base)。

    只要某一步用了含 `a` 的任何组合，`a` 在该步就记正例。
    """
    n, n_steps, n_combo = y_solvent.shape
    if n_combo != decomp.n_combo:
        raise ValueError(f"标签维度 {n_combo} 与词表大小 {decomp.n_combo} 不一致")
    flat = y_solvent.reshape(n * n_steps, n_combo).astype(np.float32)
    base = flat @ decomp.membership          # (N*5, S_base) 计数
    base = (base > 0).astype(np.float32)     # 转成 0/1
    return base.reshape(n, n_steps, decomp.n_base)


def base_prob_to_combo_prior(
    base_prob: np.ndarray,
    decomp: SolventDecomposition,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    用基础溶剂概率构造组合级先验，(N, 5, S_base) -> (N, 5, S_combo)。

    取组分概率的几何平均：`prior(a+b) = (p_a * p_b) ** (1/2)`。
    在 log 空间做矩阵乘法实现，避免逐组合 Python 循环。
    """
    n, n_steps, n_base = base_prob.shape
    if n_base != decomp.n_base:
        raise ValueError(f"基础概率维度 {n_base} 与基础词表 {decomp.n_base} 不一致")

    log_p = np.log(np.clip(base_prob, eps, 1.0).astype(np.float32))
    flat = log_p.reshape(n * n_steps, n_base)
    # (N*5, S_base) @ (S_base, S_combo) -> 每个组合的 log 概率之和
    log_sum = flat @ decomp.membership.T
    log_mean = log_sum / np.maximum(decomp.arity[None, :], 1.0)
    prior = np.exp(log_mean)
    return prior.reshape(n, n_steps, decomp.n_combo).astype(np.float32)


def blend_combo_with_prior(
    combo_prob: np.ndarray,
    prior: np.ndarray,
    weight: float = 0.3,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    把组合级主预测与基础级先验做几何加权融合：`p ** (1-w) * prior ** w`。

    `weight=0` 完全退回主预测（可用于消融），`weight=1` 完全用先验。
    融合在 log 空间进行，结果仍落在 (0, 1]。
    """
    w = float(min(max(weight, 0.0), 1.0))
    if w <= 0.0:
        return combo_prob
    log_p = np.log(np.clip(combo_prob, eps, 1.0))
    log_q = np.log(np.clip(prior, eps, 1.0))
    return np.exp((1.0 - w) * log_p + w * log_q).astype(np.float32)
