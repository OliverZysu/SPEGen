"""
数值特征嵌入：让 MLP 更会消化 21 维连续描述符。

**这里不引入任何新的输入信息**，输入仍然是那 21 个分子描述符。改的是模型内部
怎么表示它们 —— 每个标量先被展开成一个向量，再送进后续 MLP。

为什么需要这个：我们和树模型的差距集中在 top-1 排序（cartridge top-1 我们 0.51，
RandomForest 0.59），而树模型在小样本表格数据上的优势来自它能用轴对齐的阈值切分
直接表达"某个描述符超过某个值"这类分段关系。普通 MLP 的第一层是全局线性投影，
要靠多层堆叠才能逼近这种分段行为，在 9000 个训练样本上很难学出来。
把标量先做分段/周期展开，等于把这类关系变成线性可分的，这是 tabular 深度学习里
缩小 MLP 与 GBDT 差距最有效的单一改动。

两种编码：

- `periodic`：x -> [sin(2*pi*c_k*x), cos(2*pi*c_k*x)]，c 是可学习频率。
  不依赖数据统计，实现自洽。
- `ple`（piecewise-linear）：按训练集分位数把取值域切成若干桶，
  编码成"前面的桶全填 1、当前桶填部分、后面的桶填 0"。
  和决策树的切分点最接近，但需要从训练集拿分位数（只用训练集，不泄漏）。
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class PeriodicEmbedding(nn.Module):
    """
    每个特征独立的周期编码 + 逐特征线性层 + ReLU。

    输出维度是 `n_features * out_dim`，展平后接原来的 MLP 编码器。
    """

    def __init__(self, n_features: int, n_freq: int = 24, out_dim: int = 16, sigma: float = 0.05):
        super().__init__()
        self.n_features = int(n_features)
        self.out_dim = int(out_dim)
        # 频率初始化的尺度很关键：sigma 太大会让编码在输入的微小变化下剧烈振荡，
        # 相当于给模型一堆高频噪声；论文推荐的量级是 0.01~0.1。
        self.coeffs = nn.Parameter(torch.randn(self.n_features, n_freq) * float(sigma))
        # 逐特征独立的线性层，用 einsum 一次算完，避免 21 个小 Linear 的循环开销
        self.weight = nn.Parameter(torch.randn(self.n_features, 2 * n_freq, self.out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.n_features, self.out_dim))

    @property
    def output_dim(self) -> int:
        return self.n_features * self.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) -> (B, F, 2*n_freq)
        z = 2.0 * torch.pi * self.coeffs.unsqueeze(0) * x.unsqueeze(-1)
        z = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        z = torch.einsum("bfk,fko->bfo", z, self.weight) + self.bias.unsqueeze(0)
        return torch.relu(z).flatten(start_dim=1)


class PiecewiseLinearEmbedding(nn.Module):
    """
    分位数分桶的分段线性编码 + 逐特征线性层 + ReLU。

    桶边界来自**训练集**分位数（`from_train_data` 构造），推理时固定不变。
    对第 t 个桶 [b_t, b_{t+1})，编码值是 x 落在该桶内的相对位置，
    落在它之前的桶记 1、之后的桶记 0 —— 这样编码对 x 单调，且把
    "x 是否越过 b_t"变成了线性可读的信号。
    """

    def __init__(self, bin_edges: torch.Tensor, out_dim: int = 16):
        super().__init__()
        # bin_edges: (F, T+1)
        self.register_buffer("edges", bin_edges)
        self.n_features = int(bin_edges.shape[0])
        self.n_bins = int(bin_edges.shape[1]) - 1
        self.out_dim = int(out_dim)
        self.weight = nn.Parameter(torch.randn(self.n_features, self.n_bins, self.out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.n_features, self.out_dim))

    @classmethod
    def from_train_data(
        cls, x_train: np.ndarray, n_bins: int = 24, out_dim: int = 16
    ) -> "PiecewiseLinearEmbedding":
        """分位数取桶边界。只用训练集，避免泄漏。"""
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(np.asarray(x_train, dtype=np.float64), qs, axis=0).T  # (F, T+1)
        # 常数特征或重复分位点会让桶宽为 0，后面除法会炸，这里强制单调递增
        for f in range(edges.shape[0]):
            for t in range(1, edges.shape[1]):
                if edges[f, t] <= edges[f, t - 1]:
                    edges[f, t] = edges[f, t - 1] + 1e-6
        return cls(torch.tensor(edges, dtype=torch.float32), out_dim=out_dim)

    @property
    def output_dim(self) -> int:
        return self.n_features * self.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = self.edges[:, :-1].unsqueeze(0)                      # (1, F, T)
        hi = self.edges[:, 1:].unsqueeze(0)
        frac = (x.unsqueeze(-1) - lo) / (hi - lo)
        z = frac.clamp(0.0, 1.0)                                  # (B, F, T)
        z = torch.einsum("bft,fto->bfo", z, self.weight) + self.bias.unsqueeze(0)
        return torch.relu(z).flatten(start_dim=1)


def build_numeric_embedding(
    kind: str,
    n_features: int,
    x_train: Optional[np.ndarray] = None,
    n_bins: int = 24,
    out_dim: int = 16,
    sigma: float = 0.05,
) -> Optional[nn.Module]:
    """`kind` 取 `none` / `periodic` / `ple`。返回 None 表示不做嵌入（原始行为）。"""
    if kind == "none":
        return None
    if kind == "periodic":
        return PeriodicEmbedding(n_features, n_freq=n_bins, out_dim=out_dim, sigma=sigma)
    if kind == "ple":
        if x_train is None:
            raise ValueError("ple 编码需要训练集特征矩阵来取分位数桶边界")
        return PiecewiseLinearEmbedding.from_train_data(x_train, n_bins=n_bins, out_dim=out_dim)
    raise ValueError(f"未知的数值嵌入类型 {kind}")
