"""
统一的数据准备入口：在既有 `utils.dataloader.build_datasets()` 之上做两件增量的事。

1. 把官能团特征（步骤 3）拼到 `ds.xs` 后面。
2. 计算溶剂组合词表的拆解（步骤 4）。

`utils/dataloader.py` 一行没改，所以 `data/cache/dataset_cache.pkl` 仍然可以复用 ——
缓存里存的是未拼接的 26 维 `xs`，拼接发生在读缓存之后。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from utils.dataloader import DatasetArtifacts, SPEDataset, build_datasets, split_indices
from utils.parsing import STEP_NAMES

from .features import FunctionalGroupFeatures, build_functional_group_features
from .solvent_vocab import SolventDecomposition, decompose_vocab


@dataclass
class PreparedData:
    ds: SPEDataset
    artifacts: DatasetArtifacts
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: List[str]
    mean: np.ndarray
    std: np.ndarray
    fg: Optional[FunctionalGroupFeatures] = None
    decomp: Optional[SolventDecomposition] = None

    @property
    def input_dim(self) -> int:
        return int(self.ds.xs.shape[1])

    @property
    def n_steps(self) -> int:
        return len(STEP_NAMES)

    @property
    def n_base_solvent(self) -> int:
        return 0 if self.decomp is None else self.decomp.n_base


def standardize_inplace_safe(xs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    """
    标准化，但给 std 设了下界。

    `main.standardize_inplace` 用的是 `std + 1e-8`；官能团里可能出现训练集内方差极小的列，
    除以 1e-8 会炸出巨大数值。对有正常方差的数值描述符，两种写法在浮点精度内等价。
    """
    xs -= mean[None, :]
    xs /= np.maximum(std, 1e-6)[None, :]


FEATURE_TRANSFORMS = ("zscore", "rank_gauss", "log1p", "robust")


def apply_feature_transform(
    xs: np.ndarray,
    train_idx: np.ndarray,
    kind: str = "zscore",
    clip: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对输入特征做单调变换，再标准化。返回 (变换后的 xs, mean, std)。

    动机：这批分子描述符极度偏斜——标准化后仍有 19/21 维的偏度绝对值大于 2、
    20/21 维的峰度大于 10，最极端的一维偏度 53.8、峰度 3269。z-score 只做平移缩放，
    不改变分布形状，于是绝大多数样本被挤在 0 附近，少数离群点坐在几十个标准差之外，
    第一层的梯度基本被离群点支配。树模型对任何单调变换都不变，MLP 不是，
    这很可能就是二者在 top-1 类指标上差距的来源。

    所有变换的参数（分位点、均值方差）都只在训练集上拟合，避免信息泄漏。
    维度不变，也不引入任何新特征。

    - `zscore`      现状：直接标准化
    - `rank_gauss`  逐维分位数正态变换，彻底消除偏斜与离群点，零参数
    - `log1p`       有符号 log1p 压缩长尾，再标准化
    - `robust`      用中位数与四分位距标准化，再按 `clip` 截断
    """
    if kind not in FEATURE_TRANSFORMS:
        raise ValueError(f"未知的 feature_transform: {kind}，可选 {FEATURE_TRANSFORMS}")

    xs = np.asarray(xs, dtype=np.float64)
    x_train = xs[train_idx]

    if kind == "rank_gauss":
        from scipy.special import erfinv

        out = np.empty_like(xs)
        n_train = x_train.shape[0]
        for j in range(xs.shape[1]):
            ref = np.sort(x_train[:, j])
            # 每个值在训练集里的百分位；并列取区间中点，避免把同值样本人为拉开
            lo = np.searchsorted(ref, xs[:, j], side="left")
            hi = np.searchsorted(ref, xs[:, j], side="right")
            pct = (0.5 * (lo + hi)) / (n_train + 1.0)
            pct = np.clip(pct, 1e-6, 1.0 - 1e-6)
            out[:, j] = np.sqrt(2.0) * erfinv(2.0 * pct - 1.0)
        xs = out
    elif kind == "log1p":
        xs = np.sign(xs) * np.log1p(np.abs(xs))
    elif kind == "robust":
        med = np.median(x_train, axis=0)
        q75, q25 = np.percentile(x_train, [75, 25], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        xs = np.clip((xs - med[None, :]) / iqr[None, :], -clip, clip)

    # rank_gauss 出来已经近似标准正态，再标准化一次基本是恒等操作，
    # 保留是为了让四种变换共用同一个 mean/std 接口
    x_train = xs[train_idx]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    xs = (xs - mean[None, :]) / np.maximum(std, 1e-6)[None, :]
    return xs.astype(np.float32), mean, std


def prepare_data(
    data_dir: str = "data",
    cache_path: str = "data/cache/dataset_cache.pkl",
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    require_spe_info: bool = False,
    unk_solvent_token: str = "__UNK__",
    use_functional_groups: bool = True,
    fg_transform: str = "log1p",
    fg_min_pos: int = 20,
    decompose_solvent: bool = True,
    feature_transform: str = "zscore",
) -> PreparedData:
    """
    读数据、切分、（可选）拼官能团特征、标准化、（可选）拆解溶剂词表。

    切分用的是 `utils.dataloader.split_indices`，与历史实验完全同一套逻辑，
    所以同一个 seed 下测试集就是同一批样本，指标可直接对比。

    官能团的低频过滤只在**训练集**上统计（`fit_indices=train_idx`），避免信息泄漏。
    特征变换与标准化的参数同样只用训练集拟合，见 `apply_feature_transform`。
    """
    ds, artifacts = build_datasets(
        data_dir=data_dir,
        cache_path=cache_path,
        require_spe_info=require_spe_info,
        unk_solvent_token=unk_solvent_token,
    )
    train_idx, val_idx, test_idx = split_indices(
        len(ds), seed=seed, train_ratio=train_ratio, val_ratio=val_ratio
    )

    feature_names = list(artifacts.descriptor_cols)
    fg: Optional[FunctionalGroupFeatures] = None

    if use_functional_groups:
        fg = build_functional_group_features(
            cids=ds.cids,
            data_dir=data_dir,
            transform=fg_transform,
            min_pos=fg_min_pos,
            fit_indices=train_idx,
        )
        ds.xs = np.concatenate([ds.xs, fg.matrix], axis=1).astype(np.float32)
        feature_names = feature_names + [f"fg::{c}" for c in fg.columns]

    ds.xs, mean, std = apply_feature_transform(
        ds.xs, train_idx, kind=feature_transform
    )

    decomp: Optional[SolventDecomposition] = None
    if decompose_solvent:
        decomp = decompose_vocab(artifacts.solvent_vocab, unk_token=unk_solvent_token)

    return PreparedData(
        ds=ds,
        artifacts=artifacts,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_names=feature_names,
        mean=mean,
        std=std,
        fg=fg,
        decomp=decomp,
    )


def describe(prepared: PreparedData) -> str:
    """打印一段数据摘要，方便在训练日志开头核对配置是否如预期生效。"""
    lines = [
        f"样本数 {len(prepared.ds)}  切分 train/val/test = "
        f"{len(prepared.train_idx)}/{len(prepared.val_idx)}/{len(prepared.test_idx)}",
        f"输入维度 {prepared.input_dim}"
        + (
            f"（{len(prepared.artifacts.descriptor_cols)} 维描述符 + "
            f"{prepared.fg.matrix.shape[1]} 维官能团，变换={prepared.fg.transform}）"
            if prepared.fg is not None
            else f"（{len(prepared.artifacts.descriptor_cols)} 维描述符，未启用官能团）"
        ),
        f"cartridge 类别 {len(prepared.artifacts.cartridge_cols)}  "
        f"步骤 {prepared.n_steps}  溶剂组合词表 {len(prepared.artifacts.solvent_vocab)}",
    ]
    if prepared.fg is not None and prepared.fg.n_missing > 0:
        lines.append(f"警告：{prepared.fg.n_missing} 个样本未能关联到官能团表，已填 0")
    if prepared.decomp is not None:
        n_combo = int((prepared.decomp.arity > 1).sum())
        lines.append(
            f"溶剂拆解已启用：{prepared.decomp.n_combo} 个组合（其中 {n_combo} 个多组分）"
            f" -> {prepared.decomp.n_base} 个基础溶剂"
        )
    return "\n".join(lines)
