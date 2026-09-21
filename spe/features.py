"""
步骤 3：把官能团特征接进模型输入。

现状：`utils/dataloader.py:138-140` 只取 `Complexity_value` ~ `LogP_value` 这 21 列数值描述符，
`data/functional_groups_result.csv` 里现成的 85 列官能团计数完全没用上。
按默认 `min_pos=20` 过滤后可用 75 列，输入维度从 21 涨到 96。

而 `correlation-analysis/correlation_conclusions.md` 自己的结论显示官能团与溶剂的
Spearman 相关可达 0.23（`fr_halogen` × `dichloromethane+hexane`），远强于描述符层面的信号；
官能团表按 `canSMILES_value` 与分子表 **100% 可关联**（11441/11441）。

本模块只负责产出对齐好的特征矩阵，由 `spe/data.py` 拼到 `ds.xs` 后面 ——
不修改 `utils/dataloader.py`，历史实验的输入维度不受影响。
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.parsing import parse_list_str

_PAREN = re.compile(r"\s*[（(][^）)]*[）)]\s*$")


def _short_name(col: str) -> str:
    """`fr_Al_COO(脂肪族羧酸)` -> `fr_Al_COO`，便于在报告里显示。"""
    return _PAREN.sub("", col).strip()


@dataclass
class FunctionalGroupFeatures:
    """对齐到数据集样本顺序的官能团特征。"""

    matrix: np.ndarray          # (N, F) float32
    columns: List[str]          # 长度 F，已去掉括号里的中文说明
    raw_columns: List[str]      # 长度 F，CSV 原始列名
    n_missing: int              # 未能在官能团表里找到 SMILES 的样本数（这些行填 0）
    transform: str


def _load_fg_table(fg_csv_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """读官能团表，返回 {canSMILES: 计数向量} 与列名列表。"""
    with open(fg_csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        fg_cols = header[1:]
        table: Dict[str, np.ndarray] = {}
        for row in reader:
            if not row:
                continue
            smiles = row[0].strip()
            if not smiles:
                continue
            vals = np.zeros((len(fg_cols),), dtype=np.float32)
            for j, cell in enumerate(row[1 : 1 + len(fg_cols)]):
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    vals[j] = float(cell)
                except ValueError:
                    vals[j] = 0.0
            table[smiles] = vals
    return table, fg_cols


def _load_cid_to_smiles(processed_csv_path: str) -> Dict[str, str]:
    with open(processed_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        i_cid = header.index("CID")
        i_smi = header.index("canSMILES_value")
        out: Dict[str, str] = {}
        for row in reader:
            cid_list = parse_list_str(row[i_cid])
            if not cid_list:
                continue
            out[cid_list[0]] = row[i_smi].strip()
    return out


def build_functional_group_features(
    cids: Sequence[str],
    data_dir: str = "data",
    transform: str = "log1p",
    min_pos: int = 20,
    fit_indices: Optional[np.ndarray] = None,
) -> FunctionalGroupFeatures:
    """
    按 `cids` 给出的样本顺序构造官能团特征矩阵。

    参数
    ----
    transform : `"binary"`（是否出现）/ `"count"`（原始计数）/ `"log1p"`（默认，压缩长尾计数）
    min_pos   : 低频官能团过滤阈值。非零样本数低于该值的列直接丢掉，避免给模型喂噪声。
    fit_indices : 用于统计低频的样本下标，**应当只传训练集下标**以避免信息泄漏。
                  为 None 时退化为在全体样本上统计。

    未能关联到 SMILES 的样本整行填 0，并在 `n_missing` 里报出数量。
    """
    if transform not in {"binary", "count", "log1p"}:
        raise ValueError(f"未知的 transform: {transform}")

    fg_path = os.path.join(data_dir, "functional_groups_result.csv")
    proc_path = os.path.join(data_dir, "processed_molecular_data.csv")
    if not os.path.exists(fg_path):
        raise FileNotFoundError(f"找不到官能团表: {fg_path}")

    table, fg_cols = _load_fg_table(fg_path)
    cid2smi = _load_cid_to_smiles(proc_path)

    n = len(cids)
    mat = np.zeros((n, len(fg_cols)), dtype=np.float32)
    n_missing = 0
    for i, cid in enumerate(cids):
        smiles = cid2smi.get(str(cid), "")
        vec = table.get(smiles)
        if vec is None:
            n_missing += 1
            continue
        mat[i] = vec

    # 低频过滤：只在 fit_indices（训练集）上统计非零样本数
    stat_rows = mat if fit_indices is None else mat[np.asarray(fit_indices, dtype=np.int64)]
    nonzero_counts = (stat_rows > 0).sum(axis=0)
    keep = nonzero_counts >= int(min_pos)
    if not keep.any():
        raise ValueError(f"min_pos={min_pos} 过滤掉了所有官能团列，请调小该值")

    mat = mat[:, keep]
    kept_raw = [c for c, k in zip(fg_cols, keep.tolist()) if k]

    if transform == "binary":
        mat = (mat > 0).astype(np.float32)
    elif transform == "log1p":
        mat = np.log1p(mat).astype(np.float32)

    return FunctionalGroupFeatures(
        matrix=mat,
        columns=[_short_name(c) for c in kept_raw],
        raw_columns=kept_raw,
        n_missing=n_missing,
        transform=transform,
    )
