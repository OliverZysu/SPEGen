"""
把一个 v6 run 目录还原成「模型 + 数据 + 预测数组」，供各诊断工具复用。

所有配置都从 `best_model.pt` 里保存的 `args` 还原，所以不需要再手工传参，
也不会出现诊断时用错开关的情况。
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from utils.dataloader import build_loaders

from ..data import PreparedData, prepare_data
from ..evaluate import PredictionBundle, collect_predictions
from ..model import SPEModelV6, build_model


@dataclass
class LoadedRun:
    run_dir: str
    args: Dict[str, Any]
    prepared: PreparedData
    model: SPEModelV6
    per_label_thresholds: Dict[str, np.ndarray]

    @property
    def init_thresholds(self) -> Dict[str, float]:
        return {
            "cartridge": float(self.args.get("thr_cartridge", 0.5)),
            "step": float(self.args.get("thr_step", 0.5)),
            "solvent": float(self.args.get("thr_solvent", 0.5)),
        }


def load_run(run_dir: str, device_str: str = "cpu") -> LoadedRun:
    ckpt = torch.load(
        os.path.join(run_dir, "best_model.pt"), map_location=device_str, weights_only=False
    )
    a: Dict[str, Any] = ckpt["args"]

    prepared = prepare_data(
        data_dir=a["data_dir"],
        cache_path=a["cache_path"],
        seed=a["seed"],
        train_ratio=a["train_ratio"],
        val_ratio=a["val_ratio"],
        require_spe_info=a["require_spe_info"],
        unk_solvent_token=a["unk_solvent_token"],
        use_functional_groups=not a["no_functional_groups"],
        fg_transform=a["fg_transform"],
        fg_min_pos=a["fg_min_pos"],
        decompose_solvent=not a["no_decompose_solvent"],
        feature_transform=a.get("feature_transform", "zscore"),
    )
    model = build_model(
        input_dim=prepared.input_dim,
        n_cartridge=len(prepared.artifacts.cartridge_cols),
        n_steps=prepared.n_steps,
        n_solvent=len(prepared.artifacts.solvent_vocab),
        hidden_dim=a["hidden_dim"],
        enc_layers=a["enc_layers"],
        dropout=a["dropout"],
        n_base_solvent=prepared.n_base_solvent,
        predict_cardinality=not a["no_cardinality"],
        # 新增开关都用 .get 取，这样旧的 checkpoint（没有这些键）仍然能加载
        count_mode=a.get("count_mode", "regression"),
        n_car_count_bins=a.get("n_car_count_bins", 16),
        n_sol_count_bins=a.get("n_sol_count_bins", 12),
        count_reduce=a.get("count_reduce", "argmax"),
        numeric_embed=a.get("numeric_embed", "none"),
        numeric_embed_bins=a.get("numeric_embed_bins", 24),
        numeric_embed_dim=a.get("numeric_embed_dim", 16),
        numeric_embed_sigma=a.get("numeric_embed_sigma", 0.05),
        x_train=prepared.ds.xs[prepared.train_idx],
        ratio_mode=a.get("ratio_mode", "regression"),
        n_ratio_bins=a.get("n_ratio_bins", 20),
        uncertainty_weighting=a.get("uncertainty_weighting", False),
    ).to(device_str)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    thr = {k: np.asarray(v, dtype=np.float32) for k, v in ckpt["per_label_thresholds"].items()}
    return LoadedRun(run_dir=run_dir, args=a, prepared=prepared, model=model,
                     per_label_thresholds=thr)


def collect_split(
    loaded: LoadedRun,
    split: str = "test",
    device_str: str = "cpu",
) -> PredictionBundle:
    """在指定 split 上跑一遍前向。`split` 取 `train` / `val` / `test`。"""
    p = loaded.prepared
    a = loaded.args
    loaders = build_loaders(
        p.ds, p.train_idx, p.val_idx, p.test_idx,
        batch_size=a["batch_size"], num_workers=0,
    )
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]
    return collect_predictions(
        loaded.model, loader, torch.device(device_str),
        solvent_step_beta=a["solvent_step_beta"],
        decomp=p.decomp,
        base_prior_weight=a["base_prior_weight"],
    )
