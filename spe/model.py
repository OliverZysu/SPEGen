"""
步骤 2（模型侧）+ 步骤 4（模型侧）：在既有主干上加三个新头。

`SPEModelV6` 继承 `model.SPEBaselineModel`，复用它的编码器、四个任务塔、
cartridge/step/solvent 头和条件式 ratio 头，只新增：

- `cartridge_count_head` : 预测该样本有几个 cartridge，(B, 1)
- `solvent_count_head`   : 预测每一步有几个溶剂，(B, 5)
- `base_solvent_head`    : 基础溶剂多标签，(B, 5, S_base)，步骤 4 的辅助任务

前两个头服务于基数感知解码：现在模型每样本预测 5.97 个 cartridge，而真实只有 3.27 个
（1.82 倍过预测），逐标签阈值无法修正这个系统性偏差，而"预测个数 + 取 top-k"可以。

`forward()` 没有调用 `super().forward()`，而是复刻了父类的计算图后再挂新头。
这是有意的：父类 `forward` 不返回各任务塔的中间表示 `h_car`/`h_sol`，若在子类里
重复调用任务塔会二次采样 dropout，训练结果和父类不一致。
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from model import MLPEncoder, SPEBaselineModel

from .numeric_embed import build_numeric_embedding


class SPEModelV6(SPEBaselineModel):
    """带集合大小预测头与基础溶剂头的多任务模型。"""

    def __init__(
        self,
        input_dim: int,
        n_cartridge: int,
        n_steps: int,
        n_solvent: int,
        hidden_dim: int = 256,
        enc_layers: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
        step_emb_dim: int = 32,
        solvent_emb_dim: int = 64,
        conc_hidden_dim: int = 256,
        n_base_solvent: int = 0,
        predict_cardinality: bool = True,
        count_mode: str = "regression",
        n_car_count_bins: int = 16,
        n_sol_count_bins: int = 12,
        count_reduce: str = "argmax",
        numeric_embed: str = "none",
        numeric_embed_bins: int = 24,
        numeric_embed_dim: int = 16,
        numeric_embed_sigma: float = 0.05,
        x_train: Optional[np.ndarray] = None,
        ratio_mode: str = "regression",
        n_ratio_bins: int = 20,
        uncertainty_weighting: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            n_cartridge=n_cartridge,
            n_steps=n_steps,
            n_solvent=n_solvent,
            hidden_dim=hidden_dim,
            enc_layers=enc_layers,
            dropout=dropout,
            activation=activation,
            step_emb_dim=step_emb_dim,
            solvent_emb_dim=solvent_emb_dim,
            conc_hidden_dim=conc_hidden_dim,
        )
        self.n_base_solvent = int(n_base_solvent)
        self.predict_cardinality = bool(predict_cardinality)
        self.count_mode = str(count_mode)
        self.count_reduce = str(count_reduce)
        self.n_car_count_bins = int(n_car_count_bins)
        self.n_sol_count_bins = int(n_sol_count_bins)
        self.ratio_mode = str(ratio_mode)
        self.n_ratio_bins = int(n_ratio_bins)

        # 数值特征嵌入：输入信息不变，只是换一种内部表示。
        # 启用后编码器的输入维度变成 n_features * numeric_embed_dim，所以要重建编码器。
        self.numeric_embedding = build_numeric_embedding(
            kind=numeric_embed, n_features=input_dim, x_train=x_train,
            n_bins=numeric_embed_bins, out_dim=numeric_embed_dim, sigma=numeric_embed_sigma,
        )
        if self.numeric_embedding is not None:
            self.encoder = MLPEncoder(
                input_dim=self.numeric_embedding.output_dim,
                hidden_dim=hidden_dim, num_layers=enc_layers,
                dropout=dropout, activation=activation,
            )

        if self.predict_cardinality:
            if self.count_mode == "classification":
                self.cartridge_count_head = nn.Linear(hidden_dim, self.n_car_count_bins)
                self.solvent_count_head = nn.Linear(hidden_dim, n_steps * self.n_sol_count_bins)
            else:
                self.cartridge_count_head = nn.Linear(hidden_dim, 1)
                self.solvent_count_head = nn.Linear(hidden_dim, n_steps)
        else:
            self.cartridge_count_head = None
            self.solvent_count_head = None

        if self.n_base_solvent > 0:
            self.base_solvent_head = nn.Linear(hidden_dim, n_steps * self.n_base_solvent)
        else:
            self.base_solvent_head = None

        # ratio 分桶头。与回归头并存但只有一个会被用到，取决于 ratio_mode。
        if self.ratio_mode == "bucket":
            self.ratio_bucket_mlp = nn.Sequential(
                nn.Linear(hidden_dim + step_emb_dim + solvent_emb_dim, conc_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(conc_hidden_dim, self.n_ratio_bins),
            )
        else:
            self.ratio_bucket_mlp = None

        # 不确定性加权（Kendall et al.）：让四个主任务的相对权重可学习，
        # 而不是人工定 1:1:1:1。梯度会自动压低噪声大的任务，
        # 这是缓解多任务干扰最轻量的做法（只加 4 个参数）。
        self.uncertainty_weighting = bool(uncertainty_weighting)
        if self.uncertainty_weighting:
            self.task_log_var = nn.Parameter(torch.zeros(4))
        else:
            self.task_log_var = None

    def _counts_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        把个数分布变成一个标量个数，供解码用。

        默认取众数（argmax）而不是期望：取期望等于把分布重新压回均值附近，
        而"回归到均值"正是我们要摆脱的问题，用期望会把分类的好处抵消掉。
        """
        if self.count_reduce == "expectation":
            p = torch.softmax(logits, dim=-1)
            grid = torch.arange(logits.size(-1), device=logits.device, dtype=p.dtype)
            return (p * grid).sum(dim=-1)
        return torch.argmax(logits, dim=-1).to(logits.dtype)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x if self.numeric_embedding is None else self.numeric_embedding(x))

        h_car = self.cartridge_tower(h)
        h_step = self.step_tower(h)
        h_sol = self.solvent_tower(h)

        cartridge_logits = self.cartridge_head(h_car)
        step_logits = self.step_head(h_step)
        solvent_cond_logits = self.solvent_head(h_sol).view(-1, self.n_steps, self.n_solvent)
        # 与父类一致的 step 引导门控（logit 空间相加）
        solvent_logits = solvent_cond_logits + self.solvent_step_gate_alpha * step_logits.unsqueeze(-1)

        out: Dict[str, torch.Tensor] = {
            "h": h,
            "cartridge_logits": cartridge_logits,
            "step_logits": step_logits,
            "solvent_cond_logits": solvent_cond_logits,
            "solvent_logits": solvent_logits,
        }

        if self.predict_cardinality:
            if self.count_mode == "classification":
                car_cnt_logits = self.cartridge_count_head(h_car)
                sol_cnt_logits = self.solvent_count_head(h_sol).view(
                    -1, self.n_steps, self.n_sol_count_bins
                )
                out["cartridge_count_logits"] = car_cnt_logits
                out["solvent_count_logits"] = sol_cnt_logits
                out["cartridge_count"] = self._counts_from_logits(car_cnt_logits)
                out["solvent_count"] = self._counts_from_logits(sol_cnt_logits)
            else:
                # softplus 保证个数非负；训练时对 log1p(真实个数) 回归，见 spe/losses.py
                out["cartridge_count"] = nn.functional.softplus(
                    self.cartridge_count_head(h_car)
                ).squeeze(-1)
                out["solvent_count"] = nn.functional.softplus(self.solvent_count_head(h_sol))

        if self.base_solvent_head is not None:
            out["base_solvent_logits"] = self.base_solvent_head(h_sol).view(
                -1, self.n_steps, self.n_base_solvent
            )

        return out

    def ratio_bucket_logits(
        self, h: torch.Tensor, step_ids: torch.Tensor, solvent_ids: torch.Tensor
    ) -> torch.Tensor:
        """ratio 分桶头的 logits，(P, n_ratio_bins)。"""
        if self.ratio_bucket_mlp is None:
            raise RuntimeError("当前 ratio_mode 不是 bucket，没有分桶头")
        z = torch.cat(
            [self.ratio_tower(h), self.step_emb(step_ids), self.solvent_emb(solvent_ids)], dim=-1
        )
        return self.ratio_bucket_mlp(z)

    def ratio_pred(
        self, h: torch.Tensor, step_ids: torch.Tensor, solvent_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        [0, 1] 区间的 ratio 预测。分桶模式下取众数桶的中心值。

        取众数而非期望的理由同 `_counts_from_logits`：ratio 真实值集中在少数常见配比上，
        期望会把它们平均成一个中间值，正好落在两个众数之间、两边都不中。
        """
        if self.ratio_mode != "bucket":
            return super().ratio_pred(h, step_ids, solvent_ids)
        logits = self.ratio_bucket_logits(h, step_ids, solvent_ids)
        idx = torch.argmax(logits, dim=-1).to(logits.dtype)
        return (idx + 0.5) / float(self.n_ratio_bins)


def build_model(
    input_dim: int,
    n_cartridge: int,
    n_steps: int,
    n_solvent: int,
    hidden_dim: int = 512,
    enc_layers: int = 2,
    dropout: float = 0.2,
    n_base_solvent: int = 0,
    predict_cardinality: bool = True,
    count_mode: str = "regression",
    n_car_count_bins: int = 16,
    n_sol_count_bins: int = 12,
    count_reduce: str = "argmax",
    numeric_embed: str = "none",
    numeric_embed_bins: int = 24,
    numeric_embed_dim: int = 16,
    numeric_embed_sigma: float = 0.05,
    x_train: Optional[np.ndarray] = None,
    ratio_mode: str = "regression",
    n_ratio_bins: int = 20,
    uncertainty_weighting: bool = False,
) -> SPEModelV6:
    """按各开关组合出模型。全部开关取默认值时结构等价于 `SPEBaselineModel`。"""
    return SPEModelV6(
        input_dim=input_dim,
        n_cartridge=n_cartridge,
        n_steps=n_steps,
        n_solvent=n_solvent,
        hidden_dim=hidden_dim,
        enc_layers=enc_layers,
        dropout=dropout,
        n_base_solvent=n_base_solvent,
        predict_cardinality=predict_cardinality,
        count_mode=count_mode,
        n_car_count_bins=n_car_count_bins,
        n_sol_count_bins=n_sol_count_bins,
        count_reduce=count_reduce,
        numeric_embed=numeric_embed,
        numeric_embed_bins=numeric_embed_bins,
        numeric_embed_dim=numeric_embed_dim,
        numeric_embed_sigma=numeric_embed_sigma,
        x_train=x_train,
        ratio_mode=ratio_mode,
        n_ratio_bins=n_ratio_bins,
        uncertainty_weighting=uncertainty_weighting,
    )
