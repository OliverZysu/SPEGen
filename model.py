import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MLPEncoder(nn.Module):
    """Simple MLP encoder for molecular descriptors."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        assert num_layers >= 1
        act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        layers = []
        d = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            d = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskTower(nn.Module):
    """Task-specific MLP tower to reduce multi-task interference."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SPEBaselineModel(nn.Module):
    """
    Baseline multi-task model:

    Input: molecular descriptors x (B, D)
    Outputs:
      - cartridge_logits: (B, C_cartridge) multi-label
      - step_logits:      (B, 5) multi-label existence mask
      - solvent_logits:   (B, 5, S_solvent) multi-label per step
      - ratio head is conditional (regression):
          ratio = sigmoid(f([h, step_emb, solvent_emb])) -> (P,)
        computed only for queried (step, solvent) pairs.
    """
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
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_solvent = n_solvent

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=enc_layers,
            dropout=dropout,
            activation=activation,
        )

        # Task-specific towers (shared backbone + task heads)
        self.cartridge_tower = TaskTower(hidden_dim, dropout=dropout)
        self.step_tower = TaskTower(hidden_dim, dropout=dropout)
        self.solvent_tower = TaskTower(hidden_dim, dropout=dropout)
        self.ratio_tower = TaskTower(hidden_dim, dropout=dropout)

        self.cartridge_head = nn.Linear(hidden_dim, n_cartridge)
        self.step_head = nn.Linear(hidden_dim, n_steps)

        # Solvent decoder (ablation): classic linear projection from solvent tower.
        self.solvent_head = nn.Linear(hidden_dim, n_steps * n_solvent)
        # Step-guided solvent gating strength (single global coefficient).
        self.solvent_step_gate_alpha = nn.Parameter(torch.tensor(1.0))

        # conditional ratio head (regression)
        self.step_emb = nn.Embedding(n_steps, step_emb_dim)
        self.solvent_emb = nn.Embedding(n_solvent, solvent_emb_dim)

        self.ratio_mlp = nn.Sequential(
            nn.Linear(hidden_dim + step_emb_dim + solvent_emb_dim, conc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conc_hidden_dim, conc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conc_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns logits for cartridge/steps/solvent plus the latent embedding h.
        """
        h = self.encoder(x)

        h_car = self.cartridge_tower(h)
        h_step = self.step_tower(h)
        h_sol = self.solvent_tower(h)

        cartridge_logits = self.cartridge_head(h_car)
        step_logits = self.step_head(h_step)
        solvent_cond_logits = self.solvent_head(h_sol).view(-1, self.n_steps, self.n_solvent)
        # Step-guided gating in logit space:
        # if step is unlikely, solvent logits for that step are globally suppressed.
        solvent_logits = solvent_cond_logits + self.solvent_step_gate_alpha * step_logits.unsqueeze(-1)
        return {
            "h": h,
            "cartridge_logits": cartridge_logits,
            "step_logits": step_logits,
            "solvent_cond_logits": solvent_cond_logits,
            "solvent_logits": solvent_logits,
        }

    @torch.no_grad()
    def predict_ratio(
        self,
        h: torch.Tensor,
        step_ids: torch.Tensor,
        solvent_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience wrapper for inference.
        h: (P, H)
        step_ids: (P,)
        solvent_ids: (P,)
        returns ratio prediction: (P,)
        """
        return self.ratio_pred(h, step_ids, solvent_ids)

    def ratio_logits(
        self,
        h: torch.Tensor,
        step_ids: torch.Tensor,
        solvent_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute raw (unbounded) ratio logits for queried (step, solvent) pairs.
        h: (P, H)
        step_ids: (P,)
        solvent_ids: (P,)
        returns raw logits: (P, 1)
        """
        h_ratio = self.ratio_tower(h)
        se = self.step_emb(step_ids)
        ve = self.solvent_emb(solvent_ids)
        z = torch.cat([h_ratio, se, ve], dim=-1)
        return self.ratio_mlp(z)

    def ratio_pred(
        self,
        h: torch.Tensor,
        step_ids: torch.Tensor,
        solvent_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Bounded ratio prediction in [0, 1]."""
        raw = self.ratio_logits(h, step_ids, solvent_ids)
        return torch.sigmoid(raw).squeeze(-1)


def batch_graph_smoothness(
    x: torch.Tensor,
    h: torch.Tensor,
    topk: int = 8,
    clamp_min: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Within-batch graph smoothness regularizer:
      - compute cosine similarity on input descriptors x
      - connect each sample to its top-k neighbors (excluding self)
      - penalize weighted squared distance between embeddings h

    Returns a scalar tensor.
    """
    # Normalize x for cosine similarity
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    sim = x_norm @ x_norm.t()  # (B, B)
    b = sim.size(0)
    sim.fill_diagonal_(0.0)
    # top-k neighbors per row
    k = min(topk, max(1, b - 1))
    vals, idx = torch.topk(sim, k=k, dim=1)
    w = torch.clamp(vals, min=clamp_min)  # (B, k)
    # gather neighbor embeddings
    h_i = h.unsqueeze(1).expand(-1, k, -1)           # (B, k, H)
    h_j = h[idx]                                     # (B, k, H)
    dist2 = (h_i - h_j).pow(2).sum(dim=-1)          # (B, k)
    reg = (w * dist2).sum() / (b * k + eps)
    return reg
