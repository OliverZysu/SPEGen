"""
编码器超参的随机搜索。

    python -m spe.tools.hp_search --n_trials 20 --shard 0 --n_shards 2
    python -m spe.tools.hp_search --list          # 只打印将要跑的配置，不训练

动机：树基线是调过参的，而我们的 MLP 一直用默认值（hidden 256 / 3 层 / dropout 0.2 /
lr 1e-3），从来没搜过。这是一处方法学上实在的不对等，补上它既可能涨点，
也让「MLP 打不过树」这个结论更站得住。

搜索一律按**验证集**选模选配置，测试集只在最后确认阶段看一次。
`--shard` 用来把 trial 分到多条并行流上，各流互不重叠。
"""

import argparse
import itertools
import json
import os
import subprocess
from typing import Any, Dict, List

import numpy as np

SPACE: Dict[str, List[Any]] = {
    "hidden_dim": [128, 256, 384, 512],
    "enc_layers": [2, 3, 4],
    "dropout": [0.1, 0.15, 0.2, 0.3],
    "lr": [3e-4, 5e-4, 1e-3, 2e-3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [64, 128, 256],
}

# 上一轮的最优基底：溶剂 ASL + ratio 分桶 + cartridge 排序权重 2.0 + 溶剂任务权重 0.1
BASE_FLAGS = [
    "--no_functional_groups", "--no_decompose_solvent", "--no_cardinality",
    "--loss_solvent", "asl", "--ratio_mode", "bucket",
    "--lambda_rank_cartridge", "2.0", "--lambda_solvent", "0.1",
    "--decode_objective", "f1",
]


def sample_trials(n_trials: int, seed: int) -> List[Dict[str, Any]]:
    """随机采样，并去重。默认配置固定放在第 0 号作为对照。"""
    rng = np.random.default_rng(seed)
    default = {"hidden_dim": 256, "enc_layers": 3, "dropout": 0.2,
               "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 128}
    trials: List[Dict[str, Any]] = [default]
    seen = {tuple(sorted(default.items()))}
    guard = 0
    while len(trials) < n_trials and guard < 500 * n_trials:
        guard += 1
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        trials.append(cfg)
    return trials


def trial_name(idx: int, cfg: Dict[str, Any]) -> str:
    return (f"t{idx:02d}_h{cfg['hidden_dim']}_l{cfg['enc_layers']}"
            f"_d{cfg['dropout']}_lr{cfg['lr']:g}_wd{cfg['weight_decay']:g}"
            f"_bs{cfg['batch_size']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default="output/improve21/hp")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--sample_seed", type=int, default=7)
    p.add_argument("--seed", type=int, default=42, help="数据划分种子，必须固定 42")
    p.add_argument("--init_seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--feature_transform", default="rank_gauss")
    p.add_argument("--rank_loss", default="set_ce")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shards", type=int, default=1)
    p.add_argument("--only", default="",
                   help="只跑指定编号的 trial，逗号分隔。确认阶段用它复跑按验证分选出的前几名")
    p.add_argument("--list", action="store_true", help="只打印配置，不训练")
    args = p.parse_args()

    trials = sample_trials(args.n_trials, args.sample_seed)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    if args.list:
        for i, cfg in enumerate(trials):
            print(f"{i:02d}  {json.dumps(cfg, sort_keys=True)}")
        return

    wanted = None
    if args.only.strip():
        wanted = {int(s) for s in args.only.split(",") if s.strip()}
    mine = [
        (i, c) for i, c in enumerate(trials)
        if (wanted is None or i in wanted) and i % args.n_shards == args.shard
    ]
    print(f"本分片负责 {len(mine)} / {len(trials)} 个 trial", flush=True)

    for idx, cfg in mine:
        name = trial_name(idx, cfg)
        run_dir = os.path.join(args.out_dir, f"{name}_init{args.init_seed}")
        if os.path.isfile(os.path.join(run_dir, "test_metrics.json")):
            print(f"[SKIP] {name}", flush=True)
            continue
        cmd = [
            "python3", "-m", "spe.train",
            "--output_dir", run_dir,
            "--seed", str(args.seed), "--init_seed", str(args.init_seed),
            "--epochs", str(args.epochs),
            "--feature_transform", args.feature_transform,
            "--rank_loss", args.rank_loss,
            *BASE_FLAGS,
            "--hidden_dim", str(cfg["hidden_dim"]),
            "--enc_layers", str(cfg["enc_layers"]),
            "--dropout", str(cfg["dropout"]),
            "--lr", f"{cfg['lr']:g}",
            "--weight_decay", f"{cfg['weight_decay']:g}",
            "--batch_size", str(cfg["batch_size"]),
        ]
        log_path = os.path.join(args.out_dir, "logs", f"{name}.log")
        print(f"[RUN ] {name}", flush=True)
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"[FAIL] {name}，日志见 {log_path}", flush=True)

    print("本分片完成", flush=True)


if __name__ == "__main__":
    main()
