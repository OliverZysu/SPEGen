# spe/ — ICLR 论文实验入口

论文中的 SPEGen 在 21 维分子描述符上训练（官能团特征与溶剂拆解均关闭）。
从仓库根目录运行。

## 训练

```bash
python -m spe.train --output_dir output/improve21/my_run \
  --no_functional_groups --no_decompose_solvent \
  --feature_transform rank_gauss --epochs 60
```

批量实验脚本：

- `bash spe/run_screen_21d.sh` — 第一轮 21 维筛选
- `STAGE=1 bash spe/run_improve_21d.sh` — 输入变换 / 后续改进

## 基线（与模型同输入、同划分）

```bash
bash baseline_scripts.sh
python -m spe.tools.recompute_metrics --baseline_dir output/baseline_output
```

## 从已有 run 生成论文表

```bash
python -m spe.tools.make_paper_tables --out_dir output/paper_tables
```

旧的 v6（96 维官能团）实验说明与脚本在 `backup/spe/`。
