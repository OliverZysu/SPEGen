# SPE 方法关键参数生成（Baseline Project）

ICLR 论文 **SPEGen** 的训练、评测与出图入口在 `spe/` 与 `figures/`，LaTeX 源在 `overleaf/`。
未进入论文的旧代码和实验产物在 `backup/`。论文模型由 `python -m spe.train` 训练，不是下面的 `main.py`（`main.py` / `model.py` 仍保留，因为 `spe.train` 会引用其中的损失与主干）。

> 输入：污染物的分子描述符（21 维数值特征）  
> 输出：SPE 方法的关键可复现信息：**Cartridge、步骤 mask、每步溶剂集合、每个（步骤×溶剂）的 ratio（浓度比例）**。

根目录的 `main.py` 是早期多任务基线实现：

1. **多任务学习**：Cartridge / Step mask / Solvent（每步）是多标签分类；ratio 是回归。
2. **条件式 ratio 回归**：ratio 的预测以 `(step_id, solvent_id)` 为条件（嵌入后与分子表征拼接）。
3. **图结构直觉（within-batch smoothness）**：在一个 batch 内，用分子描述符的余弦相似度连 top-k 邻居，对编码表征加平滑正则。
4. **可生成方案**：测试时不仅给指标，还为每个污染物输出一套可读的“预测方案”，并对照其真实方法（所有 CasMp）。

---

## 1. 数据文件与字段含义

数据默认放在 `data/`：

### 1.1 `processed_molecular_data.csv`

- `CID`：污染物唯一标识（不重复）。
- `CasMp`：该污染物对应的所有处理方法 ID（list string），例如 `['1-143-CAS-432882', '1-116-CAS-433254']`。
- `Complexity_value` ~ `LogP_value`：21 维分子描述符（模型输入）。
- `Cartridge_Class_*`：Cartridge 多标签（**已聚合到污染物级别**：某类 Cartridge 只要在该污染物任意一个方法中出现，即为 1）。

### 1.2 `spe_solvent_ratio.csv`

该文件提供 **方法级** 的步骤溶剂与 ratio 信息。

- `CasMp`：方法 ID（同上）。
- 5 个步骤列：`Sample loading / Condition / Wash / Elute / Reconstitute`。
- 每个步骤单元格示例：
  - `"(water, null)"`：water，ratio 未知。
  - `"(methanol, null); (water, null)"`：同一步骤存在多个溶剂操作。
  - `"(acetone+water, 0.9)"`：溶剂为 acetone+water，ratio=0.9。

**注意**：本项目把 `ratio` 当作连续回归目标（0~1），`null` 表示未知，不参与 ratio loss/metric。

---

## 2. 标签构建（污染物级聚合）

一个污染物可能对应多个方法（CasMp）。训练时我们在**污染物级别**聚合标签：

### 2.1 Step mask（多标签）

对 5 个步骤取并集：只要任意方法出现该步骤，就置 1。

### 2.2 Solvent（多标签，按步骤）

对每一步的溶剂集合取并集：

- 输出张量形状：`(5, S)`（S 为溶剂词表大小）。
- 只要任意方法在该步骤用了该溶剂，就置 1。

### 2.3 Ratio（回归，按 step×solvent 条件）

ratio 只对 **存在数值标注** 的 (step, solvent) 监督：

- 若某 (step, solvent) 在多个方法中出现多个 ratio 数值，默认使用 **mean** 聚合得到污染物级标签。
- 若全部为 `null`，则该 pair 不产生 ratio label（训练/评估跳过）。

最终我们得到每个污染物的 `ratio_pairs`：

```text
ratio_pairs = [(step_id, solvent_id, ratio_value), ...]
```

---

## 3. 模型结构（方案一：强基线）

代码位置：`model.py`。

### 3.1 主干编码器

对输入分子描述符 `x ∈ R^{21}` 使用 MLP 编码：

```text
h = MLP(x)  # (B, H)
```

### 3.2 多任务输出头

1) **Cartridge head（多标签分类）**

```text
cartridge_logits = Linear(h)  # (B, C)
```

2) **Step head（多标签分类）**

```text
step_logits = Linear(h)  # (B, 5)
```

3) **Solvent head（多标签分类，按 step）**

```text
solvent_logits = Linear(h).reshape(B, 5, S)
```

4) **Ratio head（回归，条件式）**

ratio 的预测以 `(step_id, solvent_id)` 为条件：

```text
z = concat(h, Emb(step_id), Emb(solvent_id))
ratio = sigmoid(MLP_ratio(z))  # (P,) in [0,1]
```

其中 `P` 是 batch 内所有有 ratio label 的 (样本, step, solvent) pair 数量。

### 3.3 within-batch 图平滑正则

在一个 batch 内，用分子描述符余弦相似度建立 top-k 邻接，并约束编码表征 `h` 在相似样本间平滑：

```text
L_smooth = Σ_{i~j} w_ij ||h_i - h_j||^2
```

---

## 4. 训练与损失

代码位置：`main.py`。

### 4.1 损失函数

总体损失：

```text
L = λ_car * L_cartridge
  + λ_step * L_step
  + λ_solvent * L_solvent
  + λ_ratio * L_ratio
  + λ_smooth * L_smooth
```

1) `L_cartridge`：BCEWithLogits（带 pos_weight）

2) `L_step`：BCEWithLogits（带 pos_weight），并用 `has_spe_info` mask：

- 只有能在 `spe_solvent_ratio.csv` 找到至少一个方法的污染物（has_spe_info=1）才参与该项。

3) `L_solvent`：同上，mask。

4) `L_ratio`：MSELoss，仅在 `ratio_pairs` 中有 label 的 pair 上计算。

### 4.2 特征标准化

仅对 21 维分子描述符做标准化（用 train split 的均值方差）。

---

## 5. 测试输出与文件说明

默认输出目录：`output/`。

- `train_history.json`：每 epoch 的 train/val loss（含各子任务 loss）。
- `best_model.pt`：最佳验证集 loss 的模型 checkpoint（包含 mean/std 与 artifacts）。
- `test_metrics.json`：测试集指标。
- `test_predictions.jsonl`：每个测试污染物的预测方案 + 对应真实方法对照。
- `test_report.md`：可读的 Markdown 报告。

---

## 6. 指标解释（test_metrics.json）

### 6.1 多标签分类指标（Cartridge / Step / Solvent）

设所有样本、所有标签维度展平后：

- **TP**：预测为 1 且真实为 1 的数量
- **FP**：预测为 1 但真实为 0 的数量
- **FN**：预测为 0 但真实为 1 的数量

则：

- `micro_precision = TP / (TP + FP)`
- `micro_recall = TP / (TP + FN)`
- `micro_f1 = 2 * precision * recall / (precision + recall)`

样本级指标：

- `exact_match_rate`：预测集合与真实集合**完全一致**的样本比例。
- `hit_rate`：对每个样本，如果预测集合与真实集合**有交集**（且真实集合非空）则记 1，否则 0；最后取平均。
- `jaccard`：对每个样本计算 `|P∩T| / |P∪T|`，对 union 非空的样本求平均。

### 6.2 Ratio 回归指标

在所有有数值标注的 (step, solvent) pair 上计算：

- `mae`：平均绝对误差 `mean(|ŷ - y|)`
- `mse`：平均平方误差 `mean((ŷ - y)^2)`
- `rmse`：均方根误差 `sqrt(mse)`
- `r2`：决定系数（越接近 1 越好）
- `within_0.05`：`|ŷ-y|<=0.05` 的比例
- `within_0.10`：`|ŷ-y|<=0.10` 的比例
- `n` / `n_pairs_eval`：评估的 pair 数量

---

## 7. 运行方式

### 7.1 训练深度模型（强基线）

```bash
python main.py --data_dir data --output_dir output
```

常用可调参数：

- `--epochs` `--batch_size` `--lr`
- `--lambda_*`：各子任务 loss 权重
- `--smooth_topk` `--lambda_smooth`
- `--thr_*`：多标签分类阈值（Cartridge/Step/Solvent）

### 7.2 传统机器学习 baseline

代码位置：`baseline.py`。

```bash
# 决策树预测 Cartridge
python baseline.py DecisionTree Cartridge

# 随机森林预测 Step mask
python baseline.py RandomForest Step

# 线性 SVM 预测 Solvent
python baseline.py SVM Solvent

# 随机森林回归 ratio
python baseline.py RandomForest Ratio
```

输出默认写入：`output/baseline_output/<Model>_<Task>/`。

---

## 8. 代码结构

```text
.
├── main.py                # 深度模型训练/评估/报告生成
├── model.py               # 模型结构（多任务 + 条件式 ratio 回归 + 图平滑）
├── baseline.py            # 传统 ML baselines：DecisionTree / RandomForest / SVM
├── utils/
│   ├── dataloader.py      # 数据读取、聚合标签、DataLoader
│   ├── parsing.py         # spe_solvent_ratio.csv 单元格解析 + 标签聚合逻辑
│   ├── metrics.py         # 多标签指标 + ratio 回归指标
│   ├── io.py              # JSON/JSONL 输出
│   └── seed.py            # 随机种子
└── data/
    ├── processed_molecular_data.csv
    └── spe_solvent_ratio.csv
```
