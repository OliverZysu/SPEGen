"""
SPE 方案生成的改进模块（v6 系列）。

本包完全是**增量的**：不修改 `main.py` / `model.py` / `utils/` 中的任何既有代码，
因此 v16 / v21 / v22 等历史实验仍可原样复现。

四条改进主线，各自独立可开关：

1. `metrics_strict` —— 修正 Hit@1 / StrictAcc 的口径泄漏（无 ratio 标注免检、
   无流程信息样本免检），提供与旧指标并列输出的干净版本。
2. `cardinality` + `decode` —— 预测集合大小并做基数感知解码，压掉 1.8 倍的过预测，
   直接改善 exact-match / StrictAcc / SSS。
3. `features` —— 把官能团特征接进模型输入（原本只有 21 维数值描述符，接入后 96 维）。
4. `solvent_vocab` —— 把 378 类溶剂组合拆成 81 个基础溶剂做两级预测，缓解标签稀疏。

训练入口是 `spe/train.py`，评估与调优逻辑在 `spe/decode.py`。
"""

__all__ = [
    "cardinality",
    "data",
    "decode",
    "features",
    "losses",
    "metrics_strict",
    "model",
    "solvent_vocab",
]
