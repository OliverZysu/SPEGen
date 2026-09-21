"""
生成论文用的结果表格（LaTeX + Markdown + CSV）。

直接从实验输出目录读数，不写死任何数值，改了实验重跑即可刷新表格：

    python -m spe.tools.make_paper_tables --out_dir output/paper_tables

产出：
    table1_per_task.{tex,md}      各子任务指标
    table2_end_to_end.{tex,md}    端到端指标
    table3_ablation.{tex,md}      消融实验
    table4_repeats.{tex,md}       重复实验的均值/标准差（噪声底）
    all_metrics.csv               上述全部数值的长表，便于自己再排版

约定
----
输入统一 21 维分子描述符，与传统基线完全同输入。
我们的模型报 4 次重复（同一数据划分、不同初始化）的均值 ± 标准差；
传统基线是确定性训练，报单次结果。
加粗规则：该列/该行的最优值。差距小于我们那一列 2 个标准差的不算胜负。
"""

import argparse
import csv
import glob as globlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 指标定义：(键, 论文里的列名, LaTeX 列名, 小数位, 是否越小越好)
# ----------------------------------------------------------------------

Metric = Tuple[str, str, str, int, bool]

PER_TASK: List[Metric] = [
    ("car_f1", "Cartridge F1", r"Cart. F1", 4, False),
    ("car_em", "Cartridge EM", r"Cart. EM", 4, False),
    ("sol_f1", "Solvent F1", r"Solv. F1", 4, False),
    ("ratio_mae", "Ratio MAE", r"Ratio MAE", 4, True),
    ("ratio_w10", "Ratio@0.10", r"Ratio@0.10", 4, False),
]

# StrictAcc 不列入评测：它要求整套方案（全部 cartridge、全部步骤、全部溶剂、全部配比）
# 逐项精确匹配，在 378 个溶剂组合与连续配比下所有方法都低于 5%（最好的基线 0.045），
# 缺乏区分度。相关数值仍会照常计算并存在 test_metrics.json 里，只是不进论文表格。
END_TO_END: List[Metric] = [
    ("car_top1", "Cartridge top-1", r"Cart. top-1", 4, False),
    ("chain", "Chain Acc", r"Chain Acc", 4, False),
    ("hit1", "Hit@1", r"Hit@1", 4, False),
    ("sss", "SSS", r"SSS", 4, False),
    ("hc", "HC", r"HC", 4, False),
    ("gus", "GUS", r"GUS", 2, False),
]

ALL_METRICS = PER_TASK + END_TO_END

# 从 run 目录的 test_metrics.json 里取值的路径
OURS_PATHS: Dict[str, Tuple[str, ...]] = {
    "car_f1": ("cartridge", "micro_f1"),
    "car_em": ("cartridge", "exact_match_rate"),
    "sol_f1": ("solvent", "micro_f1"),
    "ratio_mae": ("ratio", "mae"),
    "ratio_w10": ("ratio", "within_0.10"),
    "car_top1": ("overall_clean", "cartridge_top1_acc"),
    "chain": ("overall_clean", "chain_acc_no_ratio"),
    "hit1": ("overall_clean", "hit_at_1_info_only"),
    "sss": ("overall_clean", "scheme_similarity_info_only"),
    "hc": ("overall_clean", "hierarchical_consistency_info_only"),
    "gus": ("overall", "global_utility_score"),
}

BASELINES = ["DecisionTree", "RandomForest", "SVM", "MLP", "XGBoost", "LightGBM"]

# 我们的模型：论文里报的三个工作点。都在同一数据划分上跑 4 个初始化，
# 且都已包含第二轮的两项有效改进（rank-gauss 输入变换 + 搜出来的编码器超参）。
HP = "hidden 512 / 2 层 / dropout 0.3 / lr 1e-3 / bs 64"
OURS_VARIANTS: List[Tuple[str, str, List[str]]] = [
    (
        "Ours (balanced)",
        f"rank-gauss + {HP} + 溶剂非对称损失 + ratio 分桶 + 排序权重 0.5",
        ["output/improve21/s5_C2t08_init*"],
    ),
    (
        "Ours (chain-oriented)",
        "同上，排序权重 2.0、溶剂任务权重 0.1",
        ["output/improve21/hp/t08_h512_l2_d0.3_lr0.001_wd0_bs64_init*"],
    ),
    (
        "Ours (exact-match)",
        "同上，加集合大小分类 + 分位数搜索 + em\\_jaccard 解码",
        ["output/improve21/s8_EMt08_init*"],
    ),
]

# 集成行：集成本身是一个确定的模型，不报 ±标准差。
# 键是 output/improve21 下的 json 文件名。
OURS_ENSEMBLES: List[Tuple[str, str, str]] = [
    (
        "Ours (bagging ensemble)",
        "链路型工作点，8 个成员，各自对训练集有放回重采样",
        "ens_bagt08",
    ),
]

# 消融表：(run 目录, 论文里的配置名, 一句话说明)
ABLATION: List[Tuple[str, str, str]] = [
    ("R0_control", "Base", "共享编码器 + 逐标签阈值调优"),
    ("R1_step2", "+ cardinality decode", "集合大小回归 + EM/Jaccard 解码目标"),
    ("R2_count_cls", "+ size classification", "集合大小改分类，取众数"),
    ("R3_count_cls_exp", "+ size cls. (expectation)", "集合大小分类，取期望"),
    ("R13_count_q", "+ size cls. (quantile)", "集合大小分类 + 验证集搜分位数"),
    ("R5_ple", "+ PLE embedding", "分段线性数值嵌入"),
    ("R6_periodic", "+ periodic embedding", "正弦数值嵌入"),
    ("R7_asl_sol", "+ ASL (solvent)", "溶剂非对称损失"),
    ("R8_ratio_bucket", "+ ratio bucketing", "ratio 回归改分桶分类"),
    ("R9_uncertainty", "+ uncertainty weighting", "可学习 log 方差任务加权"),
    ("R11_sol_w01", "+ solvent weight 0.1", "降低溶剂任务权重"),
    ("R12_rank_car_up", "+ rank loss 0.5", "加大 cartridge 排序损失"),
    ("C6_rank10", "+ rank loss 1.0", ""),
    ("C9_rank20", "+ rank loss 2.0", ""),
    ("C1_r78", "ASL + bucketing", "两个无冲突增益叠加"),
    ("C2_r78_rank", "  + rank 0.5", "= Ours (balanced)"),
    ("C8_r78_rank10", "  + rank 1.0", ""),
    ("C15_r78_rank20", "  + rank 2.0", ""),
    ("C10_r78_rank10_solw", "  + rank 1.0 + solv. 0.1", ""),
    ("C12_r78_rank20_solw", "  + rank 2.0 + solv. 0.1", "= Ours (chain-oriented)"),
    ("C13_r78_rank30_solw", "  + rank 3.0 + solv. 0.1", ""),
    ("C4_r78_step2", "  + cardinality decode", ""),
    ("C5_r78_cls_q", "  + size cls. + quantile", ""),
    ("C7_full", "  + size cls. + rank 0.5", "= Ours (exact-match)"),
]

# 第二轮改进的消融：(标签, 目录 glob, 说明)。全部报 4 次重复的均值±标准差。
ABLATION2: List[Tuple[str, List[str], str]] = [
    ("Base (z-score, 默认超参)",
     ["output/improve21/s1_C12_zscore_init*"], "上一轮的链路型工作点"),
    ("+ rank-gauss 输入变换",
     ["output/improve21/s1_C12_rank_gauss_init*"], "描述符偏度最高 53.8，z-score 不改变形状"),
    ("+ 搜索得到的编码器超参",
     ["output/improve21/hp/t08_h512_l2_d0.3_lr0.001_wd0_bs64_init*"],
     "hidden 512 / 2 层 / dropout 0.3 / lr 1e-3 / bs 64"),
    ("+ bootstrap bagging（单成员）",
     ["output/improve21/s6_bagt08_init*"], "每个成员只看到约 63% 的训练样本"),
]

# 无效尝试：报出来比藏起来有价值。(标签, 目录 glob, 结论)
NEGATIVE: List[Tuple[str, List[str], str]] = [
    ("有符号 log1p 变换", ["output/improve21/s1_C12_log1p_init*"], "弱于 rank-gauss"),
    ("中位数/IQR 变换", ["output/improve21/s1_C12_robust_init*"], "劣于 z-score"),
    ("排序损失改 any-positive", ["output/improve21/s3_L1_any_pos_init42"], "top-1 持平，其余全面下降"),
    ("排序损失改 RankNet", ["output/improve21/s3_L2_ranknet_init42"], "top-1 大幅下降"),
    ("排序损失改 top-1 铰链", ["output/improve21/s3_L3_margin_init42"], "全面崩坏"),
]

# 重复实验（噪声底）：(标签, 目录 glob 列表)
REPEAT_GROUPS: List[Tuple[str, List[str]]] = [
    ("Base", ["output/screen21/R0_control_seed42", "output/noise21/R0_control_init*"]),
    ("+ cardinality decode", ["output/screen21/R1_step2_seed42", "output/noise21/R1_step2_init*"]),
    ("Ours (balanced)", ["output/screen21/C2_r78_rank_seed42", "output/ens21/C2_init*"]),
    ("Ours (chain-oriented)", ["output/screen21/C12_r78_rank20_solw_seed42", "output/ens21/C12_init*"]),
    ("Ours (exact-match)", ["output/screen21/C7_full_seed42", "output/ens21/C7_init*"]),
]


def _dig(obj, path: Sequence[str]) -> Optional[float]:
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _expand(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        if any(ch in pat for ch in "*?["):
            out.extend(sorted(globlib.glob(pat)))
        else:
            out.append(pat)
    return [d for d in out if os.path.isfile(os.path.join(d, "test_metrics.json"))]


def read_run(run_dir: str) -> Dict[str, float]:
    with open(os.path.join(run_dir, "test_metrics.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload.get("metrics", {})
    row = {k: _dig(metrics, p) for k, p in OURS_PATHS.items()}
    row["val_score"] = payload.get("val_select_score_tuned")
    return row


def read_ensemble(stem: str) -> Dict[str, Tuple[float, Optional[float]]]:
    """读一个集成结果。集成是确定的单个模型，所以标准差位置留空。"""
    with open(f"output/improve21/{stem}.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    metrics = payload.get("metrics", payload)
    out: Dict[str, Tuple[float, Optional[float]]] = {}
    for key, path in OURS_PATHS.items():
        value = _dig(metrics, path)
        if value is not None:
            out[key] = (value, None)
    return out


def read_runs(patterns: Sequence[str]) -> Tuple[Dict[str, Tuple[float, float]], int]:
    """返回 {指标: (均值, 标准差)} 与重复次数。"""
    dirs = _expand(patterns)
    if not dirs:
        raise FileNotFoundError(f"没有找到任何 run：{patterns}")
    rows = [read_run(d) for d in dirs]
    agg: Dict[str, Tuple[float, float]] = {}
    for key in list(OURS_PATHS) + ["val_score"]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            agg[key] = (float(np.mean(vals)), float(np.std(vals)))
    return agg, len(dirs)


def read_baselines() -> Dict[str, Dict[str, float]]:
    base = "output/baseline_output"
    clean_path = os.path.join(base, "overall_metrics_clean.json")
    with open(clean_path, encoding="utf-8") as fh:
        clean_all = json.load(fh)

    per_task_src = {
        "Cartridge": [("car_f1", "micro_f1"), ("car_em", "exact_match_rate")],
        "Solvent": [("sol_f1", "micro_f1")],
        "Ratio": [("ratio_mae", "mae"), ("ratio_w10", "within_0.10")],
    }
    clean_src = {
        "car_top1": "cartridge_top1_acc",
        "chain": "chain_acc_no_ratio",
        "hit1": "hit_at_1_info_only",
        "sss": "scheme_similarity_info_only",
        "hc": "hierarchical_consistency_info_only",
    }

    out: Dict[str, Dict[str, float]] = {}
    for name in BASELINES:
        row: Dict[str, float] = {}
        for task, keys in per_task_src.items():
            path = os.path.join(base, f"{name}_{task}", "metrics.json")
            with open(path, encoding="utf-8") as fh:
                m = json.load(fh)["metrics"]
            for our_key, their_key in keys:
                row[our_key] = float(m[their_key])
        entry = clean_all[name]
        for our_key, their_key in clean_src.items():
            row[our_key] = float(entry["clean"][their_key])
        row["gus"] = float(entry["legacy"]["global_utility_score"])
        out[name] = row
    return out


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------


def fmt(value: Optional[float], digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def best_index(values: Sequence[Optional[float]], lower_better: bool) -> Optional[int]:
    valid = [(v, i) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    return (min(valid) if lower_better else max(valid))[1]


def render_tex(
    caption: str,
    label: str,
    col_names: Sequence[str],
    row_names: Sequence[str],
    cells: Sequence[Sequence[str]],
    rule_after: Sequence[int] = (),
    note: str = "",
) -> str:
    align = "l" + "r" * len(col_names)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        "Model & " + " & ".join(col_names) + r" \\",
        r"\midrule",
    ]
    for i, (name, row) in enumerate(zip(row_names, cells)):
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
        if i in rule_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(rf"\vspace{{2pt}}{{\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def render_md(
    title: str,
    col_names: Sequence[str],
    row_names: Sequence[str],
    cells: Sequence[Sequence[str]],
    note: str = "",
) -> str:
    lines = [f"### {title}", ""]
    lines.append("| Model | " + " | ".join(col_names) + " |")
    lines.append("|" + "---|" * (len(col_names) + 1))
    for name, row in zip(row_names, cells):
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    if note:
        lines += ["", note]
    return "\n".join(lines) + "\n"


def strip_tex(text: str) -> str:
    return text.replace(r"\textbf{", "").replace(r"\_", "_").replace("}", "")


def strip_tex_note(text: str) -> str:
    """表注在 Markdown 里去掉 LaTeX 转义与数学模式。"""
    return (
        text.replace(r"\ ", " ")
        .replace(r"\_", "_")
        .replace("$n=1145$", "n=1145")
        .replace("$", "")
    )


def build_main_table(
    metrics: Sequence[Metric],
    bl: Dict[str, Dict[str, float]],
    ours: List[Tuple[str, Dict[str, Tuple[float, float]], int]],
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    row_names = list(BASELINES) + [name for name, _, _ in ours]
    col_names = [tex for _, _, tex, _, _ in metrics]

    tex_cells: List[List[str]] = [[] for _ in row_names]
    md_cells: List[List[str]] = [[] for _ in row_names]

    for key, _disp, _tex, digits, lower in metrics:
        column: List[Optional[float]] = [bl[b].get(key) for b in BASELINES]
        column += [agg.get(key, (None, None))[0] for _, agg, _ in ours]
        bi = best_index(column, lower)

        for ri, value in enumerate(column):
            is_ours = ri >= len(BASELINES)
            body = fmt(value, digits)
            if is_ours:
                sd = ours[ri - len(BASELINES)][1].get(key, (None, None))[1]
                if sd is not None:
                    body = f"{body}$_{{\\pm {sd:.{digits}f}}}$"
            tex_cells[ri].append(rf"\textbf{{{body}}}" if ri == bi else body)

            md_body = fmt(value, digits)
            if is_ours:
                sd = ours[ri - len(BASELINES)][1].get(key, (None, None))[1]
                if sd is not None:
                    md_body = f"{md_body} ± {sd:.{digits}f}"
            md_cells[ri].append(f"**{md_body}**" if ri == bi else md_body)

    return row_names, col_names, tex_cells, md_cells


def build_ablation_table(
    metrics: Sequence[Metric],
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    rows: List[Tuple[str, Dict[str, float]]] = []
    for run, name, _note in ABLATION:
        path = f"output/screen21/{run}_seed42"
        if not os.path.isfile(os.path.join(path, "test_metrics.json")):
            continue
        rows.append((name, read_run(path)))

    col_names = [tex for _, _, tex, _, _ in metrics] + ["Val. score"]
    row_names = [name for name, _ in rows]
    tex_cells: List[List[str]] = [[] for _ in rows]
    md_cells: List[List[str]] = [[] for _ in rows]

    specs = list(metrics) + [("val_score", "Val. score", "Val. score", 4, False)]
    for key, _disp, _tex, digits, lower in specs:
        column = [data.get(key) for _, data in rows]
        bi = best_index(column, lower)
        for ri, value in enumerate(column):
            body = fmt(value, digits)
            tex_cells[ri].append(rf"\textbf{{{body}}}" if ri == bi else body)
            md_cells[ri].append(f"**{body}**" if ri == bi else body)
    return row_names, col_names, tex_cells, md_cells


def build_grouped_table(
    groups: Sequence[Tuple[str, List[str], str]],
    keys: Sequence[str],
    cumulative_best: bool = True,
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    """把若干组重复实验渲染成一张 均值±标准差 的表，可选按列加粗最优。"""
    lookup = {k: (disp, tex, d, low) for k, disp, tex, d, low in ALL_METRICS}
    col_names = [lookup[k][1] for k in keys] + ["$n$"]

    rows: List[Tuple[str, Dict[str, Tuple[float, float]], int]] = []
    for label, patterns, _note in groups:
        try:
            agg, n = read_runs(patterns)
        except FileNotFoundError:
            continue
        rows.append((label, agg, n))

    row_names = [label for label, _, _ in rows]
    tex_cells: List[List[str]] = [[] for _ in rows]
    md_cells: List[List[str]] = [[] for _ in rows]

    for key in keys:
        digits, lower = lookup[key][2], lookup[key][3]
        column = [agg.get(key, (None, None))[0] for _, agg, _ in rows]
        bi = best_index(column, lower) if cumulative_best else None
        for ri, (_, agg, _) in enumerate(rows):
            mean, sd = agg.get(key, (None, None))
            if mean is None:
                tex_cells[ri].append("--")
                md_cells[ri].append("--")
                continue
            tex_body = f"{mean:.{digits}f}"
            md_body = f"{mean:.{digits}f}"
            # 只有真正重复过才报标准差；单次运行的 np.std 是 0，写出来会误导
            if sd is not None and rows[ri][2] > 1:
                tex_body += rf"$_{{\pm {sd:.{digits}f}}}$"
                md_body += f" ± {sd:.{digits}f}"
            tex_cells[ri].append(rf"\textbf{{{tex_body}}}" if ri == bi else tex_body)
            md_cells[ri].append(f"**{md_body}**" if ri == bi else md_body)

    for ri, (_, _, n) in enumerate(rows):
        tex_cells[ri].append(str(n))
        md_cells[ri].append(str(n))

    return row_names, col_names, tex_cells, md_cells


def build_repeat_table() -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
    keys = ["car_f1", "car_em", "sol_f1", "ratio_w10", "car_top1", "chain", "sss", "gus"]
    lookup = {k: (disp, tex, d, low) for k, disp, tex, d, low in ALL_METRICS}
    col_names = [lookup[k][1] for k in keys] + ["$n$"]

    row_names: List[str] = []
    tex_cells: List[List[str]] = []
    md_cells: List[List[str]] = []
    for label, patterns in REPEAT_GROUPS:
        try:
            agg, n = read_runs(patterns)
        except FileNotFoundError:
            continue
        row_names.append(label)
        tex_row: List[str] = []
        md_row: List[str] = []
        for k in keys:
            digits = lookup[k][2]
            mean, sd = agg.get(k, (None, None))
            if mean is None:
                tex_row.append("--")
                md_row.append("--")
                continue
            tex_row.append(f"{mean:.{digits}f}$_{{\\pm {sd:.{digits}f}}}$")
            md_row.append(f"{mean:.{digits}f} ± {sd:.{digits}f}")
        tex_row.append(str(n))
        md_row.append(str(n))
        tex_cells.append(tex_row)
        md_cells.append(md_row)
    return row_names, col_names, tex_cells, md_cells


def write_pair(
    out_dir: str,
    stem: str,
    caption: str,
    label: str,
    md_title: str,
    row_names: Sequence[str],
    col_names: Sequence[str],
    tex_cells: Sequence[Sequence[str]],
    md_cells: Sequence[Sequence[str]],
    rule_after: Sequence[int] = (),
    note: str = "",
) -> None:
    tex = render_tex(caption, label, col_names, row_names, tex_cells, rule_after, note)
    md = render_md(
        md_title,
        [strip_tex(c) for c in col_names],
        row_names,
        md_cells,
        strip_tex_note(note),
    )
    with open(os.path.join(out_dir, f"{stem}.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex)
    with open(os.path.join(out_dir, f"{stem}.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"[SAVE] {stem}.tex / {stem}.md")


def write_csv(
    out_dir: str,
    bl: Dict[str, Dict[str, float]],
    ours: List[Tuple[str, Dict[str, Tuple[float, float]], int]],
) -> None:
    path = os.path.join(out_dir, "all_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "metric", "mean", "std", "n_repeats"])
        for name in BASELINES:
            for key, _disp, _tex, _d, _low in ALL_METRICS:
                value = bl[name].get(key)
                if value is not None:
                    writer.writerow([name, key, f"{value:.6f}", "", 1])
        for name, agg, n in ours:
            for key, _disp, _tex, _d, _low in ALL_METRICS:
                if key in agg:
                    mean, sd = agg[key]
                    writer.writerow([
                        name, key, f"{mean:.6f}", "" if sd is None else f"{sd:.6f}", n,
                    ])
    print("[SAVE] all_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="output/paper_tables")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bl = read_baselines()
    ours: List[Tuple[str, Dict[str, Tuple[float, float]], int]] = []
    for name, _desc, patterns in OURS_VARIANTS:
        agg, n = read_runs(patterns)
        ours.append((name, agg, n))

    n_rep = ours[0][2]
    for name, _desc, stem in OURS_ENSEMBLES:
        ours.append((name, read_ensemble(stem), 1))

    shared_note = (
        f"输入统一为 21 维分子描述符。我们的单模型报 {n_rep} 次重复"
        "（同一数据划分、不同随机初始化）的均值与标准差；集成与传统基线是确定的单个模型，"
        "报单次结果。加粗为该列最优。"
    )

    rn, cn, tex, md = build_main_table(PER_TASK, bl, ours)
    write_pair(
        args.out_dir, "table1_per_task",
        caption="各子任务上的测试集表现（$n=1145$）。",
        label="tab:per-task", md_title="表 1：各子任务测试集表现（n=1145）",
        row_names=rn, col_names=cn, tex_cells=tex, md_cells=md,
        rule_after=(len(BASELINES) - 1,), note=shared_note,
    )

    rn, cn, tex, md = build_main_table(END_TO_END, bl, ours)
    write_pair(
        args.out_dir, "table2_end_to_end",
        caption="端到端指标（$n=1145$）。Chain Acc 为 cartridge/step/solvent 三级 top-1 全对的比例。",
        label="tab:end-to-end", md_title="表 2：端到端指标（n=1145）",
        row_names=rn, col_names=cn, tex_cells=tex, md_cells=md,
        rule_after=(len(BASELINES) - 1,),
        note=shared_note + " SVM 在 HC 上的高值来自退化解：它的 Cartridge F1 仅 0.059，"
             "近乎不作预测，而空预测不产生层级冲突。",
    )

    rn, cn, tex, md = build_ablation_table(
        [m for m in ALL_METRICS if m[0] in
         {"car_f1", "car_em", "sol_f1", "ratio_w10", "car_top1", "chain", "sss", "gus"}]
    )
    write_pair(
        args.out_dir, "table3_ablation",
        caption="消融实验。每行只改动一处，或叠加已验证有效的改动。",
        label="tab:ablation", md_title="表 3：消融实验",
        row_names=rn, col_names=cn, tex_cells=tex, md_cells=md,
        note="单次训练。Val.\\ score 为验证集调优后的选模分，只在同一套选模权重内可比。"
             "配置选择一律依据验证集，测试集不参与任何选择。",
    )

    rn, cn, tex, md = build_repeat_table()
    write_pair(
        args.out_dir, "table4_repeats",
        caption="重复实验的均值与标准差。固定数据划分，只改变随机初始化。",
        label="tab:repeats", md_title="表 4：重复实验（噪声底）",
        row_names=rn, col_names=cn, tex_cells=tex, md_cells=md,
        note="该表给出判定改动是否真实有效的门槛：差距小于 2 个标准差的不应声称显著。",
    )

    keys2 = ["car_top1", "chain", "hit1", "car_f1", "sol_f1", "sss", "gus"]
    rn, cn, tex, md = build_grouped_table(ABLATION2, keys2)
    write_pair(
        args.out_dir, "table5_improvements",
        caption="第二轮改进的逐步消融，均为 4 次重复的均值与标准差。",
        label="tab:improve", md_title="表 5：第二轮改进的逐步消融",
        row_names=rn, col_names=cn, tex_cells=tex, md_cells=md,
        note="每行在上一行基础上叠加一处改动。bagging 单成员因只见到约 63\\% 的训练样本而"
             "低于前一行，其价值体现在集成之后（见表 2）。",
    )

    rn, cn, tex, md = build_grouped_table(NEGATIVE, keys2, cumulative_best=False)
    write_pair(
        args.out_dir, "table6_negative",
        caption="无效的尝试。均在同一基底上只改动一处。",
        label="tab:negative", md_title="表 6：无效的尝试",
        row_names=rn, col_names=cn, tex_cells=tex, md_cells=md,
        note="$n$ 列给出重复次数。对照是「rank-gauss + 默认超参」那一行"
             "（表 5 第二行，Cart.\\ top-1 $0.5878$）。列出这些失败尝试是为了说明搜索范围，"
             "也避免后续重复投入。",
    )

    write_csv(args.out_dir, bl, ours)
    print(f"\n全部表格已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
