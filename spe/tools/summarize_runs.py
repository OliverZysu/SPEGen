"""
把一批 run 目录汇总成一张表，支持按初始化种子聚合成 均值±标准差。

    python -m spe.tools.summarize_runs 'output/improve21/s1_*'
    python -m spe.tools.summarize_runs 'output/improve21/s1_*' --group

`--group` 会把名字里的 `_init<N>` 后缀剥掉后当作同一个配置的重复，输出 均值±标准差。
判定改进是否显著要用这个模式：单次运行的差异经常落在噪声里
（我们量过 cartridge top-1 的 1σ 约 0.005~0.009）。

StrictAcc 不列出：它要求整套方案逐项精确匹配，所有方法都低于 5%，缺乏区分度。
"""

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# (显示名, json 路径, 小数位)
COLUMNS: Sequence[Tuple[str, Tuple[str, ...], int]] = (
    ("Car top1", ("overall_clean", "cartridge_top1_acc"), 4),
    ("链路", ("overall_clean", "chain_acc_no_ratio"), 4),
    ("Hit@1", ("overall_clean", "hit_at_1_info_only"), 4),
    ("Car F1", ("cartridge", "micro_f1"), 4),
    ("Car EM", ("cartridge", "exact_match_rate"), 4),
    ("Sol F1", ("solvent", "micro_f1"), 4),
    ("SSS", ("overall_clean", "scheme_similarity_info_only"), 4),
    ("HC", ("overall_clean", "hierarchical_consistency_info_only"), 4),
    ("GUS", ("overall", "global_utility_score"), 2),
)


def dig(node: Dict, path: Sequence[str]) -> Optional[float]:
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def read_run(run_dir: str) -> Optional[Dict[str, float]]:
    path = os.path.join(run_dir, "test_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    # 新旧两种布局：顶层直接是指标，或裹在 "metrics" 里
    metrics = blob.get("metrics", blob)
    out: Dict[str, float] = {}
    for name, path_keys, _ in COLUMNS:
        value = dig(metrics, path_keys)
        if value is not None:
            out[name] = value
    for key in ("val_select_score_tuned", "val_select_score"):
        if key in blob:
            out["验证分"] = float(blob[key])
            break
    return out


def strip_init(name: str) -> str:
    return re.sub(r"_init\d+$", "", name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("patterns", nargs="+", help="run 目录的 glob，可给多个")
    p.add_argument("--group", action="store_true",
                   help="把 _init<N> 视作同一配置的重复，输出 均值±标准差")
    p.add_argument("--sort", default="Car top1",
                   help="按哪一列降序排，默认 Car top1")
    p.add_argument("--strip", default="", help="从配置名里去掉的前缀")
    args = p.parse_args()

    dirs: List[str] = []
    for pat in args.patterns:
        dirs.extend(sorted(glob.glob(pat)))

    runs: List[Tuple[str, Dict[str, float]]] = []
    for d in dirs:
        data = read_run(d)
        if data is None:
            continue
        name = os.path.basename(d.rstrip("/"))
        if args.strip:
            name = name.replace(args.strip, "")
        runs.append((name, data))

    if not runs:
        print("没找到任何含 test_metrics.json 的目录")
        return

    keys = [name for name, _, _ in COLUMNS] + ["验证分"]
    digits = {name: d for name, _, d in COLUMNS}
    digits["验证分"] = 4

    if args.group:
        buckets: Dict[str, List[Dict[str, float]]] = {}
        for name, data in runs:
            buckets.setdefault(strip_init(name), []).append(data)
        rows: List[Tuple[str, Dict[str, Tuple[float, Optional[float], int]]]] = []
        for name, items in buckets.items():
            agg: Dict[str, Tuple[float, Optional[float], int]] = {}
            for key in keys:
                vals = [d[key] for d in items if key in d]
                if not vals:
                    continue
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
                agg[key] = (float(np.mean(vals)), sd, len(vals))
            rows.append((name, agg))
        rows.sort(key=lambda r: -r[1].get(args.sort, (float("-inf"), None, 0))[0])

        name_w = max(len(n) for n, _ in rows) + 2
        cell_w = 17
        print("".ljust(name_w) + "".join(k.rjust(cell_w) for k in keys) + "  n")
        print("-" * (name_w + cell_w * len(keys) + 4))
        for name, agg in rows:
            line = name.ljust(name_w)
            n_rep = 0
            for key in keys:
                if key not in agg:
                    line += "-".rjust(cell_w)
                    continue
                mean, sd, n = agg[key]
                n_rep = max(n_rep, n)
                body = f"{mean:.{digits[key]}f}"
                if sd is not None:
                    body += f"±{sd:.{digits[key]}f}"
                line += body.rjust(cell_w)
            print(line + f"  {n_rep}")
    else:
        runs.sort(key=lambda r: -r[1].get(args.sort, float("-inf")))
        name_w = max(len(n) for n, _ in runs) + 2
        print("".ljust(name_w) + "".join(k.rjust(10) for k in keys))
        print("-" * (name_w + 10 * len(keys)))
        for name, data in runs:
            line = name.ljust(name_w)
            for key in keys:
                line += ("-" if key not in data
                         else f"{data[key]:.{digits[key]}f}").rjust(10)
            print(line)


if __name__ == "__main__":
    main()
