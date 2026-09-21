import ast
import csv
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Set


STEP_NAMES = ["Sample loading", "Condition", "Wash", "Elute", "Reconstitute"]


def parse_list_str(x: str) -> List[str]:
    """
    The CSV stores some fields as a Python-like list string, e.g. "['a', 'b']".
    Returns a list of string elements.
    """
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    s = str(x)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(t) for t in v]
        return [str(v)]
    except Exception:
        return [s]


_cell_pat = re.compile(r'^\(?\s*([^,]+?)\s*,\s*([^)]+?)\s*\)?$')


def parse_spe_cell(cell: object) -> List[Tuple[str, str]]:
    """
    Parse one `spe_solvent_ratio.csv` cell, which may look like:
        "(water, null)"
        "(methanol, null); (water, null)"
        "" / NaN
    Returns list of (solvent_str, conc_str) pairs as lowercase strings.
    """
    if cell is None:
        return []
    if isinstance(cell, float) and math.isnan(cell):
        return []
    s = str(cell).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    parts = [p.strip() for p in s.split(";")]
    out: List[Tuple[str, str]] = []
    for p in parts:
        m = _cell_pat.match(p)
        if not m:
            continue
        sol = m.group(1).strip().lower()
        conc = m.group(2).strip().lower()
        out.append((sol, conc))
    return out


def parse_ratio_value(
    ratio_str: str,
    null_token: str = "null",
) -> Optional[float]:
    """Parse ratio (0-1) from string.

    Returns:
      - float in [0,1] if numeric
      - None if unknown / null

    Notes:
      The input CSV stores ratio as strings like "0.9" or "null".
    """
    s = (ratio_str or "").strip().lower()
    if s == "" or s in {"nan", "none"} or s == null_token:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    # Clamp for safety
    v = min(max(v, 0.0), 1.0)
    return v


def build_method_step_map_from_csv(
    spe_csv_path: str,
    unk_solvent_token: str = "__UNK__",
    null_ratio_token: str = "null",
) -> Tuple[Dict[str, Dict[str, List[Tuple[str, str, Optional[float]]]]], List[str]]:
    """
    Build:
      method_step_map[method_id][step_name] = list of tuples (solvent_norm, ratio_raw, ratio_value)

    Where ratio_value is:
      - float in [0,1] if known
      - None if unknown ("null")

    Also returns solvent vocabulary list (sorted) including unk_solvent_token.

    This function does NOT require pandas.
    """
    solvent_set: Set[str] = set()
    method_step_map: Dict[str, Dict[str, List[Tuple[str, str, Optional[float]]]]] = {}

    with open(spe_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_id = parse_list_str(row.get("CasMp", ""))[0]
            step_info: Dict[str, List[Tuple[str, str, Optional[float]]]] = {}
            for step in STEP_NAMES:
                pairs = parse_spe_cell(row.get(step, ""))
                normed: List[Tuple[str, str, Optional[float]]] = []
                for sol, conc in pairs:
                    sol_norm = sol.strip().lower()
                    if sol_norm in {"unknown", "unk", "none", "null", ""}:
                        sol_norm = unk_solvent_token
                    ratio_val = parse_ratio_value(conc, null_token=null_ratio_token)
                    normed.append((sol_norm, conc, ratio_val))
                    solvent_set.add(sol_norm)
                step_info[step] = normed
            method_step_map[method_id] = step_info

    solvent_set.add(unk_solvent_token)
    solvent_vocab = sorted(solvent_set)
    return method_step_map, solvent_vocab


def compute_labels_for_pollutant(
    method_ids: Sequence[str],
    method_step_map: Dict[str, Dict[str, List[Tuple[str, str, Optional[float]]]]],
    solvent2id: Dict[str, int],
    n_steps: int,
    n_solvent: int,
    ratio_agg: str = "mean",
) -> Tuple[bool, List[float], List[List[float]], List[Tuple[int, int, float]]]:
    """
    Aggregate step/solvent/ratio labels across all methods for a pollutant.

    Returns:
      has_spe_info: bool (at least one method found in method_step_map)
      y_step: length n_steps multi-hot (float)
      y_solvent: (n_steps, n_solvent) multi-hot (float)
      ratio_pairs: list of (step_id, solvent_id, ratio_value)
                 only for (step, solvent) that has at least one numeric ratio.

    Notes:
      A pollutant may map to multiple methods. For a given (step, solvent) pair,
      if multiple numeric ratios exist across methods, we aggregate them.
      By default we use mean aggregation.
    """
    y_step = [0.0] * n_steps
    y_solvent = [[0.0] * n_solvent for _ in range(n_steps)]

    ratio_dict: Dict[Tuple[int, int], List[float]] = {}
    found_any = False

    for m in method_ids:
        if m not in method_step_map:
            continue
        found_any = True
        step_info = method_step_map[m]
        for j, step in enumerate(STEP_NAMES[:n_steps]):
            pairs = step_info.get(step, [])
            if len(pairs) == 0:
                continue
            y_step[j] = 1.0
            for sol_norm, ratio_raw, ratio_val in pairs:
                sid = solvent2id.get(sol_norm)
                if sid is None:
                    continue
                y_solvent[j][sid] = 1.0
                if ratio_val is None:
                    continue
                key = (j, sid)
                ratio_dict.setdefault(key, []).append(float(ratio_val))

    ratio_pairs: List[Tuple[int, int, float]] = []
    for (j, sid), vals in ratio_dict.items():
        if len(vals) == 0:
            continue
        if ratio_agg == "median":
            ratio = float(sorted(vals)[len(vals) // 2])
        else:
            ratio = float(sum(vals) / len(vals))
        ratio_pairs.append((j, sid, ratio))

    return found_any, y_step, y_solvent, ratio_pairs


def build_method_step_map_subset_from_csv(
    spe_csv_path: str,
    method_id_set: Set[str],
    unk_solvent_token: str = "__UNK__",
    null_ratio_token: str = "null",
) -> Dict[str, Dict[str, List[Tuple[str, str, Optional[float]]]]]:
    """Like build_method_step_map_from_csv, but only keeps methods in method_id_set (memory saver for reporting)."""

    method_step_map: Dict[str, Dict[str, List[Tuple[str, str, Optional[float]]]]] = {}
    if not method_id_set:
        return method_step_map

    with open(spe_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method_id = parse_list_str(row.get("CasMp", ""))[0]
            if method_id not in method_id_set:
                continue
            step_info: Dict[str, List[Tuple[str, str, Optional[float]]]] = {}
            for step in STEP_NAMES:
                pairs = parse_spe_cell(row.get(step, ""))
                normed: List[Tuple[str, str, Optional[float]]] = []
                for sol, conc in pairs:
                    sol_norm = sol.strip().lower()
                    if sol_norm in {"unknown", "unk", "none", "null", ""}:
                        sol_norm = unk_solvent_token
                    ratio_val = parse_ratio_value(conc, null_token=null_ratio_token)
                    normed.append((sol_norm, conc, ratio_val))
                step_info[step] = normed
            method_step_map[method_id] = step_info
    return method_step_map
