import csv
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .parsing import (
    STEP_NAMES,
    build_method_step_map_from_csv,
    compute_labels_for_pollutant,
    parse_list_str,
)


@dataclass
class DatasetArtifacts:
    descriptor_cols: List[str]
    cartridge_cols: List[str]
    solvent_vocab: List[str]
    solvent2id: Dict[str, int]


class SPEDataset(Dataset):
    """
    Memory-conscious Dataset:
      - Stores y_* as uint8 arrays
      - Converts to float tensors in __getitem__

    Each item returns:
      - x: FloatTensor (D,)
      - y_cartridge: FloatTensor (C_car,)
      - y_step: FloatTensor (5,)
      - y_solvent: FloatTensor (5, S)
      - ratio_pairs: list of (step_id:int, solvent_id:int, ratio:float)
      - has_spe_info: FloatTensor scalar (1.0/0.0)
      - cid: str
      - method_ids: list[str]
    """
    def __init__(
        self,
        xs: np.ndarray,
        y_cartridge: np.ndarray,
        y_step: np.ndarray,
        y_solvent: np.ndarray,
        ratio_pairs: List[List[Tuple[int, int, float]]],
        has_spe_info: np.ndarray,
        cids: List[str],
        method_lists: List[List[str]],
    ):
        self.xs = xs.astype(np.float32)
        self.y_cartridge = y_cartridge.astype(np.uint8)
        self.y_step = y_step.astype(np.uint8)
        self.y_solvent = y_solvent.astype(np.uint8)
        self.ratio_pairs = ratio_pairs
        self.has_spe_info = has_spe_info.astype(np.uint8)
        self.cids = cids
        self.method_lists = method_lists

    def __len__(self) -> int:
        return self.xs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": torch.from_numpy(self.xs[idx]).float(),
            "y_cartridge": torch.from_numpy(self.y_cartridge[idx]).float(),
            "y_step": torch.from_numpy(self.y_step[idx]).float(),
            "y_solvent": torch.from_numpy(self.y_solvent[idx]).float(),
            "ratio_pairs": self.ratio_pairs[idx],
            "has_spe_info": torch.tensor(float(self.has_spe_info[idx])),
            "cid": self.cids[idx],
            "method_ids": self.method_lists[idx],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y_cartridge = torch.stack([b["y_cartridge"] for b in batch], dim=0)
    y_step = torch.stack([b["y_step"] for b in batch], dim=0)
    y_solvent = torch.stack([b["y_solvent"] for b in batch], dim=0)
    has_spe_info = torch.stack([b["has_spe_info"] for b in batch], dim=0)

    return {
        "x": x,
        "y_cartridge": y_cartridge,
        "y_step": y_step,
        "y_solvent": y_solvent,
        "ratio_pairs": [b["ratio_pairs"] for b in batch],
        "has_spe_info": has_spe_info,
        "cids": [b["cid"] for b in batch],
        "method_ids": [b["method_ids"] for b in batch],
    }


def build_datasets(
    data_dir: str,
    cache_path: Optional[str] = None,
    require_spe_info: bool = True,
    unk_solvent_token: str = "__UNK__",
) -> Tuple[SPEDataset, DatasetArtifacts]:
    """
    Load CSVs from data_dir, build aggregated labels, and return dataset + artifacts.

    If cache_path is provided and exists, loads preprocessed arrays from cache.
    """
    proc_path = os.path.join(data_dir, "processed_molecular_data.csv")
    spe_path = os.path.join(data_dir, "spe_solvent_ratio.csv")

    # Cache is versioned to avoid stale preprocessing when task definition changes.
    CACHE_VERSION = "ratio_regression_v1"

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("cache_version") == CACHE_VERSION:
            ds = SPEDataset(**blob["dataset_kwargs"])
            artifacts = DatasetArtifacts(**blob["artifacts"])
            return ds, artifacts

    # 1) Build method->step info mapping and solvent vocab
    method_step_map, solvent_vocab = build_method_step_map_from_csv(
        spe_csv_path=spe_path,
        unk_solvent_token=unk_solvent_token,
    )
    solvent2id = {s: i for i, s in enumerate(solvent_vocab)}

    n_steps = len(STEP_NAMES)
    n_solvent = len(solvent_vocab)

    # 2) Read processed header to determine descriptor and cartridge columns
    with open(proc_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    start = header.index("Complexity_value")
    end = header.index("LogP_value")
    descriptor_cols = header[start:end + 1]
    cartridge_cols = [c for c in header if c.startswith("Cartridge_Class_")]

    idx_cid = header.index("CID")
    idx_casmp = header.index("CasMp")
    idx_desc = [header.index(c) for c in descriptor_cols]
    idx_car = [header.index(c) for c in cartridge_cols]

    # 3) First pass: count how many rows kept (to pre-allocate arrays)
    keep_mask: List[bool] = []
    with open(proc_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row in reader:
            method_ids = parse_list_str(row[idx_casmp])
            # quick check: has any method in map?
            has_info = any(m in method_step_map for m in method_ids)
            keep = (has_info or not require_spe_info)
            keep_mask.append(bool(keep))
    n_keep = int(sum(keep_mask))

    # 4) Pre-allocate dense arrays (use uint8 for labels)
    xs = np.zeros((n_keep, len(descriptor_cols)), dtype=np.float32)
    y_car = np.zeros((n_keep, len(cartridge_cols)), dtype=np.uint8)
    y_step = np.zeros((n_keep, n_steps), dtype=np.uint8)
    y_sol = np.zeros((n_keep, n_steps, n_solvent), dtype=np.uint8)
    has_info_arr = np.zeros((n_keep,), dtype=np.uint8)
    cids: List[str] = []
    method_lists: List[List[str]] = []
    ratio_pairs_list: List[List[Tuple[int, int, float]]] = []

    # 5) Second pass: fill arrays
    out_i = 0
    with open(proc_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row_i, row in enumerate(reader):
            if not keep_mask[row_i]:
                continue

            cid = parse_list_str(row[idx_cid])[0]
            method_ids = parse_list_str(row[idx_casmp])

            x = np.array([float(row[i]) for i in idx_desc], dtype=np.float32)
            ycar = np.array([float(row[i]) for i in idx_car], dtype=np.uint8)

            has_info, ystep_list, ysol_list, ratio_pairs = compute_labels_for_pollutant(
                method_ids=method_ids,
                method_step_map=method_step_map,
                solvent2id=solvent2id,
                n_steps=n_steps,
                n_solvent=n_solvent,
            )

            xs[out_i] = x
            y_car[out_i] = ycar
            y_step[out_i] = np.array(ystep_list, dtype=np.uint8)
            y_sol[out_i] = np.array(ysol_list, dtype=np.uint8)
            has_info_arr[out_i] = 1 if has_info else 0
            cids.append(cid)
            method_lists.append(method_ids)
            ratio_pairs_list.append(ratio_pairs)
            out_i += 1

    dataset_kwargs = {
        "xs": xs,
        "y_cartridge": y_car,
        "y_step": y_step,
        "y_solvent": y_sol,
        "ratio_pairs": ratio_pairs_list,
        "has_spe_info": has_info_arr,
        "cids": cids,
        "method_lists": method_lists,
    }
    ds = SPEDataset(**dataset_kwargs)

    artifacts = DatasetArtifacts(
        descriptor_cols=descriptor_cols,
        cartridge_cols=cartridge_cols,
        solvent_vocab=solvent_vocab,
        solvent2id=solvent2id,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"cache_version": CACHE_VERSION, "dataset_kwargs": dataset_kwargs, "artifacts": artifacts.__dict__},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return ds, artifacts


def split_indices(n: int, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    assert 0.0 < train_ratio < 1.0
    assert 0.0 <= val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def build_loaders(
    dataset: SPEDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_ds = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
