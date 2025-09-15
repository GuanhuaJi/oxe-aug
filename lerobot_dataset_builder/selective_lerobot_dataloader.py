# selective_lerobot_dataloader.py
from __future__ import annotations
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _load_info(root, repo_id):
    p = Path(root) / repo_id / "meta" / "info.json"
    with open(p, "r") as f:
        return json.load(f)

def discover_keys(info, suffix=None):
    feats = info.get("features", {})
    if suffix is None:
        return sorted(list(feats.keys()))
    return sorted([k for k in feats.keys() if k.endswith(suffix)])

class SelectiveLeRobotDataset(Dataset):
    """
    Read ONLY selected columns from LeRobot's HF dataset (no video decode).
    """
    def __init__(self, root, repo_id, keys, episodes=None, fmt="torch"):
        root = Path(root)
        self.root = root
        self.repo_id = repo_id

        # Build the base dataset WITHOUT downloading/decoding videos
        base = LeRobotDataset(
            repo_id=repo_id,
            root=str(root / repo_id),
            download_videos=False,         # critical: we won't touch videos
            video_backend="pyav",
        )

        # Use the HF dataset inside LeRobot and keep only requested columns
        hf = base.hf_dataset
        try:
            hf.reset_format()
        except Exception:
            pass

        # ensure we always carry episode_index (useful for grouping)
        keep = ["episode_index"] + [k for k in keys if k in hf.column_names]
        drop = [c for c in hf.column_names if c not in keep]
        if drop:
            hf = hf.remove_columns(drop)

        # Return numpy/torch-ready tensors for just those columns
        hf = hf.with_format(fmt, columns=keep)
        self.hf = hf

        # Frame-index subset for requested episodes
        ep_from = base.episode_data_index["from"]
        ep_to   = base.episode_data_index["to"]
        self._ranges = [(int(ep_from[i].item()), int(ep_to[i].item())) for i in range(len(ep_from))]
        if episodes is None:
            self.index = list(range(len(hf)))
        else:
            sel = []
            for e in episodes:
                s, t = self._ranges[e]
                sel.extend(range(s, t + 1))
            self.index = sel

        self.keys = keep

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        # Pull exactly one row from HF dataset; only selected columns are read
        row = self.hf[int(self.index[i])]
        return row


def make_selective_loader(
    root,
    repo_id,
    keys,
    batch_size=512,
    shuffle=False,
    num_workers=4,
    episodes=None,
    fmt="torch",
):
    ds = SelectiveLeRobotDataset(root, repo_id, keys, episodes=episodes, fmt=fmt)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

def make_error_only_loader(root, repo_id, batch_size=2048, num_workers=4, episodes=None):
    info = _load_info(root, repo_id)
    err_keys = discover_keys(info, suffix=".ee_error")
    if not err_keys:
        raise RuntimeError(f"No *.ee_error keys found in {repo_id}")
    return make_selective_loader(
        root=root,
        repo_id=repo_id,
        keys=["episode_index"] + err_keys,   # episode_index included anyway
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        episodes=episodes,
        fmt="torch",
    )
