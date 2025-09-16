# selective_lerobot_dataloader.py
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames  # ← frame decode

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
    Read selected columns from LeRobot (parquet) and lazily decode requested video keys.
    """
    def __init__(self, root, repo_id, keys, episodes=None, fmt="torch", download_videos=False):
        root = Path(root)
        self.root = root
        self.repo_id = repo_id
        self.ds_dir = (self.root / self.repo_id)

        # Build the base dataset; set download_videos=True if you need auto-fetch
        base = LeRobotDataset(
            repo_id=repo_id,
            root=str(root / repo_id),
            download_videos=download_videos,
            video_backend="pyav",
        )
        self.base = base
        self.meta = base.meta
        self.tolerance_s = base.tolerance_s
        self.video_backend = base.video_backend

        # Which of the requested keys are videos?
        meta_video_keys = set(getattr(self.meta, "video_keys", []))
        self._video_keys = [k for k in keys if k in meta_video_keys]
        hf_only_keys = [k for k in keys if k not in meta_video_keys]

        # Use HF dataset for non-video columns
        hf = base.hf_dataset
        try:
            hf.reset_format()
        except Exception:
            pass

        # Always keep episode_index + timestamp (needed to pick frames)
        keep = ["episode_index", "timestamp"] + [k for k in hf_only_keys if k in hf.column_names]
        drop = [c for c in hf.column_names if c not in keep]
        if drop:
            hf = hf.remove_columns(drop)

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

        # Reported keys (what __getitem__ returns)
        self.keys = ["episode_index", "timestamp"] + hf_only_keys + self._video_keys

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=None)
    def _video_path(self, episode_index: int, key: str) -> str:
        p = Path(self.meta.get_video_file_path(episode_index, key))  # e.g. "videos/chunk-000/..."
        if not p.is_absolute():
            p = self.ds_dir / p  # join with dataset root/repo_id
        return str(p)

    @staticmethod
    def _to_chw_uint8(x) -> torch.Tensor:
        t = torch.as_tensor(x)
        if t.ndim == 4:
            t = t[0]          # take first (only) frame
        if t.ndim == 3 and t.shape[-1] in (1, 3, 4):   # HWC -> CHW
            t = t.permute(2, 0, 1)
        elif t.ndim == 2:                               # HW -> 1HW
            t = t.unsqueeze(0)
        if t.dtype != torch.uint8:
            t = t.clamp(0, 255).to(torch.uint8)
        return t
    
    def _frame_to_chw_float01(self, frames) -> torch.Tensor:
        t = torch.as_tensor(frames)
        if t.ndim == 4:            # take first (only) frame
            t = t[0]
        # if decoder returned uint8, normalize; otherwise keep float [0,1]
        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        # ensure CHW
        if t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        # if t.ndim == 3 and t.shape[0] == 3:
        #     t = t[[2, 1, 0], ...]  # RGB -> BGR
        return t.contiguous()

    def __getitem__(self, i):
        row = dict(self.hf[int(self.index[i])])  # parquet-backed fields
        if self._video_keys:
            ep = int(row["episode_index"])
            ts = float(row["timestamp"])
            for k in self._video_keys:
                vpath = self._video_path(ep, k)
                frames = decode_video_frames(vpath, [ts], self.tolerance_s, self.video_backend)
                # row[k] = self._to_chw_uint8(frames)  # CHW uint8
                row[k] = self._frame_to_chw_float01(frames)
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
    download_videos=False,
):
    ds = SelectiveLeRobotDataset(root, repo_id, keys, episodes=episodes, fmt=fmt, download_videos=download_videos)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )