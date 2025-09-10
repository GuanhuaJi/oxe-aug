# local_lerobot_dataloader.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Callable, Sequence, Any

import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class LocalLeRobotDataset(Dataset):
    """
    Thin, local-only wrapper around LeRobotDataset.

    Parameters
    ----------
    root : str | Path
        Parent directory that contains <repo_id>/ (the dataset you built).
    repo_id : str
        Folder name under `root` (e.g., "taco_play_train_0_100").
    episodes : Optional[Sequence[int]]
        If given, restrict to these episode indices.
    delta_timestamps : Optional[Dict[str, List[float]]]
        LeRobot time-windowing (e.g., {"observation.images.widowX":[-0.2,0]}).
    select_keys : Optional[Sequence[str]]
        If given, only keep these keys in each sample (others dropped).
    transforms : Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]
        Per-sample transform applied after key selection.
    video_backend : str
        One of {"pyav","opencv","torchcodec"}; "pyav" is a stable default.

    Notes
    -----
    - This class loads **from disk**. Make sure `root/repo_id/meta/info.json` exists.
    - LeRobot can fetch multiple frames around an index via `delta_timestamps`. :contentReference[oaicite:2]{index=2}
    """
    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        episodes: Optional[Sequence[int]] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        select_keys: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_backend: str = "pyav",
        download_videos: bool = True,
    ):
        root = Path(root)
        self.local_dir = root / repo_id
        info_json = self.local_dir / "meta" / "info.json"
        if not info_json.exists():
            raise FileNotFoundError(
                f"Local dataset not found at {self.local_dir} (missing {info_json}). "
                f"Pass the same root/repo_id you used when building the dataset."
            )

        # load the underlying LeRobotDataset (local folder via root)
        self.ds = LeRobotDataset(
            repo_id=repo_id,
            root=str(self.local_dir),
            download_videos=download_videos,
            video_backend=video_backend
        )

        # Build an index subset if episodes are specified
        if episodes is None:
            self.index = list(range(len(self.ds)))
        else:
            episodes = list(episodes)
            # `episode_data_index` holds start/end (inclusive) per episode. :contentReference[oaicite:4]{index=4}
            starts = self.ds.episode_data_index["from"]
            ends   = self.ds.episode_data_index["to"]
            sel = []
            for e in episodes:
                s, t = int(starts[e].item()), int(ends[e].item())
                sel.extend(range(s, t + 1))
            self.index = sel

        self.select_keys = set(select_keys) if select_keys else None
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.ds[self.index[i]]  # returns a dict of tensors/values
        if self.select_keys is not None:
            sample = {k: v for k, v in sample.items() if k in self.select_keys}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


def lerobot_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Safe collate that stacks tensors, keeps strings as lists, and recurses through dicts.
    """
    from torch.utils.data._utils.collate import default_collate

    def _collate(items):
        if isinstance(items[0], dict):
            keys = items[0].keys()
            return {k: _collate([it[k] for it in items]) for k in keys}
        # strings or bytes: keep as list
        if isinstance(items[0], (str, bytes)):
            return list(items)
        # torch tensors: stack
        if torch.is_tensor(items[0]):
            return torch.stack(items, dim=0)
        # numpy arrays → tensor then stack via default
        try:
            return default_collate(items)
        except Exception:
            # fallback: keep raw list (e.g., variable-length)
            return list(items)

    return _collate(batch)


def make_dataloader(
    root: str | Path,
    repo_id: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    episodes: Optional[Sequence[int]] = None,
    delta_timestamps: Optional[Dict[str, List[float]]] = None,
    select_keys: Optional[Sequence[str]] = None,
    transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_backend: str = "pyav",
) -> DataLoader:
    ds = LocalLeRobotDataset(
        root=root,
        repo_id=repo_id,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        select_keys=select_keys,
        transforms=transforms,
        video_backend=video_backend,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lerobot_collate,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

if __name__ == "__main__":
    # Example usage: adjust paths/params as needed
    loader = make_dataloader(
        root="/home/abinayadinesh/lerobot_dataset",
        repo_id="toto_train_0_100",
        batch_size=16,
        shuffle=True,
        num_workers=4,
        episodes=[0,1,2],                 # only a few episodes
        video_backend="pyav",             # avoid torchcodec quirks
    )
    batch = next(iter(loader))
    print(batch.keys())