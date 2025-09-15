# local_dataloader.py
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

# Your config (already in your repo)
from config import RLDS_TO_LEROBOT_DATASET_CONFIGS, ROBOT_JOINT_NUMBERS

# ----------------- Helpers (self-contained) -----------------

@dataclass
class VideoSource:
    path: Path
    target_hw: Tuple[int, int]  # (H, W)

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                h, w = self.target_hw
                if (bgr.shape[0], bgr.shape[1]) != (h, w):
                    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                # BGR->RGB without negative strides
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield rgb  # uint8, HxWx3, contiguous with positive strides
        finally:
            cap.release()

def compute_ee_error(pos, quat, tgt_pos, tgt_quat):
    pos_error = tgt_pos - pos
    quat_error = tgt_quat - quat  # same simplification as writer
    return np.concatenate([pos_error, quat_error], axis=-1).astype(np.float32)

def load_robot_npz_info(npz_path: Path, robot: str) -> Dict[str, np.ndarray]:
    z = dict(np.load(str(npz_path), allow_pickle=True))
    info = {
        "pos":      np.float32(z["replay_positions"]),
        "quat":     np.float32(z["replay_quats"]),
        "grip":     np.float32(z["gripper_state"]).reshape(-1),
        "joints":   np.float32(z["joint_positions"]),
        "base_t":   np.float32(z["translation"][0]).reshape(3,),
        "base_r":   np.float32(z["rotation"][0]).reshape(1,),
        "tgt_pos":  np.float32(z["target_positions"]),
        "tgt_quat": np.float32(z["target_quats"]),
    }
    return info

def fetch_rlds_episode_fields(dataset: str, split: str, ep_index: int, mapping: List[Dict]):
    """Read RLDS fields for one episode using dict-style mapping.
    Returns {lerobot_path: list-of-values}. Set include_rlds=False to skip entirely."""
    import tensorflow_datasets as tfds
    key = f"{dataset}/0.1.0"
    if not hasattr(fetch_rlds_episode_fields, "_builders"):
        fetch_rlds_episode_fields._builders = {}
    builder = fetch_rlds_episode_fields._builders.get(key)
    if builder is None:
        builder_dir = f"{os.environ.get('RLDS_STORAGE','gs://gresearch/robotics')}/{dataset}/0.1.0"
        builder = tfds.builder_from_directory(builder_dir=builder_dir)
        fetch_rlds_episode_fields._builders[key] = builder

    split_spec = f"{split}[{ep_index}:{ep_index+1}]"
    ds = builder.as_dataset(split=split_spec, shuffle_files=False)

    out = {m["lerobot_path"]: [] for m in mapping}
    for episode in ds:
        steps_ds = episode["steps"]
        for step in steps_ds:
            for m in mapping:
                src, dst, dtype = m["rlds_path"], m["lerobot_path"], m.get("dtype")
                parts = [p for p in src.strip("/").split("/") if p]
                cur = step
                iter_parts = parts[1:] if parts and parts[0] == "steps" else parts
                for p in iter_parts:
                    cur = cur[p]
                cur = cur.numpy() if hasattr(cur, "numpy") else cur
                if dtype == "string" and isinstance(cur, (bytes, bytearray, np.bytes_)):
                    cur = cur.decode("utf-8", errors="ignore")
                out[dst].append(cur)
        break
    return out

# ----------------- Dataset -----------------

@dataclass
class EpisodeCache:
    length: int
    poses: np.ndarray
    robots: Dict[str, Dict[str, Any]]   # robot -> {"info": info, "frames": List[np.ndarray]}
    rlds_streams: Dict[str, List[Any]]
    task: Optional[str]

class LocalFolderDataset(Dataset):
    """
    Loads frames directly from your local folder structure:
      <trg_root>/<dataset>/<split>/<ep>/[ robot_overlay_*.mp4 , robot_replay_info_*.npz , end_effector_poses_*.npy ]

    Returns per-frame dicts with the same keys you write into LeRobot.
    """
    def __init__(
        self,
        trg_root: str | Path,
        dataset: str,
        split: str,
        robots: Sequence[str],
        episodes: Sequence[int],
        image_size: Tuple[int, int],
        include_rlds: bool = True,
        preload_videos: bool = True,
        strict_lengths: bool = True,
    ):
        super().__init__()
        self.trg_root = Path(trg_root)
        self.dataset = dataset
        self.split = split
        self.robots = list(robots)
        self.episodes = list(episodes)
        self.H, self.W = image_size
        self.include_rlds = include_rlds
        self.preload_videos = preload_videos
        self.strict_lengths = strict_lengths

        self.cfg = RLDS_TO_LEROBOT_DATASET_CONFIGS[self.dataset]
        self.src_robot = self.cfg["robot"]
        self.rlds_map = self.cfg["rlds_to_lerobot_mappings"]

        self._ep_cache: Dict[int, EpisodeCache] = {}
        self._flat_index: List[Tuple[int, int]] = []  # (episode, frame_idx)

        for ep in self.episodes:
            cache = self._load_episode(ep)
            self._ep_cache[ep] = cache
            self._flat_index.extend((ep, j) for j in range(cache.length))

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep, j = self._flat_index[idx]
        cache = self._ep_cache[ep]

        # Global fields from end_effector_poses
        poses = cache.poses
        fd: Dict[str, Any] = {}
        if self.dataset in ["bridge", "fractal20220817_data", "language_table"]:
            fd["observation.joints"] = np.concatenate(
                [
                    poses['joints'][j][:ROBOT_JOINT_NUMBERS[RLDS_TO_LEROBOT_DATASET_CONFIGS[self.dataset]["robot"]]],
                    np.atleast_1d(poses['grip'][j])
                ],
                axis=0
            ).astype(np.float32)
            fd["observation.ee_pose"] = np.concatenate(
                [np.asarray(poses['pos'][j], dtype=np.float32),
                np.asarray(poses['quat'][j], dtype=np.float32)],
                axis=0
            ).astype(np.float32)
        else:
            fd["observation.joints"] = np.concatenate(
                [
                    np.asarray(poses[j]["joint_positions"][:ROBOT_JOINT_NUMBERS[self.src_robot]], dtype=np.float32),
                    np.atleast_1d(np.asarray(poses[j]["gripper_state"], dtype=np.float32)),
                ],
                axis=0,
            ).astype(np.float32)

            fd["observation.ee_pose"] = np.concatenate(
                [
                    np.asarray(poses[j]["position"], dtype=np.float32),
                    np.asarray(poses[j]["quaternion"], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)

        # Per-robot fields
        for robot, blob in cache.robots.items():
            info = blob["info"]
            frame = blob["frames"][j]  # (H,W,3) uint8 RGB
            fd[f"observation.{robot}.joints"] = np.append(info["joints"][j], info["grip"][j]).astype(np.float32)
            fd[f"observation.{robot}.ee_pose"] = np.concatenate([info["pos"][j], info["quat"][j]], axis=0).astype(np.float32)
            fd[f"observation.{robot}.base_position"] = info["base_t"].astype(np.float32)
            fd[f"observation.{robot}.base_orientation"] = info["base_r"].astype(np.float32)
            fd[f"observation.{robot}.ee_error"] = compute_ee_error(
                info["pos"][j], info["quat"][j], info["tgt_pos"][j], info["tgt_quat"][j]
            )
            fd[f"observation.images.{robot}"] = frame  # uint8 HWC

        # RLDS streams
        for dst_key, seq in cache.rlds_streams.items():
            fd[dst_key] = seq[j]

        return fd

    # ---- internals ----
    def _load_episode(self, ep: int) -> EpisodeCache:
        ep_dir = self.trg_root / self.dataset / self.split / str(ep)
        robots_blob: Dict[str, Dict[str, Any]] = {}
        lengths: List[int] = []

        # per-robot info + frames
        for robot in self.robots:
            info = load_robot_npz_info(ep_dir / f"{robot}_replay_info_{ep}.npz", robot)
            vs = VideoSource(ep_dir / f"{robot}_overlay_{ep}_algo_final.mp4", (self.H, self.W))
            frames = list(vs) if self.preload_videos else list(vs)  # eager list for uniformity
            robots_blob[robot] = {"info": info, "frames": frames}
            lengths.extend([len(frames), int(info["pos"].shape[0])])

        # RLDS (optional)
        if self.include_rlds and self.rlds_map:
            rlds_streams = fetch_rlds_episode_fields(self.dataset, self.split, ep, self.rlds_map)
            # add one video-like length if present
            for m in self.rlds_map:
                if m.get("dtype") == "video":
                    lengths.append(len(rlds_streams[m["lerobot_path"]]))
                    break
        else:
            rlds_streams = {}

        # Poses (global)
        if self.dataset in ["bridge", "fractal20220817_data", "language_table"]:
            info_path = ep_dir / f"{self.src_robot}_replay_info_{ep}.npz"
            poses = load_robot_npz_info(info_path, self.src_robot)
        else:
            poses = np.load(ep_dir / f"end_effector_poses_{ep}.npy", allow_pickle=True)
            lengths.append(len(poses))

        uniq = set(lengths)
        if self.strict_lengths and len(uniq) != 1:
            raise ValueError(f"Length mismatch in episode {ep}: {lengths}")
        T = max(uniq) if uniq else min(lengths)

        task = None
        if "natural_language_instruction" in rlds_streams and rlds_streams["natural_language_instruction"]:
            task = rlds_streams["natural_language_instruction"][0]

        return EpisodeCache(length=T, poses=poses, robots=robots_blob, rlds_streams=rlds_streams, task=task)

# ----------------- Collate & Factory -----------------

def local_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stacks tensors/arrays, keeps strings as lists."""
    def _to_tensor(x):
        if isinstance(x, np.ndarray) and x.dtype != np.object_:
            return torch.from_numpy(x)
        return x
    def _collate(items):
        if isinstance(items[0], dict):
            keys = items[0].keys()
            return {k: _collate([it[k] for it in items]) for k in keys}
        if isinstance(items[0], (str, bytes)):
            return list(items)
        items = [_to_tensor(it) for it in items]
        try:
            return default_collate(items)
        except Exception:
            return items
    return _collate(batch)

def make_local_folder_dataloader(
    trg_root,
    dataset,
    split,
    robots,
    episodes,
    image_size=None,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    include_rlds=False,      # default False to avoid TFDS requirement
    preload_videos=True,
):
    if image_size is None:
        image_size = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["image_size"]
    ds = LocalFolderDataset(
        trg_root=trg_root,
        dataset=dataset,
        split=split,
        robots=robots,
        episodes=episodes,
        image_size=image_size,
        include_rlds=include_rlds,
        preload_videos=preload_videos,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=local_collate,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

# ----------------- Quick smoke test -----------------
if __name__ == "__main__":
    # Example: tweak these to your paths
    loader = make_local_folder_dataloader(
        trg_root="/home/abinayadinesh/rovi_aug_extension_full",
        dataset="toto",
        split="train",
        robots=["widowX", "xarm7", "sawyer", "ur5e"],
        episodes=[0],
        # image_size=None -> will take from config
        batch_size=2,
        shuffle=False,
        num_workers=0,
        include_rlds=False,  # flip to True if TFDS + RLDS storage available
    )
    batch = next(iter(loader))
    print("Batch keys:", list(batch.keys())[:8], "...")
    # Example shapes
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        elif isinstance(v, list) and v and isinstance(v[0], str):
            print(k, f"{len(v)} strings")
