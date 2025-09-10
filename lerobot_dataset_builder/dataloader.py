# dataloader_lerobot.py
from pathlib import Path
from typing import Optional, Sequence, Dict, Any

import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def make_lerobot_dataset(
    root,
    repo_id,
    video_backend='pyav',
    *,
    episodes=None,
    image_transforms=None,                 # e.g., torchvision.transforms.v2 pipeline
    download_videos=False,         # you're loading local data, no need to download
):
    """
    Load your local LeRobot dataset as a PyTorch-compatible Dataset.

    Args:
        root: parent directory that contains <repo_id>/ (the folder created by your writer)
        repo_id: the dataset ID string you used in LeRobotDataset.create(...)
        episodes: optional subset of episode indices to load
        image_transforms: optional torchvision v2 transforms (applied to image tensors)
        download_videos: keep False for local datasets
        video_backend: 'pyav' | 'video_reader' | 'torchcodec' (or None for default)
    """
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=str(root),                    # points to the parent dir that contains <repo_id>/
        episodes=list(episodes) if episodes is not None else None,
        image_transforms=image_transforms,
        download_videos=download_videos,
        video_backend=video_backend,
    )
    return ds


def make_lerobot_loader(
    ds,
    *,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
):
    """
    Wrap the dataset in a PyTorch DataLoader. We use the default collate:
    - Tensors are stacked to [B, ...]
    - Strings (e.g., natural_language_instruction) become List[str] in the batch
    """
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

# =========================
# Append-only export helpers
# =========================
import os
import numpy as np
import imageio.v2 as imageio

def _to_numpy(x):
    # torch.Tensor or np.ndarray -> np.ndarray (1D for vectors, HWC for images)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def export_first_episode(ds, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ep = 0

    # get index range for the first episode
    # (LeRobot exposes per-episode index windows via episode_data_index)
    from_idx = ds.episode_data_index["from"][ep]
    to_idx   = ds.episode_data_index["to"][ep]
    try:
        from_idx = int(from_idx.item())
        to_idx   = int(to_idx.item())
    except Exception:
        from_idx = int(from_idx)
        to_idx   = int(to_idx)

    idxs = range(from_idx, to_idx)

    # -------- dump numeric features to .txt --------
    wanted_scalar_keys = []
    if "observation.state" in ds.features:
        wanted_scalar_keys.append("observation.state")

    # discover robot-specific keys that exist in this dataset
    for robot in ["kuka_iiwa","xarm7","sawyer","ur5e","kinova3","jaco","widowx","panda","google_robot"]:
        k_j = f"observation.{robot}.joints"
        k_t = f"observation.{robot}.base_position"
        k_r = f"observation.{robot}.base_orientation"
        if k_j in ds.features: wanted_scalar_keys.append(k_j)
        if k_t in ds.features: wanted_scalar_keys.append(k_t)
        if k_r in ds.features: wanted_scalar_keys.append(k_r)

    # collect and save
    for key in wanted_scalar_keys:
        rows = []
        for i in idxs:
            v = _to_numpy(ds[i][key]).reshape(-1)
            rows.append(v)
        arr = np.stack(rows, axis=0)
        out_path = os.path.join(out_dir, key.replace(".", "_") + ".txt")
        np.savetxt(out_path, arr, fmt="%.6f")
        print(f"wrote {out_path}  shape={arr.shape}")

    # -------- dump videos per camera stream --------
    fps = ds.fps
    for cam_key in ds.meta.camera_keys:
        vid_path = os.path.join(out_dir, cam_key.replace(".", "_") + ".mp4")
        with imageio.get_writer(vid_path, fps=fps, codec="libx264") as w:
            for i in idxs:
                frame = _to_numpy(ds[i][cam_key])  # CHW or HWC depending on backend
                # ensure HWC uint8
                if frame.ndim == 3 and frame.shape[0] in (1,3):  # CHW -> HWC
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                w.append_data(frame)
        print(f"wrote {vid_path}")


if __name__ == "__main__":
    # Example — adjust to your paths/ID
    root = "/home/guanhuaji/lerobot_datasets/nyu_franka_play_dataset_converted_externally_to_rlds_train_0_5"   # the parent folder you passed as --out_root
    repo_id = "nyu_franka_play_dataset_converted_externally_to_rlds_train_0_5"  # whatever you used in create(...)

    ds = make_lerobot_dataset(root=root, repo_id=repo_id)

    # Inspect available keys (useful for picking what to feed your model)
    print("Num frames:", len(ds))
    print("Camera keys:", ds.meta.camera_keys)  # e.g. ['observation.images.xarm7', 'observation.images.image']
    print("All feature keys:", list(ds.features.keys())[:10], "...")

    # Basic loader
    loader = make_lerobot_loader(ds, batch_size=8, num_workers=0)

    batch = next(iter(loader))  # one minibatch (dict)
    # Images: each camera key is a tensor [B, C, H, W]
    for cam_key in ds.meta.camera_keys:
        imgs = batch[cam_key]  # torch.FloatTensor, CHW per sample → [B, C, H, W]
        print(cam_key, imgs.shape)

    # Robot-specific fields you added (if present)
    for robot in ["xarm7", "sawyer", "ur5e", "kinova3", "jaco", "kuka_iiwa", "widowx"]:
        k = f"observation.{robot}.joints"
        if k in batch:
            print(k, batch[k].shape)  # [B, J]

    # RLDS-mapped extras (if present)
    if "natural_language_instruction" in batch:
        print("natural_language_instruction example:", batch["natural_language_instruction"][0])  # list[str]
    if "observation.state" in batch:
        print("state shape:", batch["observation.state"].shape)  # [B, D]

if __name__ == "__main__":
    # change this path to wherever you want the files saved
    export_dir = "/home/guanhuaji/lerobot_datasets/nyu_franka_play_dataset_converted_externally_to_rlds_train_0_5"
    export_first_episode(ds, export_dir)