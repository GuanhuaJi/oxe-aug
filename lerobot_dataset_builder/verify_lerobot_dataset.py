#!/usr/bin/env python3
# verify_lerobot_dataset.py

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

# LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # local or HF Hub compatible
# Your config
from config import RLDS_TO_LEROBOT_DATASET_CONFIGS, ROBOT_JOINT_NUMBERS

# ---------- Helpers ----------

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def chw_to_hwc_u8(arr: np.ndarray) -> np.ndarray:
    """Accept (C,H,W) or (H,W,C) arrays in float[0..1] or uint8 and return (H,W,C) uint8."""
    a = arr
    if a.ndim != 3:
        raise ValueError(f"Expected 3D array for image, got {a.shape}")
    # Channel first?
    if a.shape[0] in (1, 3) and a.shape[2] != 3:
        a = np.transpose(a, (1, 2, 0))
    # Scale if float
    if np.issubdtype(a.dtype, np.floating):
        a = (a * 255.0).round().clip(0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return a

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)

def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def compute_ee_error(pos, quat, tgt_pos, tgt_quat):
    pos_error = tgt_pos - pos
    quat_error = tgt_quat - quat  # simple elementwise diff, same as your writer
    return np.concatenate([pos_error, quat_error], axis=-1).astype(np.float32)

def load_robot_npz_info(npz_path: Path):
    z = dict(np.load(str(npz_path), allow_pickle=True))
    info = {
        "pos":       np.float32(z["replay_positions"]),
        "quat":      np.float32(z["replay_quats"]),
        "grip":      np.float32(z["gripper_state"]).reshape(-1),
        "joints":    np.float32(z["joint_positions"]),
        "base_t":    np.float32(z["translation"][0]).reshape(3,),
        "base_r":    np.float32(z["rotation"][0]).reshape(1,),
        "tgt_pos":   np.float32(z["target_positions"]),
        "tgt_quat":  np.float32(z["target_quats"]),
    }
    return info

def iter_video_frames(path: Path, target_hw: Tuple[int, int]):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = target_hw
            if (bgr.shape[0], bgr.shape[1]) != (h, w):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            yield bgr[..., ::-1]  # to RGB
    finally:
        cap.release()

def fetch_rlds_episode_fields(dataset: str, split: str, ep_index: int, mapping: List[Dict]):
    """Re-read RLDS fields for one episode using the dict-style mapping."""
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

# ---------- Verification ----------

def verify_episode(
    repo_id: str,
    out_root: Path,
    dataset_name: str,
    split: str,
    ep_index: int,
    robots: List[str],
    trg_root: Path,
    check_images: bool = True,
    check_rlds: bool = True,
    img_psnr_min: float = 30.0,     # tolerant; mp4 re-encode will add loss
    img_mae_max: float = 3.0,
    num_atol: float = 1e-5,
) -> Dict:
    cfg = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset_name]
    H, W = cfg["image_size"]

    # Load only this episode from your constructed dataset
    local_dir = Path(out_root) / repo_id
    if not local_dir.exists():
        raise FileNotFoundError(f"Local dataset not found at: {local_dir}. "
                                f"Did you pass the correct --out_root and --repo_id?")
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=str(local_dir),
        episodes=[ep_index],
        download_videos=False,
    )
    n = len(ds)  # frames in this episode only (thanks to episodes=[])  # noqa
    report = {"episode": ep_index, "frames": n, "keys": {}}

    # Original sources
    ep_dir = trg_root / dataset_name / split / str(ep_index)
    poses = np.load(ep_dir / f"end_effector_poses_{ep_index}.npy", allow_pickle=True)
    rlds_map = cfg["rlds_to_lerobot_mappings"]
    rlds = fetch_rlds_episode_fields(dataset_name, split, ep_index, rlds_map) if check_rlds else {}

    # Preload robot data
    original = {}
    for r in robots:
        info_npz = load_robot_npz_info(ep_dir / f"{r}_replay_info_{ep_index}.npz")
        frames = []
        if check_images:
            vid_path = ep_dir / f"{r}_overlay_{ep_index}_algo_final.mp4"
            frames = list(iter_video_frames(vid_path, (H, W)))
        original[r] = {"info": info_npz, "frames": frames}

    # Iterate frames (dataset contains only this episode)
    for j in range(n):
        sample = ds[j]  # dict of tensors/values for this frame

        # ---- Global features ----
        # observation.joints built from original robot (cfg["robot"])
        src_robot = cfg["robot"]
        want_j = np.concatenate(
            [
                np.asarray(poses[j]["joint_positions"][:ROBOT_JOINT_NUMBERS[src_robot]], dtype=np.float32),
                np.atleast_1d(np.asarray(poses[j]["gripper_state"], dtype=np.float32)),
            ],
            axis=0,
        )
        got_j = to_numpy(sample["observation.joints"]).astype(np.float32)
        diff = np.abs(got_j - want_j)
        report["keys"].setdefault("observation.joints", []).append(float(diff.max()))

        want_eep = np.concatenate(
            [np.asarray(poses[j]["position"], dtype=np.float32), np.asarray(poses[j]["quaternion"], dtype=np.float32)],
            axis=0,
        )
        got_eep = to_numpy(sample["observation.ee_pose"]).astype(np.float32)
        report["keys"].setdefault("observation.ee_pose", []).append(float(np.abs(got_eep - want_eep).max()))

        # ---- Per-robot features ----
        for r in robots:
            info = original[r]["info"]

            # joints + grip
            want_r_j = np.append(info["joints"][j], info["grip"][j]).astype(np.float32)
            got_r_j = to_numpy(sample[f"observation.{r}.joints"]).astype(np.float32)
            report["keys"].setdefault(f"observation.{r}.joints", []).append(float(np.abs(got_r_j - want_r_j).max()))

            # ee_pose
            want_r_eep = np.concatenate([info["pos"][j], info["quat"][j]], axis=0).astype(np.float32)
            got_r_eep = to_numpy(sample[f"observation.{r}.ee_pose"]).astype(np.float32)
            report["keys"].setdefault(f"observation.{r}.ee_pose", []).append(float(np.abs(got_r_eep - want_r_eep).max()))

            # base_t / base_r (episode constants)
            got_bp = to_numpy(sample[f"observation.{r}.base_position"]).astype(np.float32)
            got_br = to_numpy(sample[f"observation.{r}.base_orientation"]).astype(np.float32)
            report["keys"].setdefault(f"observation.{r}.base_position", []).append(float(np.abs(got_bp - info["base_t"]).max()))
            report["keys"].setdefault(f"observation.{r}.base_orientation", []).append(float(np.abs(got_br - info["base_r"]).max()))

            # ee_error
            want_err = compute_ee_error(info["pos"][j], info["quat"][j], info["tgt_pos"][j], info["tgt_quat"][j])
            got_err = to_numpy(sample[f"observation.{r}.ee_error"]).astype(np.float32)
            report["keys"].setdefault(f"observation.{r}.ee_error", []).append(float(np.abs(got_err - want_err).max()))

            # images (PSNR/MAE)
            if check_images and f"observation.images.{r}" in sample:
                got_img = chw_to_hwc_u8(to_numpy(sample[f"observation.images.{r}"]))
                want_img = original[r]["frames"][j]
                report["keys"].setdefault(f"observation.images.{r}.psnr", []).append(psnr(got_img, want_img))
                report["keys"].setdefault(f"observation.images.{r}.mae", []).append(mae(got_img, want_img))

        # ---- RLDS (non-image keys; images can be lossy) ----
        if check_rlds:
            for m in rlds_map:
                dst = m["lerobot_path"]; dtype = m["dtype"]
                if dtype == "video":
                    continue  # skip strict pixel equality for RLDS images
                want = rlds[dst][j]
                got = sample[dst]
                if dtype == "string":
                    ok = (got == want)
                    report["keys"].setdefault(dst, []).append(0.0 if ok else 1.0)
                else:
                    w = np.asarray(want)
                    g = to_numpy(got)
                    report["keys"].setdefault(dst, []).append(float(np.abs(g - w).max()))

    # Summaries
    summary = {"episode": ep_index, "frames": n, "failures": []}
    for key, vals in report["keys"].items():
        vals = np.asarray(vals, dtype=np.float32)
        if key.endswith(".psnr"):
            mn = float(vals.min()); ok = (mn >= img_psnr_min)
            summary[key] = {"min": mn, "ok": ok}
            if not ok: summary["failures"].append(key)
        elif key.endswith(".mae"):
            mx = float(vals.max()); ok = (mx <= img_mae_max)
            summary[key] = {"max": mx, "ok": ok}
            if not ok: summary["failures"].append(key)
        else:
            mx = float(vals.max()); ok = (mx <= num_atol)
            summary[key] = {"max_abs_err": mx, "ok": ok}
            if not ok: summary["failures"].append(key)
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=Path, required=True, help="Root dir used in build_dataset (same as --out_root there)")
    ap.add_argument("--repo_id", required=True, help="Repo id used when creating the dataset (folder name under out_root)")
    ap.add_argument("--dataset", required=True, help="Source dataset name (e.g., taco_play)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, nargs="+", required=True, help="Episode indices to verify")
    ap.add_argument("--robots", nargs="+", required=True, help="Robots you stored (e.g., widowX xarm7 sawyer ur5e)")
    ap.add_argument("--trg_root", type=Path, default=Path("/home/abinayadinesh/rovi_aug_extension_full"))
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--no_rlds", action="store_true")
    args = ap.parse_args()

    all_summaries = []
    for ep in args.episodes:
        s = verify_episode(
            repo_id=args.repo_id,
            out_root=args.out_root,
            dataset_name=args.dataset,
            split=args.split,
            ep_index=ep,
            robots=args.robots,
            trg_root=args.trg_root,
            check_images=not args.no_images,
            check_rlds=not args.no_rlds,
        )
        all_summaries.append(s)

    # Pretty print
    import json
    print(json.dumps(all_summaries, indent=2))

if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/lerobot_rovi_aug/verify_lerobot_dataset.py \
  --out_root /home/abinayadinesh/lerobot_dataset \
  --repo_id toto_train_0_100 \
  --dataset toto \
  --split train \
  --episodes 0 1 2 \
  --robots kuka_iiwa xarm7 sawyer ur5e kinova3 jaco google_robot widowX \
  --trg_root /home/abinayadinesh/rovi_aug_extension_full
'''