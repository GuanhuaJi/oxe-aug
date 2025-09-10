#!/usr/bin/env python3
# compare_two_loaders.py
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

# LeRobot (local)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from config import RLDS_TO_LEROBOT_DATASET_CONFIGS, ROBOT_JOINT_NUMBERS
from lerobot_dataloader import LocalLeRobotDataset, make_dataloader
from local_dataloader import LocalFolderDataset, make_local_folder_dataloader

# ---------- tiny utils ----------
RLDS_IMAGE_KEY = "observation.images.image"

def import_local_folder_dataset(module_name="local_dataloader", module_path: Optional[str]=None):
    """
    Import LocalFolderDataset from your local_dataloader.py.
    If module_path is provided, import by absolute path (robust).
    """
    if module_path:
        p = Path(module_path)
        if not p.exists():
            raise FileNotFoundError(f"--local_dataloader_py not found: {p}")
        spec = importlib.util.spec_from_file_location(module_name, str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod.LocalFolderDataset
    # else: try regular import on PYTHONPATH
    mod = __import__(module_name, fromlist=["LocalFolderDataset"])
    return getattr(mod, "LocalFolderDataset")

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_hwc_u8(x) -> np.ndarray:
    """
    Canonicalize an image to (H,W,3) uint8 in [0,255].
    Accepts CHW/HWC, float/uint8 tensors or arrays.
    """
    a = to_numpy(x)
    if a.ndim == 3:
        # CHW?
        if a.shape[0] in (1, 3) and a.shape[-1] != 3:
            a = np.transpose(a, (1, 2, 0))
    if a.dtype.kind in ("f", "c"):
        a = (a * 255.0).round().clip(0, 255).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    # ensure contiguous positive strides
    if a.strides and any(s < 0 for s in a.strides):
        a = a.copy()
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    # 8-bit images -> MAX=255
    return 20.0 * np.log10(255.0) - 10.0 * np.log10(mse)

# ---------- main comparison ----------

def canonicalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in sample.items():
        if k.startswith("observation.images."):
            out[k] = to_hwc_u8(v)  # HWC uint8 contiguous
        elif isinstance(v, (bytes, bytearray, np.bytes_)):
            out[k] = v.decode("utf-8", errors="ignore")
        elif isinstance(v, str):
            out[k] = v
        else:
            arr = to_numpy(v)
            # --- shape normalization for base pose/orientation ---
            if k.endswith(".base_position"):
                arr = np.asarray(arr, dtype=np.float32).reshape(3,)
            elif k.endswith(".base_orientation"):
                arr = np.asarray(arr, dtype=np.float32).reshape(1,)
            else:
                try:
                    arr = np.asarray(arr, dtype=np.float32)
                except Exception:
                    out[k] = v
                    continue
            out[k] = arr
    return out

def compare_values(k: str, a: Any, b: Any, num_atol=1e-5, img_psnr_min=30.0, img_mae_max=3.0):
    # strings
    if isinstance(a, str) and isinstance(b, str):
        return (a == b, "equal", float(a == b))

    # arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        # images
        if k.startswith("observation.images."):
            if a.shape != b.shape:
                return (False, "shape_eq", 0.0)
            p = psnr(a, b)
            mae = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            # ---- special policy: RLDS image comes from lossy mp4 in LeRobot ----
            if k == RLDS_IMAGE_KEY:
                ok = (p >= img_psnr_min)  # ignore MAE or use a looser cap
                return (ok, "psnr", float(p))
            # default policy for robot overlays
            ok = (p >= img_psnr_min) and (mae <= img_mae_max)
            return (ok, "psnr/mae", float(p))

        # numeric tensors
        if a.shape != b.shape:
            return (False, "shape_eq", 0.0)
        diff = float(np.max(np.abs(a - b))) if a.size and b.size else 0.0
        ok = bool(np.allclose(a, b, atol=num_atol, rtol=0.0))
        return (ok, "max_abs_err", diff)

    # everything else
    return (a == b, "equal", float(a == b))

def iter_indices_from_episodes(ds_len: int, episode_mask: Optional[List[int]], ds_obj: Any) -> List[int]:
    """
    For LeRobotDataset we can pass episodes=[...] directly.
    For LocalFolderDataset we already flattened. For safety, just return range(ds_len).
    """
    return list(range(ds_len))

def main():
    ap = argparse.ArgumentParser()
    # LeRobot local dataset
    ap.add_argument("--lerobot_root", type=Path, required=True, help="Parent dir that contains <repo_id>/")
    ap.add_argument("--repo_id", required=True, help="Folder name under --lerobot_root")
    # Local-folder dataset
    ap.add_argument("--local_dataloader_py", type=str, default=None, help="Path to your local_dataloader.py (optional if importable)")
    ap.add_argument("--trg_root", type=Path, required=True, help="TRG_ROOT (parent of dataset/split/ep)")
    # Common selection
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--robots", nargs="+", required=True)
    # Comparison knobs
    ap.add_argument("--limit", type=int, default=200, help="Max frames to compare")
    ap.add_argument("--img_psnr_min", type=float, default=30.0)
    ap.add_argument("--img_mae_max", type=float, default=3.0)
    ap.add_argument("--num_atol", type=float, default=1e-5)
    # LeRobot video backend (safer defaults)
    ap.add_argument("--video_backend", default="pyav", choices=["pyav", "opencv", "torchcodec"])
    args = ap.parse_args()

    # ---- instantiate datasets (not DataLoaders; we want 1:1 index) ----
    # A) LeRobot (local)
    # Fail fast if local exists
    info_json = (args.lerobot_root / args.repo_id / "meta" / "info.json")
    if not info_json.exists():
        raise FileNotFoundError(f"Missing {info_json}. Check --lerobot_root/--repo_id.")
    # Avoid any Hub calls
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    ds_le = LeRobotDataset(
        repo_id=args.repo_id,
        root=str(args.lerobot_root / args.repo_id),  # points to the local folder
        episodes=args.episodes,          # restrict to target episodes
        download_videos=True,
        video_backend=args.video_backend,
    )

    # B) Local-folder dataset
    LocalFolderDataset = import_local_folder_dataset(module_path=args.local_dataloader_py)
    H, W = RLDS_TO_LEROBOT_DATASET_CONFIGS[args.dataset]["image_size"]

    ds_local = LocalFolderDataset(
        trg_root=str(args.trg_root),
        dataset=args.dataset,
        split=args.split,
        robots=args.robots,
        episodes=args.episodes,
        image_size=(H, W),
        include_rlds=True,       # compare RLDS fields too (requires TFDS access)
        preload_videos=True,
        strict_lengths=True,
    )

    n = min(len(ds_le), len(ds_local), args.limit)
    if n == 0:
        raise RuntimeError("No frames to compare (check episodes, lengths).")

    # ---- compare ----
    per_key_metrics: Dict[str, List[float]] = {}
    per_key_ok: Dict[str, List[bool]] = {}
    failures: List[Tuple[int, str, str]] = []  # (idx, key, reason)

    for i in range(n):
        a_raw = ds_le[i]
        b_raw = ds_local[i]
        a = canonicalize_sample(a_raw)
        b = canonicalize_sample(b_raw)

        # Compare intersection of keys (skip keys missing on either side)
        keys = sorted(set(a.keys()).intersection(b.keys()))
        for k in keys:
            ok, metric, val = compare_values(k, a[k], b[k], num_atol=args.num_atol,
                                             img_psnr_min=args.img_psnr_min, img_mae_max=args.img_mae_max)
            per_key_metrics.setdefault(f"{k}:{metric}", []).append(val)
            per_key_ok.setdefault(k, []).append(ok)
            if not ok:
                failures.append((i, k, metric))

    # ---- summarize ----
    print(f"Compared {n} frame(s) across episodes {args.episodes}")
    bad_keys = []
    for k, oks in per_key_ok.items():
        ok_rate = 100.0 * (sum(oks) / len(oks))
        if not all(oks):
            bad_keys.append(k)
        # Print one metric summary per key if we recorded it
        metric_keys = [mk for mk in per_key_metrics.keys() if mk.startswith(k + ":")]
        if metric_keys:
            mk = metric_keys[0]
            vals = np.asarray(per_key_metrics[mk], dtype=np.float64)
            agg = f"min={vals.min():.4g} max={vals.max():.4g} mean={vals.mean():.4g}"
            print(f"[{ok_rate:5.1f}% OK] {mk} -> {agg}")
        else:
            print(f"[{ok_rate:5.1f}% OK] {k}")

    if failures:
        print("\nSample mismatches (up to first 20):")
        for i, (idx, k, metric) in enumerate(failures[:20], 1):
            print(f" {i:2d}) idx {idx} key={k} metric={metric}")

    all_ok = not failures
    print("\nRESULT:", "PASS ✅" if all_ok else "FAIL ❌")
    if not all_ok:
        sys.exit(2)

if __name__ == "__main__":
    main()

'''
python /home/guanhuaji/lerobot_rovi_aug/check_two_datasets.py \
  --lerobot_root /home/abinayadinesh/lerobot_dataset \
  --repo_id toto_train_0_100 \
  --trg_root /home/abinayadinesh/rovi_aug_extension_full \
  --dataset toto \
  --split train \
  --episodes 0 1 \
  --robots widowX xarm7 sawyer ur5e \
  --limit 200 \
  --video_backend pyav

'''