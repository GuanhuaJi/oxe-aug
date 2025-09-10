#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export augmented episodes to RLDS/TFDS on disk.

What this does
--------------
- For each episode index in [start, end), read:
  • Original RLDS fields (image/state/instruction) via TFDS
  • Your local per-robot overlay video (RGB frames) and *_replay_info_*.npz kinematics
- Emit an RLDS dataset with:
  steps:{
    is_first, is_last, is_terminal, reward, discount,
    language_instruction (str),
    observation:{
      image (orig RLDS frame),
      state (orig RLDS proprio),
      images{<robot>: overlay frame},
      <robot>{joints, ee_pose, base_position, base_orientation, ee_error} for each robot
    }
  }
- Saves with TFDS AdhocBuilder to a versioned folder so you can load it via
  `tfds.builder_from_directory(...).as_dataset(...)`.

Requires
--------
pip install tensorflow-datasets==4.*  numpy opencv-python tqdm

Notes
-----
- Uses file_format='array_record' for random-access friendly shards.
- Set env RLDS_STORAGE to switch away from GCS:
    export RLDS_STORAGE=/path/to/local/rlds/root
"""

from __future__ import annotations

import argparse
import logging
import os
os.environ["SVT_LOG"] = "0"
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Any

import cv2
import numpy as np
from tqdm import tqdm

LOG = logging.getLogger("export_rlds")

# ────────────────────────────────────────────────────────────────────────────────
# Config / constants (kept from your reference so loading logic is identical)
# ────────────────────────────────────────────────────────────────────────────────
rlds_to_lerobot = {
    "toto": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/observation/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "nyu_franka_play_dataset_converted_externally_to_rlds": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "utaustin_mutex": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "berkeley_autolab_ur5": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/observation/language_instruction", "natural_language_instruction"),
        ("/steps/observation/robot_state", "observation.state"),
    ],
    "kaist_nonprehensile_converted_externally_to_rlds": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/robot_state", "observation.state"),
    ],
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "fractal20220817_data": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/observation/language_instruction", "natural_language_instruction"),
        ("/steps/observation/base_pose_tool_reached", "observation.state"),
    ],
    "viola": [
        ("/steps/observation/agentview_rgb", "observation.images.image"),
        ("/steps/observation/natural_language_instruction", "natural_language_instruction"),
        ("/steps/observation/ee_states", "observation.state"),
    ],
    "taco_play": [
        ("/steps/observation/rgb_static", "observation.images.image"),
        ("/steps/observation/natural_language_instruction", "natural_language_instruction"),
        ("/steps/observation/robot_obs", "observation.state"),
    ],
    "bridge": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "ucsd_kitchen_dataset_converted_externally_to_rlds": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/end_effector_pose", "observation.state"),
    ],
    "jaco_play": [
        ("/steps/observation/rgb_static", "observation.images.image"),
        ("/steps/observation/natural_language_instruction", "natural_language_instruction"),
        ("/steps/observation/end_effector_cartesian_pos", "observation.state"),
    ],
    "austin_sailor_dataset_converted_externally_to_rlds": [
        ("/steps/observation/image", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/state", "observation.state"),
    ],
    "language_table": [
        ("/steps/observation/rgb", "observation.images.image"),
        ("/steps/language_instruction", "natural_language_instruction"),
        ("/steps/observation/effector_translation", "observation.state"),
    ],
}

STATE_DIM_LOOKUP = {
    "toto": 7,
    "nyu_franka_play_dataset_converted_externally_to_rlds": 13,
    "utaustin_mutex": 24,
    "berkeley_autolab_ur5": 15,
    "kaist_nonprehensile_converted_externally_to_rlds": 21,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 20,
    "fractal20220817_data": 7,
    "viola": 16,
    "taco_play": 15,
    "bridge": 7,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 21,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": 6,
    "jaco_play": 7,
    "austin_sailor_dataset_converted_externally_to_rlds": 8,
    "language_table": 7,
}

IMAGE_SIZE = {
    "berkeley_autolab_ur5": (480, 640),
    "kaist_nonprehensile_converted_externally_to_rlds": (480, 640),
    "toto": (480, 640),
    "taco_play": (150, 200),
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": (360, 640),
    "utaustin_mutex": (128, 128),
    "austin_sailor_dataset_converted_externally_to_rlds": (128, 128),
    "austin_buds_dataset_converted_externally_to_rlds": (128, 128),
    "nyu_franka_play_dataset_converted_externally_to_rlds": (128, 128),
    "viola": (224, 224),
    "bridge": (480, 640),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": (480, 640),
    "fractal20220817_data": (256, 320),
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": (224, 224),
    "jaco_play": (224, 224),
}

ROBOT_JOINT_NUMBERS = {
    "panda": 7,
    "ur5e": 6,
    "widowX": 6,
    "xarm7": 7,
    "jaco": 6,
    "kuka_iiwa": 7,
    "kinova3": 7,
    "sawyer": 7,
    "google_robot": 7,
}

# Default local root where your overlay videos & npz live
TRG_ROOT_DEFAULT = Path("/home/abinayadinesh/rovi_aug_extension_full")

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class VideoSource:
    path: Path
    target_hw: Tuple[int, int]

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
                yield bgr[..., ::-1]  # RGB uint8
        finally:
            cap.release()

def compute_ee_error(pos: np.ndarray, quat: np.ndarray, tgt_pos: np.ndarray, tgt_quat: np.ndarray) -> np.ndarray:
    """Simple position/quat difference; replace with a quaternion metric if desired."""
    pos_error = tgt_pos - pos
    quat_error = tgt_quat - quat
    return np.concatenate([pos_error, quat_error], axis=-1).astype(np.float32)

@dataclass
class EpisodeKinematics:
    pos: np.ndarray          # (T,3)
    quat: np.ndarray         # (T,4)
    grip: np.ndarray         # (T,)
    joints: np.ndarray       # (T,J)
    base_t: np.ndarray       # (3,)
    base_r: np.ndarray       # (1,)
    tgt_pos: Optional[np.ndarray] = None  # (T,3)
    tgt_quat: Optional[np.ndarray] = None # (T,4)

def load_robot_npz_info(npz_path: Path, robot: str) -> EpisodeKinematics:
    """Load and normalize one robot's kinematic NPZ."""
    info = dict(np.load(str(npz_path), allow_pickle=True))
    pos = np.float32(info["replay_positions"])
    quat = np.float32(info["replay_quats"])
    grip = np.float32(info["gripper_state"]).reshape(-1)
    jnts = np.float32(info["joint_positions"])
    base_t = np.float32(info["translation"])
    base_r = np.float32(info["rotation"])
    tgt_pos = np.float32(info["target_positions"])
    tgt_quat = np.float32(info["target_quats"])
    base_t = np.asarray(base_t[0], np.float32).reshape(3,)
    base_r = np.asarray(base_r[0], np.float32).reshape(1,)
    return EpisodeKinematics(
        pos=pos, quat=quat, grip=grip, joints=jnts, base_t=base_t, base_r=base_r,
        tgt_pos=tgt_pos, tgt_quat=tgt_quat
    )

def fetch_rlds_episode_fields(dataset: str, split: str, ep_index: int, mapping_pairs: List[Tuple[str, str]]) -> Dict[str, List[Any]]:
    """
    Pull one RLDS episode from TFDS (local or GCS), according to rlds_to_lerobot mapping.
    Returns a dict {dest_key: list-of-steps}.
    """
    import tensorflow_datasets as tfds

    storage = os.environ.get("RLDS_STORAGE", "gs://gresearch/robotics")
    key = f"{dataset}/0.1.0"

    if not hasattr(fetch_rlds_episode_fields, "_builders"):
        fetch_rlds_episode_fields._builders = {}
    builder = fetch_rlds_episode_fields._builders.get(key)
    if builder is None:
        builder = tfds.builder_from_directory(builder_dir=f"{storage}/{dataset}/0.1.0")
        fetch_rlds_episode_fields._builders[key] = builder

    split_spec = f"{split}[{ep_index}:{ep_index+1}]"
    ds = builder.as_dataset(split=split_spec, shuffle_files=False)

    out: Dict[str, List[Any]] = {dst: [] for (src, dst) in mapping_pairs}

    for episode in ds:  # single episode slice
        steps_ds = episode["steps"]
        for step in steps_ds:
            for src, dst in mapping_pairs:
                # Descend into the nested fields (e.g., steps/observation/image)
                parts = [p for p in src.strip("/").split("/") if p]
                cur = step
                for p in parts[1:]:
                    cur = cur[p]
                cur = cur.numpy() if hasattr(cur, "numpy") else cur
                if dst == "natural_language_instruction" and isinstance(cur, (bytes, bytearray, np.bytes_)):
                    cur = cur.decode("utf-8", errors="ignore")
                out[dst].append(cur)
        break
    return out

# ────────────────────────────────────────────────────────────────────────────────
# Episode preparation (reuses your local overlay/npz layout)
# ────────────────────────────────────────────────────────────────────────────────
def _prepare_episode(ep: int, dataset: str, split: str, robots: Sequence[str], h: int, w: int, trg_root: Path):
    """Decode overlays + kinematics for each robot and fetch original RLDS fields."""
    ep_dir = trg_root / dataset / split / str(ep)
    trg: Dict[str, Dict[str, Any]] = {}
    for robot in robots:
        video_path = ep_dir / f"{robot}_overlay_{ep}_algo_final.mp4"
        info_path  = ep_dir / f"{robot}_replay_info_{ep}.npz"
        vs = VideoSource(video_path, (h, w))
        info = load_robot_npz_info(info_path, robot)
        frames = [f for f in vs]
        trg[robot] = dict(frames=frames, info=info)

    # Original RLDS (image/state/instruction)
    rlds_pairs = rlds_to_lerobot.get(dataset, [])
    rlds_streams = fetch_rlds_episode_fields(dataset, split, ep, rlds_pairs)
    task = rlds_streams.get("natural_language_instruction", [""])[0] if len(rlds_streams.get("natural_language_instruction", [])) else ""

    # Build per-step dicts containing everything needed to emit RLDS later
    T = len(next(iter(trg.values()))["frames"])
    frames_list: List[Dict[str, Any]] = []
    for j in range(T):
        fd: Dict[str, Any] = {}

        # RLDS originals
        fd["language_instruction"] = rlds_streams.get("natural_language_instruction", [""] * T)[j]
        fd["orig_image"] = rlds_streams.get("observation.images.image", [None] * T)[j]
        fd["orig_state"] = rlds_streams.get("observation.state", [None] * T)[j]

        # Per-robot overlays + kinematics
        fd["robot_images"] = {}
        fd["robot_kin"] = {}
        for robot, blob in trg.items():
            info: EpisodeKinematics = blob["info"]
            frame_rgb = blob["frames"][j]
            ee_pose = np.concatenate([info.pos[j], info.quat[j]], axis=0).astype(np.float32)
            ee_error = compute_ee_error(info.pos[j], info.quat[j], info.tgt_pos[j], info.tgt_quat[j])
            fd["robot_images"][robot] = frame_rgb
            fd["robot_kin"][robot] = {
                "joints": np.append(info.joints[j], info.grip[j]).astype(np.float32),
                "ee_pose": ee_pose,
                "base_position": info.base_t.astype(np.float32),
                "base_orientation": info.base_r.astype(np.float32),
                "ee_error": ee_error,
            }
        frames_list.append(fd)

    return (ep, task, frames_list)

# ────────────────────────────────────────────────────────────────────────────────
# RLDS writer (TFDS AdhocBuilder)
# ────────────────────────────────────────────────────────────────────────────────
def _make_rlds_features(h: int, w: int, robots: Sequence[str], state_dim: int):
    import tensorflow_datasets as tfds
    # Per-robot kinematics block
    robot_blocks = {
        r: tfds.features.FeaturesDict({
            "joints":           tfds.features.Tensor(shape=(ROBOT_JOINT_NUMBERS[r] + 1,), dtype=np.float32),
            "ee_pose":          tfds.features.Tensor(shape=(7,), dtype=np.float32),
            "base_position":    tfds.features.Tensor(shape=(3,), dtype=np.float32),
            "base_orientation": tfds.features.Tensor(shape=(1,), dtype=np.float32),
            "ee_error":         tfds.features.Tensor(shape=(7,), dtype=np.float32),
        })
        for r in robots
    }
    # Per-robot images
    images_dict = {r: tfds.features.Image(shape=(h, w, 3)) for r in robots}

    step_features = tfds.features.FeaturesDict({
        # RLDS step flags + simple reward scheme (1 on final step)
        "is_first":    tfds.features.Scalar(dtype=np.bool_),
        "is_last":     tfds.features.Scalar(dtype=np.bool_),
        "is_terminal": tfds.features.Scalar(dtype=np.bool_),
        "reward":      tfds.features.Scalar(dtype=np.float32),
        "discount":    tfds.features.Scalar(dtype=np.float32),

        "language_instruction": tfds.features.Text(),

        "observation": tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(h, w, 3)),   # original RLDS rgb
            "state": tfds.features.Tensor(shape=(state_dim,), dtype=np.float32),

            "images": tfds.features.FeaturesDict(images_dict),  # overlay frames per robot
            **robot_blocks,                                     # per-robot kinematics
        }),
    })

    features = tfds.features.FeaturesDict({
        "steps": tfds.features.Dataset(step_features),  # nested tf.data.Dataset
        "episode_metadata": tfds.features.FeaturesDict({
            "task": tfds.features.Text(),
            "episode_index": tfds.features.Scalar(dtype=np.int32),
        }),
    })
    return features

def _episode_to_rlds_steps(frames_list: List[Dict[str, Any]], robots: Sequence[str]) -> Iterable[Dict[str, Any]]:
    T = len(frames_list)
    for j, fd in enumerate(frames_list):
        step: Dict[str, Any] = {}
        step["is_first"]    = (j == 0)
        step["is_last"]     = (j == T - 1)
        step["is_terminal"] = (j == T - 1)
        step["reward"]      = np.float32(1.0 if j == T - 1 else 0.0)
        step["discount"]    = np.float32(1.0)

        step["language_instruction"] = fd.get("language_instruction", "")

        obs = {
            "image": fd["orig_image"],
            "state": np.asarray(fd["orig_state"], dtype=np.float32) if fd["orig_state"] is not None else np.zeros((1,), np.float32),
            "images": {},
        }
        for r in robots:
            obs["images"][r] = fd["robot_images"][r]
            obs[r] = fd["robot_kin"][r]

        step["observation"] = obs
        yield step

def export_split_as_rlds(
    dataset: str,
    split: str,
    robots: Sequence[str],
    start: int,
    end: int,
    out_root: Path,
    trg_root: Path,
    version: str = "0.1.0",
    name: Optional[str] = None,
    file_format: str = "array_record",
):
    """
    Persist episodes [start, end) as a TFDS RLDS dataset on disk.
    The resulting dataset can be loaded with tfds.builder_from_directory(...).as_dataset(...).
    """
    import tensorflow_datasets as tfds

    h, w = IMAGE_SIZE[dataset]
    state_dim = STATE_DIM_LOOKUP[dataset]
    features = _make_rlds_features(h, w, robots, state_dim)

    def _episode_iter():
        for ep in range(start, end):
            try:
                ep_id, task, frames_list = _prepare_episode(ep, dataset, split, robots, h, w, trg_root)
            except Exception as e:
                LOG.error("Episode %d failed: %s", ep, e)
                continue
            yield {
                "episode_metadata": {"task": str(task), "episode_index": int(ep_id)},
                "steps": _episode_to_rlds_steps(frames_list, robots),
            }

    if name is None:
        name = f"{dataset}_converted_externally_to_rlds"

    builder = tfds.dataset_builders.store_as_tfds_dataset(
        name=name,
        version=version,
        features=features,
        split_datasets={split: _episode_iter()},
        data_dir=str(out_root),
        description=f"RLDS export with overlays/kinematics for robots {list(robots)}",
        homepage="https://github.com/google-research/rlds",
        citation=(
            "@misc{ramos2021rlds, title={RLDS: an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning}, "
            "author={Sabela Ramos et al.}, year={2021}, eprint={2111.02767}, archivePrefix={arXiv}}"
        ),
        file_format=file_format,  # random-access friendly
    )
    builder.download_and_prepare()
    LOG.info("[RLDS] Wrote dataset under: %s/%s/%s", str(out_root), name, version)

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--split", default="train", type=str)
    p.add_argument("--robots", nargs="+", required=True, help="e.g. widowX xarm7 sawyer ur5e")
    p.add_argument("--start", default=0, type=int)
    p.add_argument("--end", required=True, type=int, help="exclusive")
    p.add_argument("--out_root", required=True, type=str, help="TFDS data dir for output RLDS")
    p.add_argument("--trg_root", default=str(TRG_ROOT_DEFAULT), type=str, help="Root of local overlays/npz")
    p.add_argument("--version", default="0.1.0", type=str)
    p.add_argument("--name", default=None, type=str, help="TFDS dataset name (default: <dataset>_converted_externally_to_rlds)")
    p.add_argument("--log_level", default="INFO", type=str)
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # OpenCV + determinism friendliness
    cv2.setNumThreads(0)

    export_split_as_rlds(
        dataset=args.dataset.strip(),
        split=args.split.strip(),
        robots=args.robots,
        start=args.start,
        end=args.end,
        out_root=Path(args.out_root),
        trg_root=Path(args.trg_root),
        version=args.version,
        name=args.name,
    )

if __name__ == "__main__":
    main()


'''
python /home/guanhuaji/lerobot_rovi_aug/convert_to_rlds.py \
  --dataset utaustin_mutex --split train \
  --robots xarm7 sawyer ur5e \
  --start 0 --end 100 \
  --out_root /home/guanhuaji/lerobot_datasets_rlds

'''