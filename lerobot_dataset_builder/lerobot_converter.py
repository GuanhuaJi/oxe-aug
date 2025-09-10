from __future__ import annotations

import argparse
import logging
import os
os.environ["SVT_LOG"] = "0"
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from config import RLDS_TO_LEROBOT_DATASET_CONFIGS, ROBOT_JOINT_NUMBERS

LOG = logging.getLogger("build_lerobot")

# Global set at runtime (in main) and in each worker via initializer
TRG_ROOT: Optional[Path] = None


def _init_worker(trg_root_str: str):
    """Initializer for ProcessPoolExecutor workers so they can see TRG_ROOT."""
    global TRG_ROOT
    TRG_ROOT = Path(trg_root_str)
    # OpenCV tends to misbehave with fork/spawn and many threads.
    cv2.setNumThreads(0)


@dataclass
class VideoSource:
    path: Path
    target_hw: Tuple[int, int]

    def __iter__(self):
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
                yield bgr[..., ::-1]  # RGB
        finally:
            cap.release()


def compute_ee_error(pos, quat, tgt_pos, tgt_quat):
    """Compute end-effector error (pos_error + quat_error)."""
    pos_error = tgt_pos - pos
    quat_error = tgt_quat - quat  # placeholder metric; replace if needed
    return np.concatenate([pos_error, quat_error], axis=-1).astype(np.float32)


@dataclass
class EpisodeKinematics:
    pos: np.ndarray          # (T,3)
    quat: np.ndarray         # (T,4)
    grip: np.ndarray         # (T,) or (T,1)
    joints: np.ndarray       # (T,J)
    base_t: np.ndarray       # (3,)  episode-constant
    base_r: np.ndarray       # (1,)  episode-constant
    tgt_pos: Optional[np.ndarray] = None  # (T,3)
    tgt_quat: Optional[np.ndarray] = None # (T,4)


def load_robot_npz_info(npz_path, robot, T_hint=None):
    """Load and normalize one robot's kinematic NPZ."""
    info = dict(np.load(npz_path, allow_pickle=True))

    pos = np.float32(info["replay_positions"])
    quat = np.float32(info["replay_quats"])
    grip = np.float32(info["gripper_state"]).reshape(-1)
    jnts = np.float32(info["joint_positions"])
    base_t = np.float32(info["translation"])
    base_r = np.float32(info["rotation"])
    tgt_pos = np.float32(info["target_positions"])
    tgt_quat = np.float32(info["target_quats"])
    _ = ROBOT_JOINT_NUMBERS[robot]  # num_joints not needed directly here

    base_t = np.asarray(base_t[0], np.float32).reshape(3,)
    base_r = np.asarray(base_r[0], np.float32).reshape(1,)

    return EpisodeKinematics(
        pos=pos, quat=quat, grip=grip, joints=jnts, base_t=base_t, base_r=base_r,
        tgt_pos=tgt_pos, tgt_quat=tgt_quat
    )


def fetch_rlds_episode_images(dataset, split, ep_index):
    """
    Load exactly one RLDS episode using TFDS and return RGB frames list.
    """
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(
        builder_dir=f"gs://gresearch/robotics/{dataset}/0.1.0"
    )
    split_spec = f"{split}[{ep_index}:{ep_index+1}]"
    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        shuffle_seed=None,
    )
    ds = builder.as_dataset(split=split_spec, shuffle_files=False, read_config=read_config)

    frames = []
    for episode in ds:  # at most one
        steps_ds = episode["steps"]
        for step in steps_ds:
            img = step["observation"]["image"].numpy()
            frames.append(np.asarray(img))
        break
    return frames


def fetch_rlds_episode_fields(dataset, split, ep_index, mapping):
    import tensorflow_datasets as tfds
    key = f"{dataset}/0.1.0"
    if not hasattr(fetch_rlds_episode_fields, "_builders"):
        fetch_rlds_episode_fields._builders = {}
    builder = fetch_rlds_episode_fields._builders.get(key)
    if builder is None:
        builder = tfds.builder_from_directory(
            builder_dir=f"{os.environ.get('RLDS_STORAGE','gs://gresearch/robotics')}/{dataset}/0.1.0"
        )
        fetch_rlds_episode_fields._builders[key] = builder

    split_spec = f"{split}[{ep_index}:{ep_index+1}]"
    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        shuffle_seed=None,
    )
    ds = builder.as_dataset(split=split_spec, shuffle_files=False, read_config=read_config)

    out = {m["lerobot_path"]: [] for m in mapping}
    for episode in ds:
        steps_ds = episode["steps"]
        for step in steps_ds:
            for m in mapping:
                src, dst, dtype = m["rlds_path"], m["lerobot_path"], m["dtype"]
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


def build_feature_schema(dataset, h, w, robots):
    """Build feature schema with new structure."""
    feats = {}

    # robot-specific features
    for robot in robots:
        feats.update({
            f"observation.{robot}.joints":           {"dtype": "float32", "shape": (ROBOT_JOINT_NUMBERS[robot] + 1,)},
            f"observation.{robot}.ee_pose":          {"dtype": "float32", "shape": (7,)},
            f"observation.{robot}.base_position":    {"dtype": "float32", "shape": (3,)},
            f"observation.{robot}.base_orientation": {"dtype": "float32", "shape": (1,)},
            f"observation.{robot}.ee_error":         {"dtype": "float32", "shape": (7,)},
            f"observation.images.{robot}":           {"dtype": "video",   "shape": (h, w, 3)},
        })

    for m in RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["rlds_to_lerobot_mappings"]:
        dst   = m["lerobot_path"]
        dtype = m["dtype"]
        shape = m["shape"]
        feats[dst] = {"dtype": dtype, "shape": shape}

    feats["observation.joints"]  = {"dtype": "float32", "shape": (ROBOT_JOINT_NUMBERS[RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["robot"]] + 1,)}
    feats["observation.ee_pose"] = {"dtype": "float32", "shape": (7,)}
    return feats


def _require_trg_root() -> Path:
    if TRG_ROOT is None:
        raise RuntimeError("TRG_ROOT is not initialized. Pass --trg_root or set $TRG_ROOT.")
    return TRG_ROOT


def _prepare_episode(ep, dataset, split, robots, h, w):
    cv2.setNumThreads(0)
    trg_root = _require_trg_root()

    trg = {}
    for robot in robots:
        ep_dir = trg_root / dataset / split / str(ep)
        video_path = ep_dir / f"{robot}_overlay_{ep}_algo_final.mp4"
        info_path  = ep_dir / f"{robot}_replay_info_{ep}.npz"
        vs = VideoSource(video_path, (h, w))
        info = load_robot_npz_info(info_path, robot)
        trg[robot] = dict(vs=vs, info=info)

    rlds_mappings = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["rlds_to_lerobot_mappings"]
    rlds_streams = fetch_rlds_episode_fields(dataset, split, ep, rlds_mappings)

    for r, blob in trg.items():
        frames = [f for f in blob["vs"]]
        blob["frames"] = frames

    task = rlds_streams["natural_language_instruction"][0]
    poses = np.load(trg_root / dataset / split / str(ep) / f"end_effector_poses_{ep}.npy", allow_pickle=True)

    frames_list = []
    for j in range(len(trg[robots[0]]["frames"])):
        fd = {}
        fd["observation.joints"] = np.concatenate(
            [
                poses[j]["joint_positions"][:ROBOT_JOINT_NUMBERS[RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["robot"]]],
                np.atleast_1d(poses[j]["gripper_state"])
            ],
            axis=0
        ).astype(np.float32)
        fd["observation.ee_pose"] = np.concatenate(
            [np.asarray(poses[j]["position"], dtype=np.float32),
             np.asarray(poses[j]["quaternion"], dtype=np.float32)],
            axis=0
        ).astype(np.float32)

        for robot, blob in trg.items():
            info = blob["info"]
            ee_pose = np.concatenate([info.pos[j], info.quat[j]], axis=0)
            ee_error = compute_ee_error(info.pos[j], info.quat[j], info.tgt_pos[j], info.tgt_quat[j])
            fd[f"observation.{robot}.joints"] = np.append(info.joints[j], info.grip[j]).astype(np.float32)
            fd[f"observation.{robot}.ee_pose"]          = ee_pose
            fd[f"observation.{robot}.base_position"]    = info.base_t
            fd[f"observation.{robot}.base_orientation"] = info.base_r
            fd[f"observation.{robot}.ee_error"]         = ee_error
            fd[f"observation.images.{robot}"]           = blob["frames"][j]

        for dst_key, seq in rlds_streams.items():
            fd[dst_key] = seq[j]

        frames_list.append(fd)

    return (ep, task, frames_list)


def _precheck_lengths(dataset, split, robots, start, end, h, w):
    trg_root = _require_trg_root()
    for ep in range(start, end):
        ep_dir = trg_root / dataset / split / str(ep)
        lens = []
        for robot in robots:
            info_path = ep_dir / f"{robot}_replay_info_{ep}.npz"
            info = load_robot_npz_info(info_path, robot)
            lens.append(info.pos.shape[0])

            video_path = ep_dir / f"{robot}_overlay_{ep}_algo_final.mp4"
            vs = VideoSource(video_path, (h, w))
            vcount = sum(1 for _ in vs)
            lens.append(vcount)

        pairs = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["rlds_to_lerobot_mappings"]
        if pairs:
            rlds = fetch_rlds_episode_fields(dataset, split, ep, pairs)
            # Keep this simple—if there is an image stream, check its length
            for m in pairs:
                if m["rlds_path"].rstrip("/").endswith("/image"):
                    lens.append(len(rlds[m["lerobot_path"]]))
                    break

        if len(set(lens)) != 1:
            raise ValueError(f"Length mismatch in episode {ep}: {lens}")


def build_dataset(
    dataset: str,
    split: str,
    robots: Sequence[str],
    start: int,
    end: int,
    fps: int,
    out_root: Path,
    repo_id: Optional[str] = None,
    num_workers: int = 1,
):
    dataset = dataset.strip()
    split = split.strip()
    h, w = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["image_size"]

    feats = build_feature_schema(dataset, h, w, robots)

    if repo_id is None:
        repo_id = f"{dataset}_{split}_{start}_{end}"
    out_dir = out_root / repo_id
    if out_dir.exists():
        LOG.warning("Output dir %s exists; appending episodes.", out_dir)

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(out_dir),
        robot_type="mixed",
        fps=fps,
        features=feats,
    )

    ep_range = list(range(start, end))
    LOG.info("Building with num_workers=%d over %d episodes", num_workers, len(ep_range))

    if num_workers <= 1:
        for ep in tqdm(ep_range, desc="episodes", unit="ep"):
            res = _prepare_episode(ep, dataset, split, robots, h, w)
            if res is None:
                continue
            ep_id, task, frames_list = res
            for fd in frames_list:
                ds.add_frame(fd, task=task)
            ds.save_episode()
    else:
        # Ensure workers receive the same TRG_ROOT
        trg_root = _require_trg_root()
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(trg_root),),
        ) as ex:
            futs = [ex.submit(_prepare_episode, ep, dataset, split, robots, h, w) for ep in ep_range]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="episodes", unit="ep"):
                res = fut.result()
                if res is None:
                    continue
                ep_id, task, frames_list = res
                for fd in frames_list:
                    ds.add_frame(fd, task=task)
                ds.save_episode()

    LOG.info("Done. Dataset stored in %s", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="taco_play", type=str)
    p.add_argument("--split", default="train", type=str)
    p.add_argument("--robots", nargs="+", default=["widowX", "xarm7", "sawyer", "ur5e"])
    p.add_argument("--start", default=0, type=int)
    p.add_argument("--end", default=100, type=int, help="exclusive")
    p.add_argument("--fps", default=30, type=int)
    p.add_argument("--out_root", default="/home/abinayadinesh/lerobot_dataset", type=str)
    p.add_argument("--log_level", default="INFO", type=str)
    p.add_argument("--num_workers", type=int, default=1,
                   help="Episodes prepared in parallel; writes stay sequential.")
    p.add_argument(
        "--trg_root",
        default=None,
        type=str,
        help="Root containing per-episode NPZ/MP4 (or set $TRG_ROOT). Layout: {trg_root}/{dataset}/{split}/{ep}/",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # OpenCV + multiprocessing friendliness
    mp.set_start_method("spawn", force=True)

    # Resolve TRG_ROOT from CLI or environment
    trg_root_str = args.trg_root or os.environ.get("TRG_ROOT")
    if not trg_root_str:
        p.error("--trg_root not provided and $TRG_ROOT is not set")
    global TRG_ROOT
    TRG_ROOT = Path(trg_root_str)

    build_dataset(
        dataset=args.dataset,
        split=args.split,
        robots=args.robots,
        start=args.start,
        end=args.end,
        fps=args.fps,
        out_root=Path(args.out_root),
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
