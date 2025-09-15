from __future__ import annotations

import argparse
import logging
import os
os.environ["SVT_LOG"] = "0"
import sys
from pathlib import Path
import time
from contextlib import contextmanager
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from config import RLDS_TO_LEROBOT_DATASET_CONFIGS, ROBOT_JOINT_NUMBERS

LOG = logging.getLogger("build_lerobot")

# Globals
TRG_ROOT = None
TIME_ON = False  # set by --time


# ------------------------- timing helpers -------------------------

class PhaseStats:
    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name, dt):
        self.totals[name] += dt
        self.counts[name] += 1

    def items_sorted(self):
        return sorted(self.totals.items(), key=lambda kv: kv[1], reverse=True)

    def total(self):
        return sum(self.totals.values())


@contextmanager
def timed(stats, name):
    if not TIME_ON:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add(name, time.perf_counter() - t0)


def _print_summary(stats):
    if not TIME_ON:
        return
    total = stats.total()
    if total <= 0:
        LOG.info("No timing recorded.")
        return
    LOG.info("\n===== Timing summary (all episodes) =====")
    LOG.info("{:<28s} {:>10s} {:>9s} {:>9s} {:>12s}".format(
        "phase", "total(s)", "percent", "count", "avg(ms)"
    ))
    for name, sec in stats.items_sorted():
        cnt = stats.counts.get(name, 0)
        avg_ms = (sec / max(cnt, 1)) * 1000.0
        LOG.info("{:<28s} {:>10.3f} {:>8.1f}% {:>9d} {:>12.2f}".format(
            name, sec, 100.0 * sec / total, cnt, avg_ms
        ))
    LOG.info("=========================================\n")


# --------------------------- helpers ---------------------------

class VideoSource:
    def __init__(self, path, target_hw):
        self.path = Path(path)
        self.target_hw = target_hw  # (h, w)

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
                yield bgr[..., ::-1]  # RGB uint8
        finally:
            cap.release()


class EpisodeKinematics:
    def __init__(self, pos, quat, grip, joints, base_t, base_r, tgt_pos, tgt_quat):
        self.pos = pos
        self.quat = quat
        self.grip = grip
        self.joints = joints
        self.base_t = base_t
        self.base_r = base_r
        self.tgt_pos = tgt_pos
        self.tgt_quat = tgt_quat


def compute_ee_error(pos, quat, tgt_pos, tgt_quat):
    pos_error = tgt_pos - pos
    quat_error = tgt_quat - quat  # placeholder metric
    return np.concatenate([pos_error, quat_error], axis=-1).astype(np.float32)


def load_robot_npz_info(npz_path, robot, T_hint=None):
    info = dict(np.load(npz_path, allow_pickle=True))
    pos = np.float32(info["replay_positions"])
    quat = np.float32(info["replay_quats"])
    grip = np.float32(info["gripper_state"]).reshape(-1)
    jnts = np.float32(info["joint_positions"])
    base_t = np.float32(info["translation"])
    base_r = np.float32(info["rotation"])
    tgt_pos = np.float32(info["target_positions"])
    tgt_quat = np.float32(info["target_quats"])
    _ = ROBOT_JOINT_NUMBERS[robot]  # not used directly

    base_t = np.asarray(base_t[0], np.float32).reshape(3,)
    base_r = np.asarray(base_r[0], np.float32).reshape(1,)
    return EpisodeKinematics(pos, quat, grip, jnts, base_t, base_r, tgt_pos, tgt_quat)


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
    read_config = tfds.ReadConfig(interleave_cycle_length=1, shuffle_seed=None)
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
    feats = {}
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
        feats[m["lerobot_path"]] = {"dtype": m["dtype"], "shape": m["shape"]}
    feats["observation.joints"]  = {"dtype": "float32", "shape": (ROBOT_JOINT_NUMBERS[RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["robot"]] + 1,)}
    feats["observation.ee_pose"] = {"dtype": "float32", "shape": (7,)}
    return feats


def _require_trg_root():
    if TRG_ROOT is None:
        raise RuntimeError("TRG_ROOT is not initialized. Pass --trg_root or set $TRG_ROOT.")
    return TRG_ROOT


# ---------------------------- core pipeline ----------------------------

def _prepare_episode(ep, dataset, split, robots, h, w, stats):
    cv2.setNumThreads(0)
    trg_root = _require_trg_root()

    trg = {}
    for robot in robots:
        ep_dir = Path(trg_root) / dataset / split / str(ep)
        video_path = ep_dir / f"{robot}_overlay_{ep}_algo_final.mp4"
        info_path  = ep_dir / f"{robot}_replay_info_{ep}.npz"

        with timed(stats, f"load_npz[{robot}]"):
            info = load_robot_npz_info(info_path, robot)

        with timed(stats, f"video_decode[{robot}]"):
            frames = [f for f in VideoSource(video_path, (h, w))]

        trg[robot] = dict(frames=frames, info=info)

    rlds_mappings = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["rlds_to_lerobot_mappings"]
    with timed(stats, "rlds_fetch"):
        rlds_streams = fetch_rlds_episode_fields(dataset, split, ep, rlds_mappings)

    task = rlds_streams.get("natural_language_instruction", [""])[0]

    if dataset in ["bridge", "fractal20220817_data", "language_table"]:
        source_info_path = Path(trg_root) / dataset / split / str(ep) / f"{RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]['robot']}_replay_info_{ep}.npz"
        source_info = load_robot_npz_info(source_info_path, RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]['robot'])
        poses_mode = "source_info"
    else:
        poses = np.load(Path(trg_root) / dataset / split / str(ep) / f"end_effector_poses_{ep}.npy", allow_pickle=True)
        poses_mode = "poses_file"

    with timed(stats, "build_frames_list"):
        frames_list = []
        T = len(trg[robots[0]]["frames"])
        for j in range(T):
            fd = {}

            if poses_mode == "source_info":
                fd["observation.joints"] = np.concatenate(
                    [
                        source_info.joints[j][:ROBOT_JOINT_NUMBERS[RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["robot"]]],
                        np.atleast_1d(source_info.grip[j])
                    ],
                    axis=0
                ).astype(np.float32)
                fd["observation.ee_pose"] = np.concatenate(
                    [np.asarray(source_info.pos[j], dtype=np.float32),
                     np.asarray(source_info.quat[j], dtype=np.float32)],
                    axis=0
                ).astype(np.float32)
            else:
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
                fd[f"observation.{robot}.joints"]           = np.append(info.joints[j], info.grip[j]).astype(np.float32)
                fd[f"observation.{robot}.ee_pose"]          = ee_pose
                fd[f"observation.{robot}.base_position"]    = info.base_t
                fd[f"observation.{robot}.base_orientation"] = info.base_r
                fd[f"observation.{robot}.ee_error"]         = ee_error
                # single-process, no async race; direct handoff is fine
                fd[f"observation.images.{robot}"]           = blob["frames"][j]

            for dst_key, seq in rlds_streams.items():
                fd[dst_key] = seq[j]

            frames_list.append(fd)

    return (ep, task, frames_list)


def build_dataset(dataset, split, robots, start, end, fps, out_root, repo_id=None, enable_time=False):
    stats = PhaseStats()
    h, w = RLDS_TO_LEROBOT_DATASET_CONFIGS[dataset]["image_size"]

    feats = build_feature_schema(dataset, h, w, robots)

    if repo_id is None:
        repo_id = f"{dataset}_{split}_{start}_{end}"
    out_dir = Path(out_root) / repo_id
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
    LOG.info("Building sequentially over %d episodes", len(ep_range))

    for ep in tqdm(ep_range, desc="episodes", unit="ep"):
        ep_id, task, frames_list = _prepare_episode(ep, dataset, split, robots, h, w, stats)

        for fd in frames_list:
            with timed(stats, "add_frame"):
                ds.add_frame(fd, task=task)

        with timed(stats, "save_episode"):
            ds.save_episode()

    _print_summary(stats)
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
    p.add_argument("--trg_root", default=None, type=str,
                   help="Root containing per-episode NPZ/MP4. Layout: {trg_root}/{dataset}/{split}/{ep}/")
    p.add_argument("--time", action="store_true", help="Enable timing summary.")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # OpenCV + multiprocessing friendliness (single process anyway)
    cv2.setNumThreads(0)

    # Resolve TRG_ROOT
    trg_root_str = args.trg_root or os.environ.get("TRG_ROOT")
    if not trg_root_str:
        p.error("--trg_root not provided and $TRG_ROOT is not set")
    global TRG_ROOT, TIME_ON
    TRG_ROOT = Path(trg_root_str)
    TIME_ON = bool(args.time)

    build_dataset(
        dataset=args.dataset,
        split=args.split,
        robots=args.robots,
        start=args.start,
        end=args.end,
        fps=args.fps,
        out_root=args.out_root,
        enable_time=TIME_ON,
    )


if __name__ == "__main__":
    main()





'''
python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset ucsd_kitchen_dataset_converted_externally_to_rlds --split train --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa panda --start 0 --end 150 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset utaustin_mutex --split train --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 --start 0 --end 1500 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset kaist_nonprehensile_converted_externally_to_rlds --split train --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 --start 0 --end 201 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset berkeley_autolab_ur5 --split test --robots widowX sawyer panda google_robot jaco kinova3 kuka_iiwa xarm7 --start 0 --end 104 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset toto --split test --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 --start 0 --end 101 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset utokyo_xarm_pick_and_place_converted_externally_to_rlds --split train --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa panda --start 0 --end 92 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset utokyo_xarm_pick_and_place_converted_externally_to_rlds --split train --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa panda --start 0 --end 10 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter.py --dataset taco_play --split test --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 --start 0 --end 361 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter_fast.py --dataset toto \
    --split train --robots widowX sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 \
        --start 0 --end 2 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset \
            --trg_root /home/abinayadinesh/rovi_aug_extension_full --num_workers 4 --img_writer_threads 16 --img_writer_procs 4

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter_fast.py --dataset bridge \
    --split train --robots panda sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 \
        --start 0 --end 100 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset \
            --trg_root /home/abinayadinesh/rovi_aug_extension_full --num_workers 4 --img_writer_threads 16 --img_writer_procs 4

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter_fast.py --dataset bridge \
    --split train --robots panda sawyer ur5e google_robot jaco kinova3 kuka_iiwa xarm7 \
        --start 0 --end 3 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset \
            --trg_root /home/abinayadinesh/rovi_aug_extension_full

python /home/guanhuaji/lerobot_rovi_aug/lerobot_converter_fast.py --dataset fractal20220817_data \
    --split train --robots panda sawyer ur5e jaco kinova3 kuka_iiwa xarm7 \
        --start 0 --end 10 --fps 30 --out_root /home/abinayadinesh/lerobot_dataset \
            --trg_root /home/abinayadinesh/rovi_aug_extension_full
'''