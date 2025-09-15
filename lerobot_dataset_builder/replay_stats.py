#!/usr/bin/env python3
"""
Replay-quality summary using *selective* loading (no video decode).

This script uses your SelectiveLeRobotDataset/HF table to read ONLY the
*.ee_error columns (plus episode_index) and computes, per repo and per robot:

  • episodes_all_ok_pct: % of episodes where ALL *active* timesteps pass threshold
  • frames_ok_pct:       % of *active* timesteps whose error passes threshold

Definitions
-----------
- "Active" frame for a robot: at least one element of that robot's ee_error vector
  has absolute value > --active-eps (default 1e-12). Episodes with zero active
  frames for a robot are skipped for that robot (not counted in ep_total).

- Error reduction:
    --metric l2   : ||e[:l2_dims]||2      (default l2_dims=3 → position-only)
    --metric linf : max(|e_i|)            (uses all dims)
    --metric l1   : sum(|e_i|)
    --metric mean : mean(|e_i|)

Assumptions
-----------
- episode_data_index["to"] is END-EXCLUSIVE.
- Multiple robots may be active in the same episode; we attribute that episode to
  each active robot independently. The "ALL" row is a simple sum over robots.

Examples
--------
python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo toto_train_0_100 toto_train_100_200 \
  --threshold 0.01 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo kaist_nonprehensile_converted_externally_to_rlds_train_0_201 \
  --threshold 0.01 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo iamlab_cmu_pickup_insert_converted_externally_to_rlds_train_0_100 \
  --threshold 0.03 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo language_table_train_0_10 \
  --threshold 0.04 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo taco_play_train_0_1000 \
  --threshold 0.03 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo berkeley_autolab_ur5_train_100_200 \
  --threshold 0.02 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo austin_sailor_dataset_converted_externally_to_rlds_train_0_240 \
  --threshold 0.02 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo austin_buds_dataset_converted_externally_to_rlds_train_0_50 \
  --threshold 0.02 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo viola_train_0_135 \
  --threshold 0.02 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo utokyo_xarm_pick_and_place_converted_externally_to_rlds_train_0_92 \
  --threshold 0.02 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo ucsd_kitchen_dataset_converted_externally_to_rlds_train_0_150 \
  --threshold 0.02 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo bridge_train_0_1000 \
  --threshold 0.01 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo fractal20220817_data_train_1_11 \
  --threshold 0.01 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo nyu_franka_play_dataset_converted_externally_to_rlds_train_0_365 \
  --threshold 0.01 --metric l2

python /home/guanhuaji/lerobot_rovi_aug/replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --repo jaco_play_train_0_10 \
  --threshold 0.01 --metric l2

# Only episodes 0..49 and 120..150
python replay_stats_selective.py \
  --root /.../lerobot_dataset \
  --repo language_table_train_0_10 \
  --episodes 0:50,120:151 \
  --threshold 0.01 --metric linf --csv stats.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import csv

import numpy as np
from tqdm import tqdm

from selective_lerobot_dataloader import (
    _load_info,
    discover_keys,
    SelectiveLeRobotDataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ───────────────────────────── helpers ─────────────────────────────

def parse_episodes_spec(spec, total_eps):
    """Parse '0:10,15,20:25' -> sorted unique list of ints in [0,total_eps)."""
    if not spec:
        return None
    out = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            a, b = token.split(":")
            a = int(a) if a else 0
            b = int(b) if b else total_eps  # end exclusive
            for i in range(a, b):
                if 0 <= i < total_eps:
                    out.add(i)
        else:
            i = int(token)
            if 0 <= i < total_eps:
                out.add(i)
    return sorted(out)


def key_to_robot(key: str) -> str:
    # observation.<robot>.ee_error
    parts = key.split(".")
    return parts[1] if len(parts) >= 3 else "global"


def reduce_err_batch(errs_mat: np.ndarray, metric: str, l2_dims: int | None):
    """
    errs_mat: np.ndarray [T, D]
    returns:  np.ndarray [T]
    """
    if metric == "l2":
        if l2_dims is not None:
            errs_mat = errs_mat[..., :l2_dims]
        return np.linalg.norm(errs_mat, axis=-1)
    if metric in ("linf", "max"):
        return np.max(np.abs(errs_mat), axis=-1)
    if metric == "l1":
        return np.sum(np.abs(errs_mat), axis=-1)
    if metric == "mean":
        return np.mean(np.abs(errs_mat), axis=-1)
    raise ValueError(f"Unknown metric: {metric}")


def as_pct(num, den):
    return 0.0 if den == 0 else 100.0 * float(num) / float(den)


def print_summary(repo_id, stats, overall, skipped_eps, total_eps, threshold, metric, l2_dims):
    print(f"\n=== {repo_id} ===")
    print(f"metric={metric}  threshold={threshold}  l2_dims={l2_dims if metric=='l2' else '-'}")
    print(f"episodes total in repo: {total_eps} | skipped (no active frames for any robot): {skipped_eps}\n")

    header = f"{'robot':16s} {'ep_total':>9s} {'ep_all_ok':>9s} {'ep_all_ok%':>10s}   {'frames':>9s} {'frames_ok':>10s} {'frames_ok%':>10s}"
    print(header)
    print("-" * len(header))
    for robot, r in sorted(stats.items()):
        ep_total = r["episodes_total"]
        ep_all_ok = r["episodes_all_ok"]
        fr_total = r["frames_total"]
        fr_ok = r["frames_ok"]
        print(f"{robot:16s} {ep_total:9d} {ep_all_ok:9d} {as_pct(ep_all_ok, ep_total):10.2f}   {fr_total:9d} {fr_ok:10d} {as_pct(fr_ok, fr_total):10.2f}")

    ep_total = overall["episodes_total"]
    ep_all_ok = overall["episodes_all_ok"]
    fr_total = overall["frames_total"]
    fr_ok = overall["frames_ok"]
    print("-" * len(header))
    print(f"{'ALL':16s} {ep_total:9d} {ep_all_ok:9d} {as_pct(ep_all_ok, ep_total):10.2f}   {fr_total:9d} {fr_ok:10d} {as_pct(fr_ok, fr_total):10.2f}")
    print()


def maybe_write_csv(path, repo_id, stats, overall, threshold, metric, l2_dims):
    if not path:
        return
    path = Path(path)
    rows = []
    for robot, r in stats.items():
        rows.append({
            "repo_id": repo_id,
            "robot": robot,
            "metric": metric,
            "l2_dims": l2_dims if metric == "l2" else "",
            "threshold": threshold,
            "episodes_total": r["episodes_total"],
            "episodes_all_ok": r["episodes_all_ok"],
            "episodes_all_ok_pct": as_pct(r["episodes_all_ok"], r["episodes_total"]),
            "frames_total": r["frames_total"],
            "frames_ok": r["frames_ok"],
            "frames_ok_pct": as_pct(r["frames_ok"], r["frames_total"]),
        })
    rows.append({
        "repo_id": repo_id,
        "robot": "ALL",
        "metric": metric,
        "l2_dims": l2_dims if metric == "l2" else "",
        "threshold": threshold,
        "episodes_total": overall["episodes_total"],
        "episodes_all_ok": overall["episodes_all_ok"],
        "episodes_all_ok_pct": as_pct(overall["episodes_all_ok"], overall["episodes_total"]),
        "frames_total": overall["frames_total"],
        "frames_ok": overall["frames_ok"],
        "frames_ok_pct": as_pct(overall["frames_ok"], overall["frames_total"]),
    })
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


# ─────────────────────────── core analysis ───────────────────────────

def analyze_repo(
    root: str,
    repo_id: str,
    episodes_list,
    threshold: float,
    metric: str,
    l2_dims: int,
    active_eps: float,
):
    """
    Read only *.ee_error columns (plus episode_index), treat 'to' as EXCLUSIVE,
    and compute per-robot replay stats. Multiple robots can be active in the
    same episode; each active robot contributes independently.
    """
    info = _load_info(root, repo_id)
    err_keys_all = discover_keys(info, suffix=".ee_error")
    if not err_keys_all:
        raise RuntimeError(f"No *.ee_error features found in {repo_id}")

    # Build selective dataset (keeps only episode_index + err_keys; use numpy format)
    sel_ds = SelectiveLeRobotDataset(
        root=root, repo_id=repo_id, keys=err_keys_all, episodes=None, fmt="numpy"
    )
    hf = sel_ds.hf  # numpy-formatted HF Dataset

    # Episode ranges via LeRobotDataset (no video download/decoding)
    base = LeRobotDataset(
        repo_id=repo_id,
        root=str(Path(root) / repo_id),
        download_videos=False,
        video_backend="pyav",
    )
    ep_from = base.episode_data_index["from"]
    ep_to = base.episode_data_index["to"]
    total_eps = len(ep_from)

    episodes = range(total_eps) if episodes_list is None else episodes_list

    # Determine which error columns actually exist in this split/table
    present_err_keys = [k for k in err_keys_all if k in hf.column_names]
    if not present_err_keys:
        raise RuntimeError(f"No *.ee_error columns present in the table for {repo_id}")

    stats = defaultdict(lambda: dict(episodes_total=0, episodes_all_ok=0, frames_total=0, frames_ok=0))
    overall = dict(episodes_total=0, episodes_all_ok=0, frames_total=0, frames_ok=0)

    skipped_eps = 0

    for ep in tqdm(episodes, desc=f"[{repo_id}] episodes"):
        s = int(ep_from[ep].item())
        end_excl = int(ep_to[ep].item())  # EXCLUSIVE
        if end_excl <= s:
            skipped_eps += 1
            continue

        # defensive clamp if metadata slightly disagrees with table length
        end_excl = min(end_excl, len(hf))
        rows = hf.select(range(s, end_excl))

        any_robot_active = False

        for key in present_err_keys:
            # stack episode's error vectors for this robot: [T, D]
            arr_list = rows[key]
            if not len(arr_list):
                continue
            mat = np.stack(arr_list)  # [T, D]

            # Determine active frames for this robot in this episode
            # A frame is "active" if any component magnitude > active_eps
            active_mask = (np.abs(mat) > active_eps).any(axis=-1)
            active_count = int(active_mask.sum())
            if active_count == 0:
                continue  # robot not active in this episode

            any_robot_active = True
            robot = key_to_robot(key)

            # Reduce error per (active) frame
            if metric == "l2" and l2_dims is not None:
                errs = reduce_err_batch(mat[active_mask, :], metric, l2_dims=l2_dims)
            else:
                errs = reduce_err_batch(mat[active_mask, :], metric, l2_dims=None)

            ok_mask = errs <= threshold
            frames_total = int(errs.shape[0])
            frames_ok = int(ok_mask.sum())
            all_ok = bool(ok_mask.all())

            r = stats[robot]
            r["episodes_total"] += 1
            r["frames_total"] += frames_total
            r["frames_ok"] += frames_ok
            if all_ok:
                r["episodes_all_ok"] += 1

            overall["episodes_total"] += 1
            overall["frames_total"] += frames_total
            overall["frames_ok"] += frames_ok
            if all_ok:
                overall["episodes_all_ok"] += 1

        if not any_robot_active:
            skipped_eps += 1

    return stats, overall, skipped_eps, total_eps


# ──────────────────────────────── CLI ────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Parent directory that contains <repo_id>/")
    p.add_argument("--repo", nargs="+", required=True, help="One or more repo_id folders under --root")
    p.add_argument("--threshold", type=float, required=True, help="Scalar threshold against reduced ee_error")
    p.add_argument(
        "--metric",
        default="l2",
        choices=["l2", "linf", "l1", "mean"],
        help="Reduction to apply to ee_error vector",
    )
    p.add_argument("--l2-dims", type=int, default=3,
                   help="When metric=l2, apply L2 to the first N dims (default: 3)")
    p.add_argument("--episodes", default="", help="Subset episodes like '0:50,120:151'. Empty = all.")
    p.add_argument("--active-eps", type=float, default=1e-12,
                   help="Frame considered 'active' for a robot if any |err| > this (default: 1e-12)")
    p.add_argument("--csv", default="", help="Optional: path to append CSV results")
    args = p.parse_args()

    for repo_id in args.repo:
        info = _load_info(args.root, repo_id)
        total_eps = int(info.get("total_episodes", 0)) or None
        if total_eps is None or total_eps <= 0:
            base = LeRobotDataset(
                repo_id=repo_id,
                root=str(Path(args.root) / repo_id),
                download_videos=False,
                video_backend="pyav",
            )
            total_eps = len(base.episode_data_index["from"])

        episodes_list = parse_episodes_spec(args.episodes, total_eps) if args.episodes else None

        stats, overall, skipped, total_eps_true = analyze_repo(
            root=args.root,
            repo_id=repo_id,
            episodes_list=episodes_list,
            threshold=args.threshold,
            metric=args.metric,
            l2_dims=args.l2_dims,
            active_eps=args.active_eps,
        )
        print_summary(repo_id, stats, overall, skipped, total_eps_true, args.threshold, args.metric, args.l2_dims)
        maybe_write_csv(args.csv, repo_id, stats, overall, args.threshold, args.metric, args.l2_dims)


if __name__ == "__main__":
    main()
