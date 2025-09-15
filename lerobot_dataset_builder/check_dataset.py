#!/usr/bin/env python3
# check_lengths.py
import argparse, csv
from pathlib import Path
import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm

MISSING = -1  # sentinel for missing/unreadable files

def count_frames_fast(video_path: Path) -> int:
    if not video_path.exists():
        return MISSING
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return MISSING
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n if n > 0 else MISSING

def npz_len(npz_path: Path) -> int:
    if not npz_path.exists():
        return MISSING
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            if "replay_positions" not in d:
                return MISSING
            n = int(d["replay_positions"].shape[0])
            return n if n > 0 else MISSING
    except Exception:
        # Corrupt/partial npz counts as mismatch
        return MISSING

def mode_length(lengths: list[int]) -> int:
    """Most common length among valid (>=0) entries; if tie, choose the larger."""
    vals = [x for x in lengths if x >= 0]
    if not vals:
        return MISSING
    counts = Counter(vals).most_common()
    top = counts[0][1]
    candidates = [v for v, c in counts if c == top]
    return max(candidates)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, help="Root dir containing episode subfolders 0,1,2,...")
    ap.add_argument("--start", type=int, help="start episode (inclusive)")
    ap.add_argument("--end", type=int, help="end episode (exclusive)")
    ap.add_argument("--robots", nargs="+", required=True, help="robot names, e.g. xarm7 sawyer ur5e ...")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root)
    if args.start is None and args.end is None:
        episodes = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    else:
        episodes = list(range(args.start, args.end))

    rows = []

    for ep in tqdm(episodes):
        ep_dir = root / str(ep)

        per_robot: dict[str, tuple[int, int]] = {}
        all_lengths: list[int] = []

        for r in args.robots:
            vid = ep_dir / f"{r}_overlay_{ep}_algo_final.mp4"
            npz = ep_dir / f"{r}_replay_info_{ep}.npz"

            vlen = count_frames_fast(vid)
            nlen = npz_len(npz)

            per_robot[r] = (vlen, nlen)

            # Only contribute valid lengths to the expected calculation
            if vlen >= 0:
                all_lengths.append(vlen)
            if nlen >= 0:
                all_lengths.append(nlen)

        expected = mode_length(all_lengths)

        for r, (vlen, nlen) in per_robot.items():
            # Count as mismatch if either file is missing/unreadable OR doesn't match expected
            is_missing = (vlen == MISSING) or (nlen == MISSING)
            wrong_len = (expected != MISSING) and (vlen != expected or nlen != expected)
            if is_missing or wrong_len:
                rows.append({
                    "episode": ep,
                    "robot": r,
                    "expected_len": expected,
                    "video_len": vlen,
                    "npz_len": nlen,
                })

    # Derive dataset/split names robustly
    root_parts = Path(args.root).parts
    dataset = root_parts[-2] if len(root_parts) >= 2 else Path(args.root).name
    split = root_parts[-1]

    out_path = Path(args.out) / f"{dataset}_{split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "robot", "expected_len", "video_len", "npz_len"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} mismatches → {out_path}")

if __name__ == "__main__":
    main()


'''
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/nyu_franka_play_dataset_converted_externally_to_rlds/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/toto/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/viola/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/austin_buds_dataset_converted_externally_to_rlds/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/austin_sailor_dataset_converted_externally_to_rlds/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/berkeley_autolab_ur5/train --robots panda sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/iamlab_cmu_pickup_insert_converted_externally_to_rlds/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/taco_play/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/ucsd_kitchen_dataset_converted_externally_to_rlds/train --robots panda sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/utaustin_mutex/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/utokyo_xarm_pick_and_place_converted_externally_to_rlds/train --robots ur5e sawyer panda jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/bridge/train --robots widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/fractal20220817_data/train --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot --out /home/guanhuaji/lerobot_rovi_aug/check --start 0 --end 87212
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/jaco_play/train --robots ur5e sawyer xarm7 panda widowX kuka_iiwa kinova3 google_robot --out /home/guanhuaji/lerobot_rovi_aug/check


python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/bridge/test --robots panda ur5e sawyer panda jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/nyu_franka_play_dataset_converted_externally_to_rlds/val --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check --start 0 --end 1


python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/toto/test --robots ur5e sawyer xarm7 jaco kuka_iiwa kinova3 google_robot widowX --out /home/guanhuaji/lerobot_rovi_aug/check
python /home/guanhuaji/lerobot_rovi_aug/check_dataset.py --root /home/abinayadinesh/rovi_aug_extension_full/language_table/train --robots ur5e sawyer jaco kuka_iiwa kinova3 google_robot panda xarm7 --out /home/guanhuaji/lerobot_rovi_aug/check --start 0 --end 442226


'''