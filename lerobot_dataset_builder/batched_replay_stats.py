#!/usr/bin/env python3
"""
Replay-quality summary across all repos sharing a prefix (local-only, video-free).

What it does
------------
• Discovers all repos under --root whose folder names start with --prefix
• Loads ONLY parquet tables (no videos, no LeRobotDataset) using 🤗 Datasets
• Computes per-repo, per-robot stats:
    - episodes_all_ok_pct: % of episodes where ALL *active* frames pass threshold
    - frames_ok_pct:       % of *active* frames that pass threshold
    - 4-bin histogram over error magnitudes:
        [0,0.01), [0.01,0.02), [0.02,0.03), [0.03, +inf)
• Prints repo summaries + a GLOBAL summary across all repos
• Appends a CSV with counts and percentages per bin, per robot, per repo,
  plus a per-repo "ALL" row and a global "__ALL_REPOS__/ALL" row
• Optional: saves a GLOBAL histogram PNG (--hist-png). Uses a headless backend.

Assumptions
-----------
• Parquet files live under <repo_id>/data/chunk-*/episode_*.parquet
• The table has 'episode_index' (int) and one or more '*.ee_error' vector columns
• L2 reduction uses first --l2-dims dims (default 3). 'linf'/'l1'/'mean' use all dims.
• A frame is "active" for a robot if any(|ee_error|) > --active-eps.

Examples
--------
python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix bridge \
  --threshold 0.01 --metric l2 \
  --csv bridge_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix toto \
  --threshold 0.01 --metric l2 \
  --csv toto_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix jaco_play \
  --threshold 0.01 --metric l2 \
  --csv jaco_play_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix taco_play \
  --threshold 0.01 --metric l2 \
  --csv taco_play_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix berkeley_autolab_ur5 \
  --threshold 0.01 --metric l2 \
  --csv berkeley_autolab_ur5_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix ucsd_kitchen_dataset_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv ucsd_kitchen_dataset_converted_externally_to_rlds_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix utokyo_xarm_pick_and_place_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv utokyo_xarm_pick_and_place_converted_externally_to_rlds_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix kaist_nonprehensile_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv kaist_nonprehensile_converted_externally_to_rlds_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix austin_buds_dataset_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv austin_buds_dataset_converted_externally_to_rlds_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix utaustin_mutex \
  --threshold 0.01 --metric l2 \
  --csv utaustin_mutex_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix austin_sailor_dataset_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv austin_sailor_dataset_converted_externally_to_rlds_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix iamlab_cmu_pickup_insert_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv iamlab_cmu_pickup_insert_converted_externally_to_rlds_stats.csv --outdir /home/guanhuaji/dataset_stats

python /home/guanhuaji/lerobot_rovi_aug/batched_replay_stats.py \
  --root /home/abinayadinesh/lerobot_dataset \
  --prefix viola \
  --threshold 0.01 --metric l2 \
  --csv viola_stats.csv --outdir /home/guanhuaji/dataset_stats
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit(
        "This script requires 🤗 Datasets. Install with: pip install -U datasets"
    ) from e

# 4 fixed bins (edges) and labels
BIN_EDGES = np.array([0.0, 0.01, 0.02, 0.03, np.inf], dtype=float)
BIN_LABELS = ["0-0.01", "0.01-0.02", "0.02-0.03", ">=0.03"]


# ─────────────────────────── discovery & utils ───────────────────────────

def find_repo_ids(root: Path, prefix: str) -> List[str]:
    """List repo folder names in `root` that start with `prefix`."""
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            out.append(p.name)
    return out


def glob_parquet_files(repo_dir: Path) -> List[str]:
    """Return sorted list of parquet paths under <repo>/data/ (recursive)."""
    data_dir = repo_dir / "data"
    files = sorted(str(p) for p in data_dir.rglob("*.parquet"))
    return files


def load_hf_table_from_parquet(parquet_files: List[str]):
    """Load a 🤗 Datasets Dataset from a list of parquet files (local)."""
    if not parquet_files:
        raise RuntimeError("No parquet files found.")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    return ds


def present_error_keys(column_names: List[str]) -> List[str]:
    """Return all columns that end with '.ee_error'."""
    return sorted([c for c in column_names if c.endswith(".ee_error")])


def key_to_robot(key: str) -> str:
    # observation.<robot>.ee_error
    parts = key.split(".")
    return parts[1] if len(parts) >= 3 else "global"


def as_pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else 100.0 * float(num) / float(den)


def reduce_err_batch(errs_mat: np.ndarray, metric: str, l2_dims: int | None) -> np.ndarray:
    """errs_mat: [T, D] → per-frame scalar errors [T]."""
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


def print_repo_summary(repo_id: str, stats: Dict[str, dict], overall: dict,
                       skipped_eps: int, total_eps: int,
                       threshold: float, metric: str, l2_dims: int):
    print(f"\n=== {repo_id} ===")
    print(f"metric={metric}  threshold={threshold}  l2_dims={l2_dims if metric=='l2' else '-'}")
    print(f"episodes in repo (distinct episode_index): {total_eps} | skipped (no active frames for any robot): {skipped_eps}\n")

    header = f"{'robot':16s} {'ep_total':>9s} {'ep_all_ok':>9s} {'ep_all_ok%':>10s}   {'frames':>9s} {'frames_ok':>10s} {'frames_ok%':>10s}"
    print(header)
    print("-" * len(header))
    for robot, r in sorted(stats.items()):
        ep_total = r["episodes_total"]
        ep_all_ok = r["episodes_all_ok"]
        fr_total = r["frames_total"]
        fr_ok = r["frames_ok"]
        print(f"{robot:16s} {ep_total:9d} {ep_all_ok:9d} {as_pct(ep_all_ok, ep_total):10.2f}   {fr_total:9d} {fr_ok:10d} {as_pct(fr_ok, fr_total):10.2f}")

    ep_total = int(overall["episodes_total"])
    ep_all_ok = int(overall["episodes_all_ok"])
    fr_total = int(overall["frames_total"])
    fr_ok = int(overall["frames_ok"])
    print("-" * len(header))
    print(f"{'ALL':16s} {ep_total:9d} {ep_all_ok:9d} {as_pct(ep_all_ok, ep_total):10.2f}   {fr_total:9d} {fr_ok:10d} {as_pct(fr_ok, fr_total):10.2f}")
    print()

    # Per-repo histogram (overall)
    counts = overall["bin_counts"]
    total = int(counts.sum())
    if total > 0:
        print("Per-repo frame error histogram:")
        for lab, c in zip(BIN_LABELS, counts):
            print(f"  {lab:>8s}: {int(c):10d}  ({as_pct(int(c), total):6.2f}%)")
        print()


def write_csv_rows(csv_path: Path, rows: List[dict]):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_episode_bounds(episode_index_col: np.ndarray) -> List[Tuple[int, int, int]]:
    """From full 'episode_index' column → list of (episode_id, start, end_exclusive)."""
    epi = np.asarray(episode_index_col).reshape(-1)
    if epi.size == 0:
        return []
    change = np.nonzero(np.diff(epi) != 0)[0] + 1
    bounds = np.concatenate(([0], change, [epi.size]))
    out: List[Tuple[int, int, int]] = []
    for i in range(len(bounds) - 1):
        s = int(bounds[i])
        e = int(bounds[i + 1])  # exclusive
        ep_id = int(epi[s])
        out.append((ep_id, s, e))
    return out


# ───────────────────────────── core analysis ─────────────────────────────

def analyze_repo_local_parquet(
    repo_dir: Path,
    threshold: float,
    metric: str,
    l2_dims: int,
    active_eps: float,
    episodes_filter: List[int] | None,
):
    """
    Local-only analysis using parquet files + 🤗 Datasets.
    Returns (stats_by_robot, overall_dict, skipped_eps, total_eps_distinct).
    """
    parquet_files = glob_parquet_files(repo_dir)
    hf = load_hf_table_from_parquet(parquet_files)

    # Keep only columns we need: episode_index + all *.ee_error
    cols = hf.column_names
    err_keys = present_error_keys(cols)
    if not err_keys:
        raise RuntimeError(f"No *.ee_error columns present in table for {repo_dir.name}")

    keep = ["episode_index"] + err_keys
    drop = [c for c in cols if c not in keep]
    if drop:
        hf = hf.remove_columns(drop)  # projection
    hf = hf.with_format("numpy", columns=keep)   # selective materialization

    # Compute episode ranges from the column itself
    epi_col = np.asarray(hf["episode_index"]).reshape(-1)
    ep_bounds = compute_episode_bounds(epi_col)
    total_eps = len(ep_bounds)

    # Optional filter by episode ids
    if episodes_filter is not None:
        keep_set = set(int(x) for x in episodes_filter)
        ep_bounds = [(eid, s, e) for (eid, s, e) in ep_bounds if eid in keep_set]

    stats = defaultdict(lambda: dict(
        episodes_total=0, episodes_all_ok=0, frames_total=0, frames_ok=0,
        bin_counts=np.zeros(4, dtype=np.int64)
    ))
    overall = dict(
        episodes_total=0, episodes_all_ok=0, frames_total=0, frames_ok=0,
        bin_counts=np.zeros(4, dtype=np.int64)
    )
    skipped_eps = 0

    for (eid, s, e) in tqdm(ep_bounds, desc=f"[{repo_dir.name}] episodes"):
        if e <= s:
            skipped_eps += 1
            continue

        rows = hf.select(range(s, e))
        any_robot_active = False

        for key in err_keys:
            if key not in rows.column_names:
                continue
            arrs = rows[key]
            if not len(arrs):
                continue
            mat = np.stack(arrs)  # [T, D]

            # Determine active frames for this robot
            active_mask = (np.abs(mat) > active_eps).any(axis=-1)
            if not active_mask.any():
                continue

            any_robot_active = True
            robot = key_to_robot(key)

            # Reduce error for active frames
            errs_active = reduce_err_batch(
                mat[active_mask, :], metric, l2_dims=(l2_dims if metric == "l2" else None)
            )

            ok_mask = errs_active <= threshold
            frames_total = int(errs_active.shape[0])
            frames_ok = int(ok_mask.sum())
            all_ok = bool(ok_mask.all())

            # Bin counts
            bin_counts = np.histogram(errs_active, bins=BIN_EDGES)[0].astype(np.int64)

            r = stats[robot]
            r["episodes_total"] += 1
            r["frames_total"] += frames_total
            r["frames_ok"] += frames_ok
            r["bin_counts"] += bin_counts
            if all_ok:
                r["episodes_all_ok"] += 1

            overall["episodes_total"] += 1
            overall["frames_total"] += frames_total
            overall["frames_ok"] += frames_ok
            overall["bin_counts"] += bin_counts
            if all_ok:
                overall["episodes_all_ok"] += 1

        if not any_robot_active:
            skipped_eps += 1

    return stats, overall, skipped_eps, total_eps


# ──────────────────────────────── plotting I/O ────────────────────────────────

def _sanitize_component(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def save_hist(counts: np.ndarray, out_path: Path, title: str) -> bool:
    """Headless bar chart; returns True if saved (non-empty)."""
    total = int(np.sum(counts))
    if total <= 0:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-GUI backend
        import matplotlib.pyplot as plt
        plt.ioff()

        percents = [as_pct(int(c), total) for c in counts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["[0,0.01)", "[0.01,0.02)", "[0.02,0.03)", "[0.03,∞)"], percents)
        ax.set_ylabel("Percentage of frames (%)")
        ax.set_xlabel("Error magnitude bins")
        ax.set_title(title)
        for i, v in enumerate(percents):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] Could not save histogram to {out_path}: {e}")
        return False


def save_contact_sheet(png_paths: List[Path], out_path: Path, pad: int = 12) -> None:
    """
    Create a single PNG contact sheet from a list of PNGs (keeps sizes; centers each).
    """
    if not png_paths:
        return
    try:
        from PIL import Image  # pip install pillow
    except Exception as e:
        print("[WARN] Pillow not installed; skipping concatenated sheet. Install with: pip install pillow")
        return

    import math
    ims = [Image.open(p) for p in png_paths]
    try:
        cell_w = max(im.width for im in ims)
        cell_h = max(im.height for im in ims)
        n = len(ims)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

        sheet_w = pad + cols * (cell_w + pad)
        sheet_h = pad + rows * (cell_h + pad)
        sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))

        for i, im in enumerate(ims):
            r, c = divmod(i, cols)
            x = pad + c * (cell_w + pad) + (cell_w - im.width) // 2
            y = pad + r * (cell_h + pad) + (cell_h - im.height) // 2
            sheet.paste(im.convert("RGB"), (x, y))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path)
        print(f"Saved concatenated histogram sheet to: {out_path}")
    finally:
        for im in ims:
            try:
                im.close()
            except Exception:
                pass


# ──────────────────────────────── main ────────────────────────────────

def parse_episodes_spec(spec: str) -> List[int] | None:
    """Parse '0:10,15,20:25' → sorted unique episode IDs. Empty → None."""
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
            b = int(b) if b else 10**12
            for i in range(a, b):
                out.add(i)
        else:
            out.add(int(token))
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Folder containing all repo subfolders")
    p.add_argument("--prefix", required=True, help="Dataset prefix (e.g., 'bridge')")
    p.add_argument("--threshold", type=float, required=True, help="Threshold for pass/fail on reduced error")
    p.add_argument("--metric", default="l2", choices=["l2", "linf", "l1", "mean"],
                   help="Reduction to apply to ee_error vector")
    p.add_argument("--l2-dims", type=int, default=3,
                   help="When metric=l2, apply L2 to the first N dims (default: 3)")
    p.add_argument("--episodes", default="",
                   help="Subset episodes like '0:1000,2000:2500' by episode_index value (empty = all)")
    p.add_argument("--active-eps", type=float, default=1e-12,
                   help="Frame is 'active' for a robot if any |err| > this (default 1e-12)")
    p.add_argument("--csv", default="", help="Optional CSV path to append results")
    p.add_argument("--outdir", required=True,
                   help="Base directory to save figures; a subfolder named after --prefix will be created inside.")
    args = p.parse_args()

    root = Path(args.root)
    repo_ids = find_repo_ids(root, args.prefix)
    if not repo_ids:
        raise SystemExit(f"No repos found under {root} starting with '{args.prefix}'")

    episodes_filter = parse_episodes_spec(args.episodes)

    # GLOBAL accumulators across all repos
    global_stats_by_robot = defaultdict(lambda: dict(
        episodes_total=0, episodes_all_ok=0, frames_total=0, frames_ok=0,
        bin_counts=np.zeros(4, dtype=np.int64)
    ))
    global_overall = dict(
        episodes_total=0, episodes_all_ok=0, frames_total=0, frames_ok=0,
        bin_counts=np.zeros(4, dtype=np.int64)
    )

    all_csv_rows: List[dict] = []

    for repo_id in repo_ids:
        repo_dir = root / repo_id
        stats, overall, skipped, total_eps_true = analyze_repo_local_parquet(
            repo_dir=repo_dir,
            threshold=args.threshold,
            metric=args.metric,
            l2_dims=args.l2_dims,
            active_eps=args.active_eps,
            episodes_filter=episodes_filter,
        )

        # Per-repo summary
        print_repo_summary(repo_id, stats, overall, skipped, total_eps_true,
                           args.threshold, args.metric, args.l2_dims)

        # CSV rows for this repo
        for robot, r in sorted(stats.items()):
            fr_total = int(r["frames_total"])
            b0, b1, b2, b3 = [int(x) for x in r["bin_counts"]]
            all_csv_rows.append({
                "repo_id": repo_id,
                "robot": robot,
                "metric": args.metric,
                "l2_dims": args.l2_dims if args.metric == "l2" else "",
                "threshold": args.threshold,
                "episodes_total": int(r["episodes_total"]),
                "episodes_all_ok": int(r["episodes_all_ok"]),
                "episodes_all_ok_pct": as_pct(int(r["episodes_all_ok"]), int(r["episodes_total"])),
                "frames_total": fr_total,
                "frames_ok": int(r["frames_ok"]),
                "frames_ok_pct": as_pct(int(r["frames_ok"]), fr_total),
                "frames_bin_0_0p01": b0,
                "frames_bin_0p01_0p02": b1,
                "frames_bin_0p02_0p03": b2,
                "frames_bin_ge_0p03": b3,
                "pct_bin_0_0p01": as_pct(b0, fr_total),
                "pct_bin_0p01_0p02": as_pct(b1, fr_total),
                "pct_bin_0p02_0p03": as_pct(b2, fr_total),
                "pct_bin_ge_0p03": as_pct(b3, fr_total),
            })

            # Accumulate GLOBAL per-robot
            g = global_stats_by_robot[robot]
            g["episodes_total"] += int(r["episodes_total"])
            g["episodes_all_ok"] += int(r["episodes_all_ok"])
            g["frames_total"] += int(r["frames_total"])
            g["frames_ok"] += int(r["frames_ok"])
            g["bin_counts"] += r["bin_counts"]

        # Accumulate GLOBAL overall
        global_overall["episodes_total"] += int(overall["episodes_total"])
        global_overall["episodes_all_ok"] += int(overall["episodes_all_ok"])
        global_overall["frames_total"] += int(overall["frames_total"])
        global_overall["frames_ok"] += int(overall["frames_ok"])
        global_overall["bin_counts"] += overall["bin_counts"]

    # GLOBAL summary print
    print("\n================== GLOBAL (all repos with prefix) ==================")
    header = f"{'robot':16s} {'ep_total':>9s} {'ep_all_ok':>9s} {'ep_all_ok%':>10s}   {'frames':>9s} {'frames_ok':>10s} {'frames_ok%':>10s}"
    print(header)
    print("-" * len(header))
    for robot, r in sorted(global_stats_by_robot.items()):
        ep_total = int(r["episodes_total"])
        ep_all_ok = int(r["episodes_all_ok"])
        fr_total = int(r["frames_total"])
        fr_ok = int(r["frames_ok"])
        print(f"{robot:16s} {ep_total:9d} {ep_all_ok:9d} {as_pct(ep_all_ok, ep_total):10.2f}   {fr_total:9d} {fr_ok:10d} {as_pct(fr_ok, fr_total):10.2f}")

    ep_total = int(global_overall["episodes_total"])
    ep_all_ok = int(global_overall["episodes_all_ok"])
    fr_total = int(global_overall["frames_total"])
    fr_ok = int(global_overall["frames_ok"])
    print("-" * len(header))
    print(f"{'ALL':16s} {ep_total:9d} {ep_all_ok:9d} {as_pct(ep_all_ok, ep_total):10.2f}   {fr_total:9d} {fr_ok:10d} {as_pct(fr_ok, fr_total):10.2f}")

    # Write CSV if requested
    if args.csv and all_csv_rows:
        write_csv_rows(Path(args.csv), all_csv_rows)
        print(f"Appended {len(all_csv_rows)} rows to CSV: {args.csv}")

    # ── Save GLOBAL + per-robot histograms and concatenated sheet ──
    out_root = Path(args.outdir) / _sanitize_component(args.prefix)
    out_root.mkdir(parents=True, exist_ok=True)

    created_pngs: List[Path] = []
    global_png = out_root / "global_hist.png"
    if save_hist(global_overall["bin_counts"], global_png, "GLOBAL frame error distribution"):
        print(f"Saved GLOBAL histogram to: {global_png}")
        created_pngs.append(global_png)

    for robot, r in sorted(global_stats_by_robot.items()):
        robot_png = out_root / f"robot_{_sanitize_component(robot)}.png"
        if save_hist(r["bin_counts"], robot_png, f"{robot} frame error distribution"):
            print(f"Saved {robot} histogram to: {robot_png}")
            created_pngs.append(robot_png)

    # Always attempt to build a concatenated sheet (if at least one PNG exists)
    sheet_png = out_root / "all_histograms.png"
    save_contact_sheet(created_pngs, sheet_png)


if __name__ == "__main__":
    main()

