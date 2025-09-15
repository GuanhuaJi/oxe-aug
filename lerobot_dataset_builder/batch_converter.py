import argparse
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────── YOUR JOBS ──────────────────────────
jobs = [
    # ("bridge", "test", ["panda", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "xarm7"], 0, 3475, 1000),
    # ("ucsd_kitchen_dataset_converted_externally_to_rlds", "train", ["panda", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "widowX"], 0, 150, 1000),
    # ("nyu_franka_play_dataset_converted_externally_to_rlds", "train", ["xarm7", "ur5e", "sawyer", "jaco", "kuka_iiwa", "kinova3", "google_robot", "widowX"], 0, 365, 1000),
    # ("language_table", "train", ["panda", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "xarm7"], 0, 10, 10),
    # ("nyu_franka_play_dataset_converted_externally_to_rlds", "val", ["xarm7", "ur5e", "sawyer", "jaco", "kuka_iiwa", "kinova3", "google_robot", "widowX"], 0, 91, 1000),
    # ("nyu_franka_play_dataset_converted_externally_to_rlds", "train", ["xarm7", "ur5e", "sawyer", "jaco", "kuka_iiwa", "kinova3", "google_robot", "widowX"], 0, 365, 1000),
    # ("toto", "train", ["widowX", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "xarm7"], 0, 902, 100),
    # ("toto", "test", ["widowX", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "xarm7"], 0, 101, 100),
    # ("taco_play", "test", ["widowX", "sawyer", "ur5e", "google_robot", "jaco", "kinova3", "kuka_iiwa", "xarm7"], 0, 361, 1000),
    ("jaco_play", "train", ["widowX", "sawyer", "ur5e", "google_robot", "panda", "kinova3", "kuka_iiwa", "xarm7"], 0, 976, 1000),
    ("jaco_play", "test", ["widowX", "sawyer", "ur5e", "google_robot", "panda", "kinova3", "kuka_iiwa", "xarm7"], 0, 108, 1000),
]

# ───────────── defaults (override via CLI) ─────
DEFAULT_CONVERTER = "/home/guanhuaji/lerobot_rovi_aug/lerobot_converter_fast.py"
DEFAULT_OUT_ROOT  = "/home/abinayadinesh/lerobot_dataset"
DEFAULT_TRG_ROOT  = "/home/abinayadinesh/rovi_aug_extension_full"


def chunks(start, end, size):
    """
    Yield [s, e) windows covering [start, end).
    If the tail chunk length is strictly less than half 'size',
    merge that tail into the previous chunk.

    Examples:
      start=0, end=101, size=100 -> [(0, 101)]
      start=0, end=250, size=100 -> [(0,100), (100,200), (200,250)]  # tail == 50 (== half) => no merge
    """
    if size <= 0 or end <= start:
        return  # nothing to yield

    total = end - start
    nfull = total // size
    rem = total - nfull * size

    half = size / 2.0  # strict comparison against half

    # Case 1: no remainder -> yield all full chunks
    if rem == 0:
        s = start
        for _ in range(nfull):
            yield s, s + size
            s += size
        return

    # Case 2: remainder exists but we want to keep it as a separate chunk
    # Conditions: remainder >= half, or there were no full chunks to merge with.
    if rem >= half or nfull == 0:
        s = start
        for _ in range(nfull):
            yield s, s + size
            s += size
        # tail as its own chunk
        yield s, end
        return

    # Case 3: small tail (rem < half) and at least one full chunk -> merge tail into the last full chunk
    # Emit the first nfull-1 full chunks, then one merged chunk to 'end'.
    s = start
    for _ in range(max(0, nfull - 1)):
        yield s, s + size
        s += size
    # merged final chunk
    yield s, end


def build_cmd(converter, dataset, split, robots, start, end,
              fps, out_root, trg_root):
    return [
        "python", converter,
        "--dataset", dataset,
        "--split", split,
        "--robots", *robots,
        "--start", str(start),
        "--end", str(end),
        "--fps", str(fps),
        "--out_root", out_root,
        "--trg_root", trg_root,
    ]


def cmd_to_str(cmd):
    return " ".join(shlex.quote(x) for x in cmd)


def main():
    ap = argparse.ArgumentParser(description="Batch launcher for LeRobot conversion chunks (no type hints).")
    ap.add_argument("--converter", default=DEFAULT_CONVERTER, help="Path to lerobot_converter_fast.py")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--trg_root", default=DEFAULT_TRG_ROOT)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("-j", "--jobs", type=int, default=1, help="Max concurrent converter processes")
    ap.add_argument("--dry-run", action="store_true", help="Print commands and exit without running")
    args = ap.parse_args()

    # Build all commands
    all_cmds = []
    for dataset, split, robots, start_ep, end_ep, per_ds in jobs:
        if per_ds <= 0:
            raise ValueError(f"Episodes-per-dataset must be > 0 (got {per_ds})")
        # create chunk list with smart tail merge
        chunk_list = list(chunks(start_ep, end_ep, per_ds))
        for s, e in chunk_list:
            all_cmds.append(build_cmd(
                args.converter, dataset, split, robots, s, e,
                args.fps, args.out_root, args.trg_root
            ))

    if args.dry_run:
        for c in all_cmds:
            print(cmd_to_str(c))
        return

    # Run (optionally) in parallel
    if args.jobs <= 1:
        for c in all_cmds:
            print(f"[RUN] {cmd_to_str(c)}")
            subprocess.run(c, check=True)
    else:
        print(f"[PARALLEL x{args.jobs}] Launching {len(all_cmds)} commands…")
        failures = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            fut2cmd = {ex.submit(subprocess.run, c, check=True): c for c in all_cmds}
            for fut in as_completed(fut2cmd):
                c = fut2cmd[fut]
                try:
                    fut.result()
                    print(f"[OK] {cmd_to_str(c)}")
                except subprocess.CalledProcessError as e:
                    failures += 1
                    print(f"[ERR exit={e.returncode}] {cmd_to_str(c)}")
        if failures:
            raise SystemExit(f"{failures} command(s) failed")


if __name__ == "__main__":
    main()
