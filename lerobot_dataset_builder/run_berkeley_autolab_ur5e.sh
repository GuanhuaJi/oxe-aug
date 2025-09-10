#!/usr/bin/env bash
set -euo pipefail

PY=python
SCRIPT=/home/guanhuaji/lerobot_rovi_aug/convert_to_lerobot_v3.py
DATASET=berkeley_autolab_ur5
SPLIT=train
ROBOTS=(kuka_iiwa xarm7 sawyer panda kinova3 jaco google_robot widowX)
FPS=30
OUT_ROOT=/home/abinayadinesh/lerobot_dataset
NUM_WORKERS=1

# Change this to control parallelism
MAX_JOBS=${MAX_JOBS:-4}

ranges=(
  "0 100"
  "100 200"
  "200 300"
  "300 400"
  "400 500"
  "500 600"
  "600 700"
  "700 800"
  "800 896"   # last one
)

run_slice() {
  local start="$1" end="$2"
  echo "[$(date +%T)] Launching slice ${start}-${end}"
  "$PY" "$SCRIPT" \
    --dataset "$DATASET" --split "$SPLIT" \
    --robots "${ROBOTS[@]}" \
    --start "$start" --end "$end" --fps "$FPS" \
    --out_root "$OUT_ROOT" --num_workers "$NUM_WORKERS"
}

for pair in "${ranges[@]}"; do
  set -- $pair
  # throttle to MAX_JOBS concurrent tasks
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do sleep 1; done
  run_slice "$1" "$2" &
done

wait
echo "All slices completed."
