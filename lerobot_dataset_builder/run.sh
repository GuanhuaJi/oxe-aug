#!/usr/bin/env bash
# run_chunks.sh

set -euo pipefail

PY="/home/guanhuaji/lerobot_rovi_aug/convert_to_lerobot_v3.py"
DATASET="toto"
SPLIT="train"
ROBOTS=("kuka_iiwa" "xarm7" "sawyer" "ur5e" "kinova3" "jaco" "google_robot" "widowX")
FPS=30
OUT_ROOT="/home/abinayadinesh/lerobot_dataset"

MAX_JOBS=4  # how many chunks to run in parallel

for s in $(seq 0 100 800); do
  if [ "$s" -eq 800 ]; then
    e=902
  else
    e=$((s + 100))
  fi

  python "$PY" \
    --dataset "$DATASET" --split "$SPLIT" \
    --robots "${ROBOTS[@]}" \
    --start "$s" --end "$e" --fps "$FPS" \
    --out_root "$OUT_ROOT" --num_workers 1 &

  # throttle parallel jobs
  while (( $(jobs -r -p | wc -l) >= MAX_JOBS )); do sleep 1; done
done

wait
echo "All chunks finished."
