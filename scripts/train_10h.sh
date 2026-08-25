#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SOURCE_CHECKPOINT="outputs/granite-diffusion-97m/final"
if [[ ! -d "$SOURCE_CHECKPOINT" ]]; then
  echo "Missing $SOURCE_CHECKPOINT; run scripts/train_initial.sh first." >&2
  exit 1
fi
if [[ ! -f data/processed-256/train.jsonl ]]; then
  echo "Missing prepared full dataset; run scripts/prepare_full_data.sh first." >&2
  exit 1
fi

TRAIN_COMMAND=(
  .venv/bin/python scripts/train.py
  --model "$SOURCE_CHECKPOINT"
  --train-file data/processed-256/train.jsonl
  --validation-file data/processed-256/validation.jsonl
  --output-dir outputs/granite-diffusion-97m-10h
  --max-length 256
  --epochs 1000
  --micro-batch-size 6
  --gradient-accumulation 1
  --head-only-steps 0
  --learning-rate 1e-5
  --head-learning-rate 1e-5
  --warmup-steps 200
  --scheduler constant
  --save-steps 0
  --save-minutes 120
  --max-duration-hours 10
  --log-steps 100
  --no-gradient-checkpointing
)

if command -v systemd-inhibit >/dev/null 2>&1; then
  exec systemd-inhibit \
    --what=sleep \
    --mode=block \
    --why="Granite Diffusion 10-hour local training" \
    "${TRAIN_COMMAND[@]}"
fi
exec "${TRAIN_COMMAND[@]}"
