#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

.venv/bin/python scripts/train.py \
  --model ibm-granite/granite-embedding-97m-multilingual-r2 \
  --train-file data/processed/train.jsonl \
  --validation-file data/processed/validation.jsonl \
  --output-dir outputs/granite-diffusion-97m \
  --max-length 96 \
  --micro-batch-size 8 \
  --gradient-accumulation 1 \
  --epochs 3 \
  --head-only-steps 100 \
  --save-steps 500 \
  --log-steps 25 \
  --no-gradient-checkpointing
