#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

.venv/bin/python scripts/prepare_data.py \
  --parquet data/raw/data/train-00000-of-00006.parquet \
  --output-dir data/processed \
  --train-size 7000 \
  --validation-size 256 \
  --max-length 96
