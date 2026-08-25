#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

.venv/bin/python scripts/prepare_data.py \
  --parquet data/raw/data/train-00000-of-00006.parquet \
            data/raw/data/train-00001-of-00006.parquet \
            data/raw/data/train-00002-of-00006.parquet \
            data/raw/data/train-00003-of-00006.parquet \
            data/raw/data/train-00004-of-00006.parquet \
            data/raw/data/train-00005-of-00006.parquet \
  --output-dir data/processed-256 \
  --train-size 0 \
  --validation-size 1024 \
  --max-length 256
