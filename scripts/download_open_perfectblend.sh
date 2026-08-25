#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data/raw/data"
BASE_URL="https://huggingface.co/datasets/mlabonne/open-perfectblend/resolve/main/data"

mkdir -p "$DATA_DIR"
for index in 0 1 2 3 4 5; do
  printf -v shard "%05d" "$index"
  filename="train-${shard}-of-00006.parquet"
  target="$DATA_DIR/$filename"
  if [[ -s "$target" ]]; then
    echo "Already downloaded: $filename"
    continue
  fi
  echo "Downloading $filename"
  curl --fail --location --retry 3 --output "$target.partial" "$BASE_URL/$filename"
  mv "$target.partial" "$target"
done

echo "Dataset shards ready in $DATA_DIR"
