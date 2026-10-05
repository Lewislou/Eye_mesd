#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
NPROC="${NPROC:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
mkdir -p "$OUTPUT_DIR"
torchrun --nproc_per_node="$NPROC" eye_mesd/train/train.py \
  --config-file configs/vitl14.yaml \
  --output-dir "$OUTPUT_DIR" \
  "$@"
