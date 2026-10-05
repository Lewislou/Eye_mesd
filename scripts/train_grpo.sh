#!/usr/bin/env bash
# Launch Eye-MESD GRPO pretraining.
# Default is 2 GPUs x batch 80. Override with NPROC=1 or by editing the config.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
NPROC="${NPROC:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eye_mesd_pretrain}"
mkdir -p "$OUTPUT_DIR"
torchrun --nproc_per_node="$NPROC" dinov2/train/train.py \
  --config-file configs/eye_mesd_pretrain.yaml \
  --output-dir "$OUTPUT_DIR" \
  "$@"
