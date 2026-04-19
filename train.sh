#!/usr/bin/env bash
# =============================================================
# train.sh — Banking Intent Classification Training Pipeline
# Usage:
#   bash train.sh                   # default paths from train.yml
#   DATA_DIR=/path/to/data bash train.sh
#   OUTPUT_DIR=/path/to/out bash train.sh
# =============================================================

set -euo pipefail   # exit on error, unset var, or pipe failure

# ── Resolve project root (directory containing this script) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="./configs/train.yml"

echo "============================================================"
echo " Banking Intent Classifier — Training Pipeline"
echo "============================================================"
echo " Config   : $CONFIG_FILE"
echo " Data dir : ${DATA_DIR:-'(from config)'}"
echo " Output   : ${OUTPUT_DIR:-'(from config)'}"
echo "============================================================"

# ── Step 1: Install dependencies ─────────────────────────────
echo ""
echo "[1/3] Installing dependencies..."
pip install unsloth datasets pandas pyyaml trl -q

# ── Step 2: (Optional) Preprocess raw data ───────────────────
# Uncomment the block below if you need to regenerate sample_data/
# from a raw CSV source before training.
#
# echo ""
# echo "[2/3] Preprocessing data..."
# python scripts/preprocess_data.py \
#     --input  ./raw_data/raw_banking77.csv \
#     --output ./sample_data

echo ""
echo "[2/3] Skipping preprocessing (sample_data/ already prepared)."

# ── Step 3: Run training ─────────────────────────────────────
echo ""
echo "[3/3] Starting training..."

python scripts/train.py

echo ""
echo "============================================================"
echo " Training complete!"
echo " Checkpoints saved to: ${OUTPUT_DIR:-./checkpoints/qwen2.5-banking77}"
echo "============================================================"
