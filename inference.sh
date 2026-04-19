#!/usr/bin/env bash
# =============================================================
# inference.sh — Banking Intent Classification Inference Pipeline
#
# Usage on Kaggle (3-cell notebook):
#
#   Cell 1: !git clone https://github.com/<user>/banking-intent-unsloth.git && cd banking-intent-unsloth
#   Cell 2: Set env vars and run:
#
#   MODEL_CHECKPOINT=/kaggle/working/checkpoints/qwen2.5-banking77/final_checkpoint \
#   OUTPUT_PATH=/kaggle/working/predictions.csv \
#   bash inference.sh
#
# Environment variable overrides:
#   MODEL_CHECKPOINT  — path to the fine-tuned LoRA checkpoint directory
#   TEST_DATA_PATH    — path to test CSV (default: ./sample_data/test.csv)
#   OUTPUT_PATH       — where to write predictions CSV (default: ./results/predictions.csv)
# =============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="./configs/inference.yml"

echo "============================================================"
echo " Banking Intent Classifier — Inference Pipeline"
echo "============================================================"
echo " Config      : $CONFIG_FILE"
echo " Checkpoint  : ${MODEL_CHECKPOINT:-'(from config)'}"
echo " Test data   : ${TEST_DATA_PATH:-'(from config)'}"
echo " Output CSV  : ${OUTPUT_PATH:-'(from config)'}"
echo "============================================================"

# ── Step 1: Install dependencies ─────────────────────────────
echo ""
echo "[1/2] Installing dependencies..."
pip install unsloth datasets pandas pyyaml -q

# ── Step 2: Run batch inference ───────────────────────────────
echo ""
echo "[2/2] Running inference..."

python scripts/inference.py --config "$CONFIG_FILE"

echo ""
echo "============================================================"
echo " Inference complete!"
echo " Results saved to: ${OUTPUT_PATH:-./results/predictions.csv}"
echo "============================================================"
