#!/bin/bash
# scripts/run_eval_all.sh
# Jalankan evaluation pipeline otomatis dengan papermill.
# Cocok untuk overnight run di Kaggle (time limit 9 jam).
#
# Usage:
#   pip install papermill
#   bash scripts/run_eval_all.sh [PROJECT_ROOT]

set -e
PROJECT_ROOT="${1:-$(pwd)}"
NOTEBOOKS="$PROJECT_ROOT/notebooks"
LOG_DIR="$PROJECT_ROOT/artifacts/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOG_DIR"

echo "================================================"
echo "  Soccer Homography Benchmark — Eval Runner"
echo "  Project : $PROJECT_ROOT"
echo "  Started : $(date)"
echo "================================================"

command -v papermill &>/dev/null || pip install papermill -q

echo ""
echo "Checking weights..."
WEIGHTS_DIR="$PROJECT_ROOT/artifacts/weights"
MISSING=0
for fv in "yolo11/small" "yolo11/medium" "yolo11/xlarge" \
           "hrnet/w18" "hrnet/w32" "hrnet/w48" \
           "vitpose/small" "vitpose/base" "vitpose/large" \
           "detr/r50" "detr/r101"; do
    if ls "$WEIGHTS_DIR/$fv"/*.pt "$WEIGHTS_DIR/$fv"/*.pth 2>/dev/null | head -1 &>/dev/null; then
        echo "  OK $fv"
    else
        echo "  -- $fv (belum ada)"
        MISSING=$((MISSING+1))
    fi
done
[ $MISSING -gt 0 ] && echo "  WARN: $MISSING model belum siap — akan di-skip saat eval"

run_nb() {
    local nb="$1"
    echo ""
    echo "--- Running: $nb  [$(date)] ---"
    [ -f "$NOTEBOOKS/$nb" ] || { echo "NOT FOUND: $nb"; return 1; }
    papermill "$NOTEBOOKS/$nb" \
        "$LOG_DIR/executed_${nb%.ipynb}_${TIMESTAMP}.ipynb" \
        -p PROJECT_ROOT "$PROJECT_ROOT" \
        --execution-timeout 28800 \
        --kernel python3 \
        2>&1 | tee "$LOG_DIR/log_${nb%.ipynb}_${TIMESTAMP}.txt"
    echo "--- Done: $nb ---"
}

run_nb "03_evaluation.ipynb";         sleep 30
run_nb "04_homography_pipeline.ipynb"; sleep 30
run_nb "05_results_visualization.ipynb"

echo ""
echo "================================================"
echo "  Pipeline selesai! $(date)"
echo "  Results : $PROJECT_ROOT/artifacts/logs/evaluation/"
echo "  Figures : $PROJECT_ROOT/artifacts/results/figures/"
echo "================================================"
