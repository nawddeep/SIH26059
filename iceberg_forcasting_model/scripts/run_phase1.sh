#!/bin/bash
# Complete Phase 1 workflow - from data download to model evaluation
# Usage: bash scripts/run_phase1.sh

set -e  # Exit on error

echo "=================================="
echo "Phase 1: SIC Forecasting Workflow"
echo "=================================="
echo ""

# Configuration
START_DATE="2010-01-01"
END_DATE="2022-12-31"

echo "Step 1: Download NSIDC data"
echo "----------------------------"
python scripts/download_data.py \
    --start-date $START_DATE \
    --end-date $END_DATE
echo ""

echo "Step 2: Prepare data (mask, splits, windows)"
echo "---------------------------------------------"
python scripts/prepare_data.py
echo ""

echo "Step 3: Evaluate baselines"
echo "--------------------------"
python scripts/evaluate_baselines.py --split val
echo ""

echo "Step 4: Train U-Net model"
echo "-------------------------"
python scripts/train.py
echo ""

echo "Step 5: Evaluate trained model"
echo "-------------------------------"
python scripts/evaluate_model.py \
    --checkpoint models/sic_unet_v001_best.pt \
    --split test \
    --visualize 5
echo ""

echo "=================================="
echo "Phase 1 Complete!"
echo "=================================="
echo ""
echo "Results are in:"
echo "  - Models: models/"
echo "  - Results: output/"
echo "  - Plots: output/plots/"
