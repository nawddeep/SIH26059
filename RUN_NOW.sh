#!/bin/bash
# Antarctic Navigation AI - Complete Demo Script
# Just run: bash RUN_NOW.sh

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════"
echo "  ANTARCTIC NAVIGATION AI - AUTOMATED DEMO"
echo "════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Go to project directory
cd "$(dirname "$0")/seaice_forecast"

# Activate virtual environment
echo -e "${BLUE}[1/6]${NC} Activating Python environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# System check
echo -e "${BLUE}[2/6]${NC} Running system readiness test..."
python test_model_readiness.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ System check passed${NC}"
else
    echo -e "${YELLOW}⚠ System check had warnings (continuing anyway)${NC}"
fi
echo ""

# Quick training (5 epochs for demo)
echo -e "${BLUE}[3/7]${NC} Training model (quick 5-epoch demo)..."
echo "This will take ~1-2 minutes on Apple Silicon GPU..."
python scripts/training/train_phase1.py \
    --epochs 5 \
    --batch-size 4

echo -e "${GREEN}✓ Training complete${NC}"
echo ""

# Find the trained model
MODEL_PATH="checkpoints/sic_unet_v001_best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH=$(find checkpoints models -name "*best.pt" -o -name "*.pt" 2>/dev/null | head -1)
fi

echo "Using model: $MODEL_PATH"
echo ""

# Evaluate model
echo -e "${BLUE}[4/7]${NC} Evaluating model on test data..."
python scripts/evaluation/evaluate_model.py \
    --checkpoint "$MODEL_PATH" \
    --split test

echo -e "${GREEN}✓ Evaluation complete${NC}"
echo ""

# Create visualizations
echo -e "${BLUE}[5/7]${NC} Creating prediction visualizations..."
python scripts/visualization/visualize_predictions.py

echo -e "${GREEN}✓ Visualizations created${NC}"
echo ""

# Run uncertainty & risk demo (Phase E)
echo -e "${BLUE}[6/7]${NC} Running Phase E: Uncertainty & IMO POLARIS risk demo..."
if [ -f "demo_phase_e.py" ]; then
    python demo_phase_e.py --config configs/phase_e.yaml
    echo -e "${GREEN}✓ Phase E demo complete${NC}"
else
    echo -e "${YELLOW}⚠ Phase E demo not available (optional)${NC}"
fi
echo ""

# Run route optimizer demo (Phase F)
echo -e "${BLUE}[7/7]${NC} Running Phase F: Route optimization demo..."
if [ -f "demo_phase_f.py" ]; then
    python demo_phase_f.py --config configs/phase_f.yaml
    echo -e "${GREEN}✓ Route demo complete${NC}"
else
    echo -e "${YELLOW}⚠ Phase F demo not available (optional)${NC}"
fi
echo ""

# Summary
echo "════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}ALL DEMOS COMPLETE!${NC}"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "📁 Files created for your mentor:"
echo ""
echo "1. Trained Model Checkpoint:"
echo "   $MODEL_PATH"
echo ""
echo "2. Evaluation Results:"
echo "   output/model_results_test.json"
echo ""
echo "3. Visualizations:"
echo "   output/plots/metric_comparison_test.png"
echo "   output/plots/error_map_test.png"
echo "   output/plots/ice_edge_comparison_test.png"
echo ""
echo "4. Uncertainty & Risk (Phase E):"
echo "   output/phase_e/phase_e_uncertainty_demo.png"
echo "   output/phase_e/phase_e_risk_vs_sic.png"
echo "   output/phase_e/phase_e_risk_table.json"
echo ""
echo "5. Route Optimization (Phase F):"
echo "   output/phase_f/phase_f_route.png"
echo "   output/phase_f/phase_f_route_summary.json"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "To view results:"
echo "  open output/plots/metric_comparison_test.png"
echo "  open output/phase_e/phase_e_risk_vs_sic.png"
echo "  open output/phase_f/phase_f_route.png"
echo "════════════════════════════════════════════════════════════"

