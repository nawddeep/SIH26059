# Getting Started with Sea-Ice Forecasting Model

This guide walks you through running Phase 1 from scratch.

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows with WSL
- **Python**: 3.9 or higher
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: 50GB free space (for data and models)
- **GPU**: Optional but recommended (CUDA-compatible for faster training)

### Python Environment

```bash
# Check Python version
python --version  # Should be 3.9+

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import xarray; print(xarray.__version__)"
```

## Quick Start (5 Minutes)

### Option 1: Automated Workflow

Run everything with one command:

```bash
bash scripts/run_phase1.sh
```

This will:
1. Download NSIDC data (2010-2022, ~13 years)
2. Preprocess data (mask, splits, windows)
3. Evaluate baselines
4. Train U-Net model
5. Evaluate trained model with visualizations

**Estimated time**: 2-6 hours (depends on download speed and GPU availability)

### Option 2: Step-by-Step

If you prefer to run each step manually:

```bash
# Step 1: Download data (30 min - 2 hours)
python scripts/download_data.py \
    --start-date 2010-01-01 \
    --end-date 2022-12-31

# Step 2: Prepare data (10-30 minutes)
python scripts/prepare_data.py

# Step 3: Evaluate baselines (2-5 minutes)
python scripts/evaluate_baselines.py

# Step 4: Train model (1-3 hours)
python scripts/train.py

# Step 5: Evaluate model (5-10 minutes)
python scripts/evaluate_model.py \
    --checkpoint models/sic_unet_v001_best.pt \
    --split test \
    --visualize 5
```

## Understanding Each Step

### Step 1: Download Data

**What it does**: Downloads daily sea-ice concentration from NOAA/NSIDC

**Files created**: `data/raw/nsidc/seaice_conc_daily_sh_YYYYMMDD_*.nc`

**Troubleshooting**:
```bash
# Check availability first (faster)
python scripts/download_data.py \
    --start-date 2020-01-01 \
    --end-date 2020-01-10 \
    --check-only

# If downloads fail, check network
curl -I https://noaadata.apps.nsidc.org/

# Download a smaller test range first
python scripts/download_data.py \
    --start-date 2020-01-01 \
    --end-date 2020-12-31
```

### Step 2: Prepare Data

**What it does**:
- Creates land-ocean mask from data consistency
- Splits data chronologically (train/val/test)
- Builds sliding windows (7-day input → 1-day target)
- Computes normalization statistics

**Files created**:
- `data/processed/land_ocean_mask.npy`
- `data/processed/{train,val,test}_data.npz`
- `data/processed/{train,val,test}_metadata.json`
- `data/processed/normalization_stats.json`

**What to check**:
```bash
# Verify processed data exists
ls -lh data/processed/*.npz

# Check split sizes
python -c "
import numpy as np
train = np.load('data/processed/train_data.npz')
val = np.load('data/processed/val_data.npz')
test = np.load('data/processed/test_data.npz')
print(f'Train: {len(train[\"inputs\"])} samples')
print(f'Val: {len(val[\"inputs\"])} samples')
print(f'Test: {len(test[\"inputs\"])} samples')
"
```

### Step 3: Evaluate Baselines

**What it does**: Computes performance of persistence and climatology baselines

**Why important**: These set the performance bar the U-Net must beat

**Files created**:
- `output/baseline_results_val.json`
- `output/climatology_model.npz` (optional)

**What to check**:
```bash
# View baseline results
python -c "
import json
with open('output/baseline_results_val.json') as f:
    results = json.load(f)
    pers = results['baselines']['persistence']
    clim = results['baselines']['climatology']
    print('Persistence - MAE:', pers['mae'])
    print('Climatology - MAE:', clim['mae'])
"
```

**Expected baseline performance**:
- MAE: 0.03-0.05
- RMSE: 0.06-0.08
- Correlation: 0.70-0.80

### Step 4: Train Model

**What it does**: Trains U-Net with masked MAE loss and early stopping

**Files created**:
- `models/sic_unet_v001_best.pt` - Best model (use this!)
- `models/sic_unet_v001_final.pt` - Final model
- `models/sic_unet_v001_history.json` - Training history
- `output/plots/sic_unet_v001_training_history.png` - Loss curves

**Monitoring training**:
```bash
# Watch training in real-time
tail -f output/logs/training.log  # If logging to file

# Or just watch terminal output for:
# - Train loss decreasing
# - Val loss decreasing (if increasing → overfitting)
# - Early stopping trigger
```

**Training options**:
```bash
# Use CPU (no GPU)
python scripts/train.py --device cpu

# Reduce batch size (if out of memory)
python scripts/train.py --batch-size 4

# Fewer epochs (for testing)
python scripts/train.py --epochs 20

# Resume from checkpoint
python scripts/train.py \
    --checkpoint models/sic_unet_v001_final.pt
```

**Expected training time**:
- With GPU: 1-2 hours
- With CPU: 4-8 hours

**What to expect**:
- Best validation loss: ~0.02-0.03 (better than baselines)
- Early stopping: around epoch 30-50
- U-Net should beat baselines by epoch 10-20

### Step 5: Evaluate Model

**What it does**: Evaluates trained model on test set and compares with baselines

**Files created**:
- `output/model_results_test.json` - Metrics
- `output/plots/metric_comparison_test.png` - Bar chart
- `output/plots/best_prediction_*.png` - Best forecasts
- `output/plots/worst_prediction_*.png` - Worst forecasts
- `output/plots/error_map_test.png` - Spatial error
- `output/plots/ice_edge_comparison_test.png` - Ice edge

**What to check**:
```bash
# View model performance
python -c "
import json
with open('output/model_results_test.json') as f:
    results = json.load(f)
    metrics = results['metrics']
    print('MAE:', metrics['mae'])
    print('RMSE:', metrics['rmse'])
    print('Correlation:', metrics['spatial_correlation'])
    print('Ice Edge (px):', metrics['ice_edge_displacement'])
"

# Compare with baselines
cat output/baseline_results_test.json | grep mae
cat output/model_results_test.json | grep mae
```

**Success criteria**:
- U-Net MAE < baseline MAE
- U-Net RMSE < baseline RMSE
- U-Net correlation > baseline correlation
- Plots show reasonable predictions

## Common Issues & Solutions

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Reduce batch size
python scripts/train.py --batch-size 4

# Or use CPU
python scripts/train.py --device cpu
```

### Issue: Downloads are Very Slow

**Solution**:
```bash
# Download a smaller date range first (test)
python scripts/download_data.py \
    --start-date 2020-01-01 \
    --end-date 2020-12-31

# Then update config with these dates
# Edit src/seaice_forecast/config/settings.yaml
```

### Issue: Model Not Beating Baselines

**Possible causes**:
1. Not enough training (increase epochs)
2. Learning rate too high/low (try 0.0001 or 0.01)
3. Data issue (check mask is correct)
4. Normalization issue (verify stats computed from train only)

**Debugging**:
```bash
# Check if loss is decreasing
cat models/sic_unet_v001_history.json | grep val_loss

# Visualize training curve
python -c "
import json, matplotlib.pyplot as plt
with open('models/sic_unet_v001_history.json') as f:
    h = json.load(f)
plt.plot(h['val_loss'])
plt.title('Validation Loss')
plt.show()
"
```

### Issue: Import Errors

**Solution**:
```bash
# Make sure you're in venv
which python  # Should show venv path

# Reinstall dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Verifying Success

After running all steps, you should have:

```
✓ data/raw/nsidc/*.nc (4000+ files)
✓ data/processed/*.npz (3 files: train, val, test)
✓ data/processed/land_ocean_mask.npy
✓ output/baseline_results_val.json
✓ output/baseline_results_test.json
✓ models/sic_unet_v001_best.pt
✓ models/sic_unet_v001_history.json
✓ output/model_results_test.json
✓ output/plots/*.png (10+ plots)
```

Check with:
```bash
bash scripts/verify_outputs.sh  # If we created this
# Or manually:
ls data/processed/*.npz | wc -l  # Should be 3
ls output/plots/*.png | wc -l   # Should be 10+
```

## Next Steps

### Analyze Results

1. **Check MODEL_VERSION.md** - Fill in actual performance numbers
2. **Review plots** in `output/plots/` - Look for patterns
3. **Compare with baselines** - How much improvement?
4. **Identify failure modes** - Where does it struggle?

### Customize Configuration

Edit `src/seaice_forecast/config/settings.yaml`:

```yaml
# Try different date ranges
data:
  date_ranges:
    train_start: "2015-01-01"  # Shorter training period

# Try different architecture
architecture:
  encoder_channels: [7, 16, 32, 64, 128]  # Smaller model

# Try different hyperparameters
training:
  batch_size: 16          # Larger batches
  learning_rate: 0.0001   # Lower LR
```

### Experiment

```bash
# Train multiple models
for lr in 0.001 0.0001 0.00001; do
    python scripts/train.py --learning-rate $lr
done

# Compare results
python scripts/compare_models.py  # If we create this
```

## Getting Help

### Check Documentation
- `README.md` - Full documentation
- `MODEL_VERSION.md` - Model specifications
- `PHASE1_CHECKLIST.md` - Implementation checklist

### Debug Mode
```bash
# Run with verbose logging
python scripts/train.py --verbose

# Python debugger
python -m pdb scripts/train.py
```

### Common Commands
```bash
# Check GPU
nvidia-smi

# Monitor GPU during training
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check data file size
du -sh data/
```

## Summary

**Minimum viable workflow**:
```bash
pip install -r requirements.txt
bash scripts/run_phase1.sh
# Wait 2-6 hours
# Check output/plots/
```

**Success indicators**:
1. All steps complete without errors
2. U-Net beats both baselines
3. Plots look reasonable
4. Ready to move to Phase 2

---

**Questions?** See README.md for detailed documentation.

**Ready to start?** Run `bash scripts/run_phase1.sh`
