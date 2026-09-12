# Antarctic Sea-Ice Concentration Forecasting Model

## Phase 1: SIC-Only Baseline

This repository contains Phase 1 of the Antarctic Sea-Ice Concentration (SIC) forecasting model. The goal is to establish a baseline using historical SIC data only, before adding environmental variables and more complex temporal architectures in later phases.

## Overview

**Objective**: Forecast next-day Antarctic sea-ice concentration using 7 days of historical SIC data.

**Model**: U-Net architecture with encoder-decoder structure and skip connections.

**Baseline Comparisons**: Persistence and climatology baselines.

**Data Source**: NOAA/NSIDC Sea Ice Concentration CDR v6 (G02202), Antarctic polar stereographic grid (25km resolution).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete Phase 1 workflow
bash scripts/run_phase1.sh
```

Or run steps individually:

```bash
# 1. Download data
python scripts/download_data.py --start-date 2010-01-01 --end-date 2022-12-31

# 2. Prepare data (create mask, splits, windows)
python scripts/prepare_data.py

# 3. Evaluate baselines (establishes performance bar)
python scripts/evaluate_baselines.py

# 4. Train U-Net model
python scripts/train.py

# 5. Evaluate trained model
python scripts/evaluate_model.py --checkpoint models/sic_unet_v001_best.pt --split test
```

## Project Structure

```
iceberg_forcasting_model/
├── src/seaice_forecast/       # Core package
│   ├── config/                # Configuration files (settings.yaml)
│   ├── data_processing/       # Data download, preprocessing, dataset
│   ├── models/                # U-Net, baselines
│   ├── evaluation/            # Training, evaluation, metrics
│   └── utils/                 # Visualization utilities
├── scripts/                   # Executable scripts
│   ├── download_data.py       # Download NSIDC data
│   ├── prepare_data.py        # Preprocess data
│   ├── evaluate_baselines.py # Baseline evaluation
│   ├── train.py               # Train U-Net
│   ├── evaluate_model.py      # Model evaluation
│   └── run_phase1.sh          # Complete workflow
├── data/                      # Data directory
│   ├── raw/                   # Raw NSIDC NetCDF files
│   └── processed/             # Preprocessed NPZ files, mask
├── models/                    # Saved model checkpoints
├── output/                    # Results JSON, plots, logs
└── tests/                     # Unit tests
```

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

## Detailed Workflow

### 1. Data Preparation

#### Download NSIDC Data
Downloads daily sea-ice concentration from NOAA/NSIDC CDR v6:

```bash
python scripts/download_data.py \
    --start-date 2010-01-01 \
    --end-date 2022-12-31 \
    --output-dir data/raw/nsidc

# Check availability without downloading
python scripts/download_data.py \
    --start-date 2010-01-01 \
    --end-date 2010-01-31 \
    --check-only
```

**Output**: Raw NetCDF files in `data/raw/nsidc/`

#### Preprocess Data
Creates land-ocean mask, splits data chronologically, builds sliding windows:

```bash
python scripts/prepare_data.py

# Skip mask creation if it already exists
python scripts/prepare_data.py --skip-mask
```

**Output**:
- `data/processed/land_ocean_mask.npy` - Binary mask (1=ocean, 0=land)
- `data/processed/{train,val,test}_data.npz` - Input/target arrays
- `data/processed/{train,val,test}_metadata.json` - Dates and metadata
- `data/processed/normalization_stats.json` - Mean/std from training set

**Data Splits** (default from config):
- Train: 2010-01-01 to 2018-12-31
- Validation: 2019-01-01 to 2020-12-31
- Test: 2021-01-01 to 2022-12-31

### 2. Baseline Evaluation

Evaluates persistence and climatology baselines **before** training U-Net:

```bash
# Evaluate on validation set
python scripts/evaluate_baselines.py

# Evaluate on test set
python scripts/evaluate_baselines.py --split test

# Save fitted climatology model
python scripts/evaluate_baselines.py --save-climatology
```

**Output**:
- `output/baseline_results_{split}.json` - Metrics for both baselines
- `output/climatology_model.npz` - Fitted climatology (optional)

**Why this matters**: These baselines set the performance bar. The U-Net **must** beat both to be considered successful.

### 3. Model Training

Trains U-Net with masked MAE loss, early stopping, and learning rate scheduling:

```bash
# Train with default config
python scripts/train.py

# Override config settings
python scripts/train.py --epochs 50 --batch-size 16 --device cuda

# Resume from checkpoint
python scripts/train.py --checkpoint models/sic_unet_v001_final.pt
```

**Training Features**:
- Masked MAE loss (excludes land pixels)
- Adam optimizer with initial LR 1e-3
- ReduceLROnPlateau scheduler
- Early stopping (patience: 15 epochs)
- Checkpoint saving (best and final)

**Output**:
- `models/sic_unet_v001_best.pt` - Best model by validation loss
- `models/sic_unet_v001_final.pt` - Final model
- `models/sic_unet_v001_history.json` - Training history
- `output/plots/sic_unet_v001_training_history.png` - Loss curves

**Monitoring**: Watch for:
- Training loss should decrease steadily
- Validation loss should decrease (if increasing, model is overfitting)
- U-Net should beat baselines by epoch 10-20

### 4. Model Evaluation

Comprehensive evaluation on test set with baseline comparison and visualizations:

```bash
# Evaluate on test set
python scripts/evaluate_model.py \
    --checkpoint models/sic_unet_v001_best.pt \
    --split test \
    --visualize 5

# Evaluate on validation set
python scripts/evaluate_model.py \
    --checkpoint models/sic_unet_v001_best.pt \
    --split val

# Skip baseline comparison
python scripts/evaluate_model.py \
    --checkpoint models/sic_unet_v001_best.pt \
    --no-baseline-comparison
```

**Output**:
- `output/model_results_{split}.json` - Model metrics
- `output/plots/metric_comparison_{split}.png` - Bar chart comparison
- `output/plots/best_prediction_{1-5}.png` - Best predictions
- `output/plots/worst_prediction_{1-5}.png` - Worst predictions
- `output/plots/error_map_{split}.png` - Spatial error distribution
- `output/plots/ice_edge_comparison_{split}.png` - Ice edge accuracy

## Key Features

### Data Pipeline
- **NSIDC CDR v6**: Daily Antarctic SIC at 25km resolution (316×332 grid)
- **Land-Ocean Mask**: Automatically created from data consistency
- **Sliding Windows**: 7-day input → 1-day forecast
- **Chronological Splits**: No temporal leakage between train/val/test
- **Missing Data**: Filled with 0 (no ice) and documented
- **Normalization**: Computed from training set only

### Baseline Models

#### Persistence
Simplest forecast: tomorrow's SIC = today's SIC
```python
SIC_hat(t+1) = SIC(t)
```

#### Climatology
Day-of-year expected SIC with smoothing:
- Averages all training years for each calendar day
- 15-day smoothing window (configurable)
- Handles leap years

### U-Net Architecture

**Input**: [Batch, 7, H, W] - 7 days of SIC history  
**Output**: [Batch, 1, H, W] - Next-day SIC forecast

**Architecture**:
- Encoder: 4 stages [32, 64, 128, 256 channels]
- Bottleneck: 512 channels
- Decoder: 4 stages with skip connections
- Output: Sigmoid activation for [0,1] bounded SIC

**Parameters**: ~2.3M trainable parameters

**Training Details**:
- Loss: Masked MAE (land pixels excluded)
- Optimizer: Adam (LR 1e-3)
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 5)
- Early stopping: 15 epochs patience
- Batch size: 8 (adjustable)

### Evaluation Metrics

All metrics computed on ocean pixels only (land excluded):

1. **MAE** (Mean Absolute Error)
   - Average pixel-wise difference
   - Reported in same units as SIC [0-1]
   - Lower is better

2. **RMSE** (Root Mean Squared Error)
   - Penalizes large errors more than MAE
   - Lower is better

3. **Spatial Correlation**
   - Pearson correlation between predicted and actual fields
   - Measures pattern accuracy, not absolute values
   - Higher is better (range: -1 to 1)

4. **Ice Edge Displacement**
   - Distance between predicted and actual ice edge (15% SIC threshold)
   - Measured in pixels (1 pixel = 25km)
   - Lower is better

## Configuration

Edit `src/seaice_forecast/config/settings.yaml` to customize:

```yaml
# Model metadata
model:
  name: "sic_unet_v001"
  version: "0.1.0"

# Data date ranges
data:
  date_ranges:
    train_start: "2010-01-01"
    train_end: "2018-12-31"
    # ... val and test ranges

# Model architecture
architecture:
  encoder_channels: [7, 32, 64, 128, 256]
  output_activation: "sigmoid"

# Training hyperparameters
training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.001
  early_stopping:
    patience: 15
```

## Expected Results

### Performance Bar (Baselines)
Typical baseline performance on Antarctic SIC:
- **Persistence**: MAE ~0.03-0.05, RMSE ~0.06-0.08
- **Climatology**: MAE ~0.05-0.07, RMSE ~0.08-0.10

### Target U-Net Performance
Phase 1 success criteria:
- MAE < 0.03 (beat best baseline by >10%)
- RMSE < 0.05
- Spatial correlation > 0.85
- Ice edge displacement < 5 pixels (125km)

### What Can Go Wrong

**U-Net doesn't beat baselines**:
- Check if data normalization is applied correctly
- Verify mask is excluding land pixels in loss
- Increase model capacity or training epochs
- Check for data leakage in splits

**Training loss plateaus early**:
- Learning rate may be too low
- Try different optimizer or scheduler settings
- Check if mask is correct (not masking too much)

**Predictions are too smooth**:
- Common with MSE/MAE loss
- Consider adding perceptual loss or adversarial loss in Phase 2

## Outputs

After running the complete workflow:

```
models/
├── sic_unet_v001_best.pt      # Best checkpoint (use this!)
├── sic_unet_v001_final.pt     # Final checkpoint
└── sic_unet_v001_history.json # Training history

output/
├── baseline_results_val.json
├── baseline_results_test.json
├── model_results_test.json
└── plots/
    ├── sic_unet_v001_training_history.png
    ├── metric_comparison_test.png
    ├── best_prediction_1.png
    ├── worst_prediction_1.png
    ├── error_map_test.png
    └── ice_edge_comparison_test.png
```

## Next Phases

### Phase 2: Environmental Variables
Add wind (u/v), SST, and ocean currents to input:
- Fetch ERA5 and Copernicus Marine data
- Reproject to SIC grid
- Expand input channels: [7 timesteps × 7 variables]

### Phase 3: Temporal Modeling
Replace static U-Net with ConvLSTM:
- Better capture temporal dependencies
- Improved multi-day forecasting

### Phase 4: Multi-Horizon Forecasting
Extend to 3-7 day forecasts:
- Recursive or direct multi-output approach
- Quantify error accumulation with lead time

### Phase 5: Operational Integration
- Define risk function: `risk = f(SIC)`
- Convert SIC forecasts to cost surfaces
- Interface with route optimization

## Troubleshooting

### Data Download Issues
```bash
# If downloads fail, check connectivity to NSIDC
curl -I https://noaadata.apps.nsidc.org/NOAA/G02202_V6/south/daily/2020/

# Check data availability first
python scripts/download_data.py --start-date 2020-01-01 --end-date 2020-01-10 --check-only
```

### Memory Issues
```bash
# Reduce batch size
python scripts/train.py --batch-size 4

# Use CPU instead of GPU
python scripts/train.py --device cpu
```

### CUDA Out of Memory
```bash
# Smaller batch size
python scripts/train.py --batch-size 4

# Reduce model size in config (fewer encoder channels)
```

## Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# With coverage
pytest --cov=seaice_forecast tests/
```

### Code Style
```bash
# Format code
black src/ scripts/

# Lint
flake8 src/ scripts/

# Type checking
mypy src/
```

## License

Research use only.

## Citation

[To be added]

## Contact

[To be added]
