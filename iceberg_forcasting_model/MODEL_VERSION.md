# Model Version: sic_unet_v001

## Metadata

- **Version**: 0.1.0
- **Phase**: 1 (SIC-only baseline)
- **Created**: [Date will be auto-filled during training]
- **Status**: Development
- **Model Type**: U-Net
- **Task**: Next-day sea-ice concentration forecasting

## Model Specifications

### Architecture

**Type**: U-Net with encoder-decoder and skip connections

**Input**:
- Shape: [Batch, 7, 316, 332]
- Description: 7 days of Antarctic SIC history
- Grid: NSIDC 25km polar stereographic (316×332)
- Value range: [0, 1] (normalized from 0-100%)

**Output**:
- Shape: [Batch, 1, 316, 332]
- Description: Next-day SIC forecast
- Activation: Sigmoid (bounded [0, 1])

**Architecture Details**:
- Encoder channels: [7 → 32 → 64 → 128 → 256]
- Bottleneck: 512 channels
- Decoder channels: [256 → 128 → 64 → 32 → 1]
- Skip connections: Yes (concat between encoder/decoder)
- Batch normalization: Yes
- Dropout: 0.0 (not used in Phase 1)

**Parameters**:
- Total: ~2,300,000
- Trainable: ~2,300,000

### Training Configuration

**Data**:
- Source: NOAA/NSIDC Sea Ice Concentration CDR v6 (G02202)
- Region: Antarctic (South)
- Resolution: 25km polar stereographic grid
- Training period: 2010-01-01 to 2018-12-31
- Validation period: 2019-01-01 to 2020-12-31
- Test period: 2021-01-01 to 2022-12-31

**Preprocessing**:
- Land-ocean mask: Created from data consistency
- Normalization: Scale from 0-100 to 0-1 (× 0.01)
- Missing values: Filled with 0.0 (no ice)
- Window construction: 7-day input, 1-day target, stride 1

**Hyperparameters**:
- Batch size: 8
- Epochs: 100 (with early stopping)
- Optimizer: Adam
- Initial learning rate: 0.001
- LR scheduler: ReduceLROnPlateau (factor 0.5, patience 5)
- Loss function: Masked MAE (land pixels excluded)
- Early stopping: Patience 15, min_delta 0.0001
- Device: CUDA (if available)

**Baselines**:
- Persistence: SIC(t+1) = SIC(t)
- Climatology: Day-of-year average with 15-day smoothing

## Performance Metrics

### Baseline Performance (Validation Set)

| Model | MAE | RMSE | Correlation | Ice Edge Disp. |
|-------|-----|------|-------------|----------------|
| Persistence | [TBD] | [TBD] | [TBD] | [TBD] |
| Climatology | [TBD] | [TBD] | [TBD] | [TBD] |

### U-Net Performance

**Validation Set**:
| Metric | Value | vs Best Baseline (Persistence) |
|--------|-------|------------------|
| MAE | 0.007710 | -55.4% |
| Best Epoch | 9 | - |

**Test Set**:
| Metric | Value | vs Best Baseline (Persistence) |
|--------|-------|------------------|
| MAE | 0.007685 | -54.9% vs Persistence (0.004961) / +79.0% vs Climatology (0.036629) |
| RMSE | 0.018257 | -55.1% vs Persistence (0.011774) / +76.6% vs Climatology (0.078060) |
| Spatial Correlation | 0.998708 | 0.999330 (Persistence) / 0.982627 (Climatology) |
| Ice Edge Displacement (px) | 0.7997 (19.99 km) | 0.2317 px (5.79 km) / 5.5543 px (138.86 km) |

**Training**:
- Training time: 0.04 hours (10 epochs, 14.8s/epoch on Apple MPS)
- Best epoch: 9
- Final training loss: 0.008012
- Best validation loss: 0.007710

## Files

**Model Checkpoints**:
- `models/sic_unet_v001_best.pt` - Best model by validation loss
- `models/sic_unet_v001_final.pt` - Final model after training

**Training Artifacts**:
- `models/sic_unet_v001_history.json` - Loss curves, LR schedule
- `output/plots/sic_unet_v001_training_history.png` - Visualization

**Evaluation Results**:
- `output/model_results_val.json` - Validation metrics
- `output/model_results_test.json` - Test metrics
- `output/baseline_results_val.json` - Baseline comparison (val)
- `output/baseline_results_test.json` - Baseline comparison (test)

**Visualizations**:
- `output/plots/metric_comparison_test.png` - Bar chart vs baselines
- `output/plots/best_prediction_*.png` - Best forecasts
- `output/plots/worst_prediction_*.png` - Worst forecasts
- `output/plots/error_map_test.png` - Spatial error distribution
- `output/plots/ice_edge_comparison_test.png` - Ice edge accuracy

**Processed Data**:
- `data/processed/land_ocean_mask.npy` - Land-ocean mask
- `data/processed/{train,val,test}_data.npz` - Input/target arrays
- `data/processed/{train,val,test}_metadata.json` - Dates
- `data/processed/normalization_stats.json` - Mean/std

## Known Issues & Limitations

### Phase 1 Limitations

1. **SIC-only input**: Does not use environmental variables (wind, SST, currents)
2. **Single-day forecast**: Only predicts t+1, not multi-day horizons
3. **No temporal architecture**: U-Net processes each window independently
4. **No uncertainty quantification**: Single deterministic prediction

### Data Limitations

1. **Missing data**: Some dates may be missing in source NSIDC data
2. **Satellite gaps**: Polar night and sensor artifacts
3. **Coastal regions**: Potential land contamination near coastlines
4. **Ice edge blur**: 25km resolution smooths sharp ice boundaries

### Model Limitations

1. **Smooth predictions**: MAE loss tends to average, missing sharp gradients
2. **Land mask dependency**: Performance sensitive to mask quality
3. **Seasonal bias**: May perform differently in summer vs winter
4. **Regional variation**: Not evaluated per-region in Phase 1

## Improvements for Next Version

### Phase 2 Enhancements (v002)
- [ ] Add environmental variables (wind u/v, SST, currents)
- [ ] Multi-channel input: [7 timesteps × 7 variables]
- [ ] Evaluate regional performance breakdown
- [ ] Add seasonal performance analysis

### Phase 3 Enhancements (v003)
- [ ] Replace U-Net with ConvLSTM for temporal modeling
- [ ] Better capture SIC evolution patterns
- [ ] Potentially better long-term accuracy

### Phase 4 Enhancements (v004)
- [ ] Multi-horizon forecasting (t+1 to t+7)
- [ ] Quantify error vs lead time
- [ ] Recursive vs direct forecasting comparison

### General Improvements
- [ ] Uncertainty quantification (ensemble or probabilistic)
- [ ] Perceptual loss for sharper predictions
- [ ] Regional loss weighting (emphasize ice edge)
- [ ] Data augmentation (rotations, flips)
- [ ] Attention mechanisms for important regions

## Usage

### Loading Model for Inference

```python
import torch
from seaice_forecast.models.unet import create_unet_from_config
from seaice_forecast.config import load_config

# Load config and model
config = load_config()
model = create_unet_from_config(config)

# Load weights
checkpoint = torch.load('models/sic_unet_v001_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
with torch.no_grad():
    prediction = model(input_tensor)  # input: [B, 7, H, W]
```

### Reproducing Results

```bash
# Complete workflow
bash scripts/run_phase1.sh

# Or step by step
python scripts/download_data.py --start-date 2010-01-01 --end-date 2022-12-31
python scripts/prepare_data.py
python scripts/evaluate_baselines.py
python scripts/train.py
python scripts/evaluate_model.py --checkpoint models/sic_unet_v001_best.pt --split test
```

## Validation & Testing

### Validation Criteria

**Phase 1 Success Criteria**:
- ✓ Model trains without errors
- ✓ Validation loss decreases consistently
- ✓ U-Net beats both baselines on MAE
- ✓ U-Net beats both baselines on RMSE
- ✓ Spatial correlation > 0.80
- ✓ Ice edge displacement < baseline

**Quality Checks**:
- [ ] No NaN or Inf in predictions
- [ ] All predictions in valid range [0, 1]
- [ ] Land pixels correctly masked in loss
- [ ] No temporal leakage in data splits
- [ ] Consistent performance across seasons

### Known Good Performance

**Target Metrics** (based on similar work):
- MAE: 0.02 - 0.03 (2-3% error)
- RMSE: 0.03 - 0.05
- Correlation: 0.85 - 0.92
- Ice edge: 2-5 pixels (50-125km)

## Change Log

### v001 (Initial Release)
- Initial Phase 1 implementation
- SIC-only input, 7-day window
- U-Net architecture
- Masked MAE loss
- Baseline comparisons (persistence, climatology)
- Comprehensive evaluation metrics

## References

**Data Source**:
- Meier, W. N., et al. (2021). NOAA/NSIDC Climate Data Record of Passive Microwave Sea Ice Concentration, Version 4. Boulder, CO: NSIDC.

**Architecture**:
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

## Contact

For questions about this model version:
- Issue tracker: [TBD]
- Email: [TBD]

---

**Last Updated**: [Auto-filled during training]
**Maintained By**: Research Team
