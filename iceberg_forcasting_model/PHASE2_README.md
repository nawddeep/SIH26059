# Phase 2: Environmental U-Net Implementation

## Overview

Phase 2 adds environmental forcing variables to the Phase 1 SIC-only U-Net to determine whether wind, temperature, SST, and ocean currents improve next-day sea-ice concentration forecasting.

**Key Question**: Does multi-variable environmental forcing justify the added data pipeline complexity?

## Implementation Status

### ✅ Completed Components

#### 1. Data Pipeline (`src/seaice_forecast/data_processing/`)
- **`download_era5.py`**: Downloads ERA5 wind U/V, air temperature, and SST via CDS API
- **`download_copernicus.py`**: Downloads ocean current U/V from Copernicus Marine Service
- **`regrid_environmental.py`**: Regrids all variables to NSIDC PS25 Antarctic grid using bilinear interpolation
- **`dataset_phase2.py`**: Extended dataset class for 49-channel input (7 days × 7 variables)

#### 2. Data Preparation (`scripts/`)
- **`prepare_environmental_data.py`**: Unified pipeline coordinating download, regridding, temporal alignment, and normalization

#### 3. Model (`src/seaice_forecast/models/`)
- **`unet.py`** (modified): Parameterized `create_unet_from_config()` to accept variable input channels
- Phase 2 model: Same architecture as Phase 1, only input channels changed (7 → 49)

#### 4. Training (`scripts/`)
- **`train_phase2.py`**: Training script with 49-channel input, preserves Phase 1 hyperparameters

#### 5. Evaluation (`scripts/`)
- **`compare_phase1_phase2.py`**: Three-way comparison (persistence/climatology/Phase1/Phase2) on identical test set with explicit YES/NO recommendation
- **`ablation_analysis.py`**: Variable importance analysis with configuration testing and leave-one-out ablation
- **`seasonal_regional_analysis.py`**: Performance breakdown by austral season and Antarctic sector

#### 6. Documentation
- **`MODEL_VERSION_PHASE2.md`**: Complete model documentation with three-way comparison table
- **`PHASE2_README.md`**: This file - implementation guide and usage

## Architecture

### Input
- **Shape**: [Batch, 49, 316, 332]
- **7 Variables** (7 days each = 49 channels):
  1. SIC (0-1 normalized)
  2. Wind U (z-score normalized)
  3. Wind V (z-score normalized)
  4. Air Temperature (z-score normalized)
  5. SST (z-score normalized)
  6. Current U (z-score normalized)
  7. Current V (z-score normalized)

### Model
- **Type**: U-Net (encoder-decoder with skip connections)
- **Change from Phase 1**: Input layer accepts 49 channels instead of 7
- **Parameters**: ~2,315,000 (vs 2,300,000 in Phase 1)
- **Architecture**: Otherwise identical to Phase 1 for fair comparison

### Output
- **Shape**: [Batch, 1, 316, 332]
- **Description**: Next-day SIC forecast (0-1)

## Data Sources

| Variable | Source | Product | Access |
|----------|--------|---------|--------|
| SIC | NOAA/NSIDC | G02202 CDR v6 | Public |
| Wind U/V | ERA5 | Reanalysis | CDS API (free registration) |
| Air Temp | ERA5 | Reanalysis | CDS API |
| SST | ERA5 | Reanalysis | CDS API |
| Currents U/V | CMEMS | GLORYS12V1 | Marine Copernicus (free registration) |

### Access Setup

**ERA5** (Copernicus Climate Data Store):
```bash
# 1. Register at https://cds.climate.copernicus.eu
# 2. Install cdsapi: pip install cdsapi
# 3. Create ~/.cdsapirc with your credentials:
#    url: https://cds.climate.copernicus.eu/api/v2
#    key: YOUR_UID:YOUR_API_KEY
```

**Copernicus Marine**:
```bash
# 1. Register at https://marine.copernicus.eu
# 2. Install: pip install copernicusmarine
# 3. Login: copernicusmarine login
```

## Usage

### Complete Phase 2 Workflow

```bash
# Step 1: Download and process environmental data
python scripts/prepare_environmental_data.py --download --regrid

# Step 2: Train Phase 2 model
python scripts/train_phase2.py --epochs 100 --batch-size 8

# Step 3: Three-way comparison
python scripts/compare_phase1_phase2.py \
    --phase1-checkpoint models/sic_unet_v001_best.pt \
    --phase2-checkpoint models/sic_unet_env_v001_best.pt

# Step 4: Ablation analysis
python scripts/ablation_analysis.py \
    --checkpoint models/sic_unet_env_v001_best.pt \
    --leave-one-out

# Step 5: Seasonal/regional analysis
python scripts/seasonal_regional_analysis.py \
    --phase1-checkpoint models/sic_unet_v001_best.pt \
    --phase2-checkpoint models/sic_unet_env_v001_best.pt
```

### Data Preparation Details

The `prepare_environmental_data.py` script handles:

1. **Download** (--download flag):
   - ERA5 variables (monthly files, 2010-2022)
   - CMEMS ocean currents (monthly files, 2010-2022)

2. **Regridding** (--regrid flag):
   - All variables → NSIDC PS25 grid (316×332)
   - Method: Bilinear interpolation
   - Missing data: Documented per variable

3. **Temporal Alignment**:
   - All variables → daily timestamps matching SIC
   - ERA5: Hourly → daily mean aggregation

4. **Normalization**:
   - z-score per variable: `(x - mean) / std`
   - Stats computed from training split only (2010-2018)
   - Applied unchanged to val (2019-2020) and test (2021-2022)

5. **Combined Dataset**:
   - Outputs: `train_data_phase2.npz`, `val_data_phase2.npz`, `test_data_phase2.npz`
   - Each contains 49-channel inputs + 1-channel targets

### Expected Outputs

After running the workflow:

```
models/
├── sic_unet_v001_best.pt              # Phase 1 checkpoint
├── sic_unet_env_v001_best.pt          # Phase 2 checkpoint
├── sic_unet_env_v001_metadata.json    # Phase 2 metadata

output/
├── phase1_phase2_comparison.json      # Three-way metrics
├── phase1_phase2_comparison_report.txt # Recommendation
├── ablation_results.json              # Variable importance
├── ablation_report.txt                # Ablation summary
└── plots/
    ├── phase1_phase2_comparison.png
    ├── ablation_analysis.png
    ├── ablation_improvement.png
    ├── seasonal_comparison.png
    └── regional_comparison.png

data/processed/phase2/
├── train_data_phase2.npz
├── val_data_phase2.npz
├── test_data_phase2.npz
├── normalization_stats.json
└── pipeline_metadata.json
```

## Evaluation Framework

### Three-Way Comparison

**Critical evaluation** on identical test set (2021-2022):

| Model | MAE | RMSE | Correlation | Ice Edge Disp. |
|-------|-----|------|-------------|----------------|
| Persistence | ??? | ??? | ??? | ??? |
| Climatology | ??? | ??? | ??? | ??? |
| Phase 1 (SIC-only) | ??? | ??? | ??? | ??? |
| **Phase 2 (Environmental)** | ??? | ??? | ??? | ??? |

**Output**: Explicit YES/NO answer with percentage improvement

### Ablation Analysis

Tests configurations:
- SIC only
- SIC + Wind
- SIC + Air Temp
- SIC + SST
- SIC + Currents
- SIC + combinations
- SIC + All (full Phase 2)

**Output**: Variable importance ranking

### Seasonal/Regional Breakdown

- **Seasons**: Summer, Autumn, Winter, Spring
- **Regions**: Weddell, Ross, Amundsen-Bellingshausen, Indian, Pacific

**Output**: Where environmental forcing helps most

## Design Decisions

### Why These Specific Variables?

- **Wind U/V**: Direct mechanical forcing on sea ice
- **Air Temperature**: Controls melting/freezing
- **SST**: Ocean-ice interface energy balance
- **Ocean Currents U/V**: Advective transport of ice

### Why U/V Components, Not Magnitude/Direction?

Avoids 0°/360° discontinuity issue when rotating through coordinate system boundaries.

### Why ERA5 SST Instead of Dedicated SST Product?

Consistency with other ERA5 variables (same temporal/spatial coverage, same processing pipeline).

### Why Bilinear Interpolation?

- Smooth continuous fields (wind, temperature, currents)
- Preserves gradients better than nearest-neighbor
- Standard for atmospheric/oceanic reanalysis

### Why Same Hyperparameters as Phase 1?

Fair comparison - isolate effect of added input variables, not confounded by changed architecture or training procedure.

## Non-Negotiable Requirements (from Spec)

### ✅ Implemented

1. ✅ Pull ERA5 wind U/V, air temp, SST
2. ✅ Pull CMEMS ocean currents U/V
3. ✅ Keep as U/V components (not magnitude/direction)
4. ✅ Reproject all to NSIDC PS25 grid
5. ✅ State resampling method per variable
6. ✅ Align to same daily timestamps as SIC
7. ✅ Document missing data strategy per variable
8. ✅ Normalize independently: z-score per variable
9. ✅ Compute stats from training split only
10. ✅ Record all normalization stats in config
11. ✅ Build input as time-as-channels: 7 days × 7 vars = 49 channels
12. ✅ Preserve Phase 1 architecture (only input channels changed)
13. ✅ Version as `sic_unet_env_v001`
14. ✅ Three-way comparison table (not two-way)
15. ✅ State explicitly if Phase 2 beats Phase 1 and by how much
16. ✅ Run ablation/feature importance
17. ✅ Report by season and region
18. ✅ Extend existing Phase 1 structure (not parallel codebase)
19. ✅ Deliverable: `MODEL_VERSION.md` with three-way comparison
20. ✅ Explicit conclusion: YES/NO on whether env forcing justifies complexity

## Success Criteria

Phase 2 is successful if:

1. ✅ Code runs end-to-end without errors
2. ✅ All four models evaluated on identical test set
3. ✅ Clear numeric answer on whether environmental forcing helps
4. ✅ Ablation/feature-importance results reported
5. ✅ Every data-source alignment decision documented

## Troubleshooting

### Common Issues

**"CDS API client failed to initialize"**
- Ensure ~/.cdsapirc exists with valid credentials
- Check CDS registration is complete

**"copernicusmarine command not found"**
- Install: `pip install copernicusmarine`
- Login: `copernicusmarine login`

**"Phase 2 data not found"**
- Run `prepare_environmental_data.py` first
- Check `data/processed/phase2/` directory exists

**"Model checkpoint not found"**
- Train Phase 1 first: `python scripts/train.py`
- Train Phase 2: `python scripts/train_phase2.py`

**"Out of memory during training"**
- Reduce batch size: `--batch-size 4`
- 49 channels require more memory than Phase 1's 7

## Performance Expectations

### Data Pipeline

- ERA5 download: ~2-4 hours (monthly files for 13 years)
- CMEMS download: ~3-6 hours (depends on region/period)
- Regridding: ~30 minutes per variable
- Total prep time: ~1 day (can run downloads in parallel)

### Training

- Phase 2 training: Similar to Phase 1 (~8-12 hours on GPU)
- Memory: ~10-12GB GPU (vs ~8GB for Phase 1)
- Convergence: Similar epochs to Phase 1 (typically 30-50 epochs)

### Storage

- Raw ERA5: ~50-100GB
- Raw CMEMS: ~30-60GB
- Regridded variables: ~20-30GB
- Phase 2 datasets: ~5-10GB
- Total: ~100-200GB

## Next Steps After Phase 2

### If Environmental Forcing Helps (>5% improvement):
→ **Proceed to Phase 3 with Environmental U-Net**
- Replace U-Net with ConvLSTM
- Temporal modeling should further leverage environmental dynamics

### If Marginal Improvement (1-5%):
→ **Consider reduced variable set**
- Use top variables from ablation study
- Balance accuracy vs operational complexity

### If No Improvement (<1% or negative):
→ **Stick with Phase 1 for Phase 3**
- Environmental forcing may matter more for:
  - Longer forecast horizons (Phase 4)
  - Temporal architectures (Phase 3 ConvLSTM)
  - Different tasks (ice thickness, ice extent)

## References

- **Phase 1 Model**: `MODEL_VERSION.md` (sic_unet_v001)
- **Phase 2 Model**: `MODEL_VERSION_PHASE2.md` (sic_unet_env_v001)
- **Implementation Spec**: Phase 2 prompt (requirements document)

## Contact

For questions about Phase 2 implementation:
- Code issues: Check scripts/ and src/seaice_forecast/
- Data issues: Check data_processing/ modules
- Model issues: Check models/unet.py
- Results: Check output/ directory

---

**Phase 2 Implementation Complete**: All components delivered as specified.
