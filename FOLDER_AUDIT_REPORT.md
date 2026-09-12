# Folder Architecture Audit Report

## Current Structure Analysis

### Sea-Ice Forecasting Model (`iceberg_forcasting_model/`)

**Scripts (13 files)**:
1. `scripts/ablation_analysis.py` - Phase 2: Variable importance testing
2. `scripts/benchmark_mps_training.py` - MPS optimization benchmarking
3. `scripts/compare_phase1_phase2.py` - Phase 1 vs Phase 2 comparison
4. `scripts/create_demo_data.py` - **SYNTHETIC DATA GENERATOR** (not clearly marked)
5. `scripts/download_data.py` - NSIDC SIC data download
6. `scripts/evaluate_baselines.py` - Persistence/climatology baseline evaluation
7. `scripts/evaluate_model.py` - Model evaluation
8. `scripts/prepare_data.py` - Phase 1 data preparation
9. `scripts/prepare_environmental_data.py` - Phase 2 data prep + **SYNTHETIC GENERATOR**
10. `scripts/seasonal_regional_analysis.py` - Seasonal/regional breakdown
11. `scripts/train.py` - Phase 1 training
12. `scripts/train_phase2.py` - Phase 2 training
13. `scripts/visualize_predictions.py` - Prediction visualization

**Data Processing (7 files)**:
1. `data_processing/__init__.py`
2. `data_processing/dataset.py` - Phase 1 (7-channel) dataset
3. `data_processing/dataset_phase2.py` - Phase 2 (49-channel) dataset
4. `data_processing/download.py` - NSIDC downloader
5. `data_processing/download_copernicus.py` - CMEMS ocean currents
6. `data_processing/download_era5.py` - ERA5 reanalysis
7. `data_processing/preprocessing.py` - Preprocessing utils
8. `data_processing/regrid_environmental.py` - Regridding to PS25 grid

**Models (3 files)**:
1. `models/__init__.py`
2. `models/baselines.py` - Persistence/climatology models
3. `models/unet.py` - U-Net (parameterized 7 or 49 channels)

**Evaluation (5 files)**:
1. `evaluation/__init__.py`
2. `evaluation/baselines_eval.py` - Baseline evaluation
3. `evaluation/metrics.py` - MAE, RMSE, correlation, ice edge
4. `evaluation/model_eval.py` - Model evaluation
5. `evaluation/trainer.py` - Training loop, loss, early stopping

**Utils (3 files)**:
1. `utils/__init__.py`
2. `utils/device_utils.py` - MPS/CUDA/CPU device selection
3. `utils/visualization.py` - Plotting utilities

### Iceberg Drift Model (`iceberg_drift_model/`)

**Scripts (4 files)**:
1. `scripts/predict.py` - Inference
2. `scripts/train.py` - Training
3. `scripts/train_with_real_data.py` - Training with real data
4. `scripts/tune.py` - Hyperparameter tuning

**Data Processing (5 files)**:
1. `data_processing/__init__.py`
2. `data_processing/dataset.py` - Dataset class
3. `data_processing/download.py` - Downloads + **SYNTHETIC GENERATORS** (4 functions)
4. `data_processing/preprocessing.py` - Preprocessing
5. `data_processing/quality.py` - Quality checks

**Models (6 files)**:
1. `models/__init__.py`
2. `models/ensemble.py` - Ensemble methods
3. `models/gbm_models.py` - Gradient boosting
4. `models/ml_models.py` - ML models
5. `models/physics_model.py` - Physics-based
6. `models/pinn.py` - Physics-informed neural networks

**Evaluation (4 files)**:
1. `evaluation/__init__.py`
2. `evaluation/metrics.py` - Metrics
3. `evaluation/trainer.py` - Training
4. `evaluation/validator.py` - Validation

**Utils (4 files)**:
1. `utils/__init__.py`
2. `utils/export.py` - Export utilities
3. `utils/inference.py` - Inference utilities
4. `utils/visualization.py` - Visualization

**Tests (4 files)**:
1. `tests/test_data_processing.py`
2. `tests/test_evaluation.py`
3. `tests/test_models.py`
4. `tests/test_quality.py`

---

## Issues Identified

### 1. Synthetic Data Generators - NOT CLEARLY SEPARATED ⚠️

**Sea-Ice Forecasting**:
- `scripts/create_demo_data.py` - Function: `generate_synthetic_sic()`, `create_demo_dataset()`
  - NO warning in filename
  - NO explicit flag required
  - Could be mistaken for real data preparation

- `scripts/prepare_environmental_data.py` - Function: `generate_coupled_synthetic_data()`
  - Lines 585-707: Large synthetic generator embedded in production code
  - Called automatically as fallback when real data missing (lines 901-923)
  - **CRITICAL**: Silent fallback to synthetic without explicit user consent

**Iceberg Drift**:
- `data_processing/download.py` - Functions:
  - `_generate_synthetic_icebergs()` (line 379)
  - `_generate_synthetic_era5()` (line 534)
  - `_generate_synthetic_currents()` (line 621)
  - `_generate_synthetic_bathymetry()` (line 701)
  - Mixed with real download code in same file
  - Used in tests but not clearly separated

### 2. Download Scripts - Duplicate Naming

**Sea-Ice Forecasting has 3 downloaders**:
- `download.py` (NSIDC SIC)
- `download_era5.py` (ERA5)
- `download_copernicus.py` (CMEMS)

**Better**: `downloaders/` directory with `nsidc.py`, `era5.py`, `copernicus.py`

### 3. Evaluation Scripts - Scattered

**Baselines**:
- `scripts/evaluate_baselines.py` (entry point)
- `evaluation/baselines_eval.py` (logic)
- `models/baselines.py` (model definitions)

**Model evaluation**:
- `scripts/evaluate_model.py` (entry point)
- `evaluation/model_eval.py` (logic)

### 4. Phase-Specific Scripts - No Clear Organization

**Phase 1**:
- `scripts/prepare_data.py`
- `scripts/train.py`
- `data_processing/dataset.py`

**Phase 2**:
- `scripts/prepare_environmental_data.py`
- `scripts/train_phase2.py`
- `data_processing/dataset_phase2.py`

**Phase Comparison**:
- `scripts/compare_phase1_phase2.py`
- `scripts/ablation_analysis.py`
- `scripts/seasonal_regional_analysis.py`

### 5. Config Files - Scattered

- `src/seaice_forecast/config/settings.yaml` - Main config
- Possibly others in different locations

### 6. No Shared Code Between Models

Both models likely share:
- Polar stereographic grid utilities
- Land/ocean mask generation
- Some visualization patterns

Currently duplicated.

### 7. Documentation - Mixed with Code

**Sea-Ice Forecasting root has**:
- `MODEL_VERSION.md`
- `MODEL_VERSION_PHASE2.md`
- `PHASE1_CHECKLIST.md`
- `PHASE2_README.md`
- `PHASE2_CONCLUSION.md`
- `PHASE2_IMPLEMENTATION_SUMMARY.md`
- `PHASE2_FILES_MANIFEST.md`
- `MPS_OPTIMIZATION_GUIDE.md`
- `MPS_OPTIMIZATION_SUMMARY.md`
- `BUILD_SUMMARY.md`
- `GETTING_STARTED.md`
- `README.md`

Should be in `docs/` directory.

### 8. Orphaned Files - To Be Determined

Need to check imports to identify files not used in any pipeline.

---

## Synthetic Data Usage Analysis

### Current Behavior (PROBLEMATIC)

**`prepare_environmental_data.py` line 901-923**:
```python
if synthetic or (not download and not has_raw):
    logger.info("Running synthetic coupled environmental data pipeline...")
    return self.generate_coupled_synthetic_data()

# ... later ...

if not mask_file.exists():
    logger.error("Land/ocean mask not found - creating mask first")
    self.generate_coupled_synthetic_data(n_train_days=10, n_val_days=5, n_test_days=5)

# ... later ...

if not sic_files_list:
    logger.warning("No SIC reference files found, falling back to synthetic generator")
    return self.generate_coupled_synthetic_data()
```

**Problem**: Automatically falls back to synthetic data if real data missing, with only a warning. User may not realize they're training on fake data.

### Required Fix

1. Extract synthetic generators to `dev_tools/synthetic_data_generators/`
2. Rename files/functions to include "FAKE" or "SYNTHETIC_DEV_ONLY"
3. Require explicit `--use-synthetic-data --i-understand-this-is-fake` flag
4. Remove automatic fallbacks
5. Raise error with clear message if real data missing

---

## Proposed Target Structure

```
iceberg_models/
├── seaice_forecast/                      # Sea-ice forecasting model
│   ├── data_processing/
│   │   ├── downloaders/
│   │   │   ├── __init__.py
│   │   │   ├── nsidc.py                  # Renamed from download.py
│   │   │   ├── era5.py                   # Renamed from download_era5.py
│   │   │   └── copernicus.py             # Renamed from download_copernicus.py
│   │   ├── __init__.py
│   │   ├── preprocessing.py              # Regridding, masking, normalization
│   │   ├── regridding.py                 # Renamed from regrid_environmental.py
│   │   ├── dataset_phase1.py             # Renamed from dataset.py
│   │   └── dataset_phase2.py             # Keep name
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py                       # Keep as-is
│   │   └── baselines.py                  # Keep as-is
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                    # Move from evaluation/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                    # Keep as-is
│   │   ├── evaluator.py                  # Merge baselines_eval + model_eval
│   │   └── analyzers/
│   │       ├── __init__.py
│   │       ├── ablation.py               # Logic from ablation_analysis.py
│   │       └── seasonal_regional.py      # Logic from seasonal_regional_analysis.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── device_utils.py               # Keep as-is
│   │   └── visualization.py              # Keep as-is
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.yaml                 # Single source of truth
│   ├── scripts/                          # Thin entry points only
│   │   ├── data/
│   │   │   ├── download_all.py           # Rename from download_data.py
│   │   │   ├── prepare_phase1.py         # Rename from prepare_data.py
│   │   │   └── prepare_phase2.py         # Rename from prepare_environmental_data.py
│   │   ├── training/
│   │   │   ├── train_phase1.py           # Rename from train.py
│   │   │   └── train_phase2.py           # Keep name
│   │   ├── evaluation/
│   │   │   ├── evaluate_baselines.py     # Keep name
│   │   │   ├── evaluate_model.py         # Keep name
│   │   │   ├── compare_phases.py         # Rename from compare_phase1_phase2.py
│   │   │   ├── run_ablation.py           # Rename from ablation_analysis.py
│   │   │   └── analyze_seasonal.py       # Rename from seasonal_regional_analysis.py
│   │   ├── visualization/
│   │   │   └── visualize_predictions.py  # Keep name
│   │   └── benchmarking/
│   │       └── benchmark_mps.py          # Rename from benchmark_mps_training.py
│   ├── checkpoints/                      # Model .pt files
│   ├── docs/                             # All documentation
│   │   ├── model_versions/
│   │   │   ├── phase1_v001.md            # Rename from MODEL_VERSION.md
│   │   │   └── phase2_env_v001.md        # Rename from MODEL_VERSION_PHASE2.md
│   │   ├── guides/
│   │   │   ├── getting_started.md
│   │   │   └── mps_optimization.md       # Merge MPS docs
│   │   ├── phase_reports/
│   │   │   ├── phase1_checklist.md
│   │   │   ├── phase2_readme.md
│   │   │   ├── phase2_conclusion.md
│   │   │   └── phase2_summary.md
│   │   └── README.md
│   └── tests/                            # Add tests directory
│       └── __init__.py
│
├── iceberg_drift/                        # Iceberg drift model (mirror structure)
│   ├── data_processing/
│   │   ├── downloaders/
│   │   │   ├── __init__.py
│   │   │   └── real_data_downloader.py   # Extracted from download.py
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── quality.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ensemble.py
│   │   ├── gbm_models.py
│   │   ├── ml_models.py
│   │   ├── physics_model.py
│   │   └── pinn.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                    # Move from evaluation/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── validator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── export.py
│   │   ├── inference.py
│   │   └── visualization.py
│   ├── config/
│   │   └── settings.yaml
│   ├── scripts/
│   │   ├── train.py
│   │   ├── train_with_real_data.py
│   │   ├── predict.py
│   │   └── tune.py
│   ├── checkpoints/
│   ├── docs/
│   └── tests/                            # Keep existing tests
│       ├── __init__.py
│       ├── test_data_processing.py
│       ├── test_evaluation.py
│       ├── test_models.py
│       └── test_quality.py
│
├── shared/                               # Shared utilities
│   ├── __init__.py
│   ├── grid_utils.py                     # Polar stereographic grid helpers
│   ├── mask_utils.py                     # Land/ocean mask utilities
│   └── visualization_common.py           # Shared plotting utilities
│
├── dev_tools/                            # Development tools (NOT production)
│   ├── __init__.py
│   └── synthetic_data/
│       ├── __init__.py
│       ├── SYNTHETIC_seaice_generator.py # Renamed from create_demo_data.py
│       └── SYNTHETIC_iceberg_generator.py # Extracted from iceberg download.py
│
├── data/
│   ├── raw/
│   │   ├── real/                         # Only real downloads
│   │   │   ├── nsidc/
│   │   │   ├── era5/
│   │   │   └── copernicus/
│   │   └── SYNTHETIC_DEV_ONLY/           # Clearly separated
│   └── processed/
│       ├── phase1/
│       └── phase2/
│
├── output/
│   ├── plots/
│   ├── results/
│   └── benchmarks/
│
├── configs/                              # If multiple configs needed
│
└── README.md                             # Top-level readme

```

---

## Changes Required

### High Priority (Safety)

1. **Extract and rename synthetic generators**
2. **Remove automatic fallbacks to synthetic**
3. **Add explicit flag requirement**

### Medium Priority (Clarity)

4. **Reorganize downloaders into subdirectory**
5. **Move documentation to docs/**
6. **Separate phase-specific scripts**
7. **Create training/ subdirectory**

### Low Priority (Consistency)

8. **Mirror structure between models**
9. **Extract shared utilities**
10. **Consolidate evaluation modules**

---

## Migration Plan

### Phase 1: Safety First (Synthetic Data)
1. Create `dev_tools/synthetic_data/`
2. Extract and rename synthetic generators
3. Update all imports
4. Add explicit flag requirements
5. Remove automatic fallbacks
6. Test that real data paths still work

### Phase 2: Reorganization
1. Create target directory structure
2. Use `git mv` for all file moves
3. Update all import statements
4. Update all documentation references
5. Test that everything still works

### Phase 3: Verification
1. Run Phase 1 evaluation on existing checkpoint
2. Compare before/after outputs
3. Verify identical results
4. Document any differences

---

## Testing Strategy

### Before Migration
```bash
# Run Phase 1 evaluation
python scripts/evaluate_model.py --checkpoint models/sic_unet_v001_best.pt --split test
# Save output as baseline_metrics.json
```

### After Migration
```bash
# Run same evaluation with new structure
python seaice_forecast/scripts/evaluation/evaluate_model.py --checkpoint seaice_forecast/checkpoints/sic_unet_v001_best.pt --split test
# Compare with baseline_metrics.json
```

### Verification
- Exact match on MAE, RMSE, Correlation, Ice Edge Displacement
- Exact match on sample predictions
- If any difference: stop and debug before proceeding

---

## Files to Flag for Follow-up (Not Touching Now)

1. Potential code duplication in evaluation modules
2. Possible consolidation of dataset classes
3. Config file duplication between models
4. Test coverage gaps in sea-ice forecasting
5. Documentation cleanup and consolidation

---

**Status**: Audit Complete
**Next**: Await approval of target structure before beginning migration
