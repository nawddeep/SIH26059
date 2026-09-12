# Folder Architecture Migration Plan

## Overview

This document tracks the systematic reorganization of both sea-ice forecasting and iceberg drift models.

**Goals:**
- Cleaner, more maintainable structure
- Clear separation between models
- Shared utilities extracted
- Better organization of scripts, docs, and modules

**Method:**
- Use `git mv` to preserve history
- Update imports systematically
- Verify outputs remain identical

---

## Phase 2a: Sea-Ice Forecasting Model Reorganization

### Directory Structure Creation

```bash
# Top-level model directory (rename)
git mv iceberg_forcasting_model seaice_forecast

cd seaice_forecast

# Create new subdirectories
mkdir -p data_processing/downloaders
mkdir -p training
mkdir -p evaluation/analyzers
mkdir -p scripts/data
mkdir -p scripts/training
mkdir -p scripts/evaluation
mkdir -p scripts/visualization
mkdir -p scripts/benchmarking
mkdir -p docs/model_versions
mkdir -p docs/guides
mkdir -p docs/phase_reports
mkdir -p checkpoints
mkdir -p tests
```

### File Moves - Data Processing

```bash
# Downloaders subdirectory
git mv src/seaice_forecast/data_processing/download.py \
        src/seaice_forecast/data_processing/downloaders/nsidc.py

git mv src/seaice_forecast/data_processing/download_era5.py \
        src/seaice_forecast/data_processing/downloaders/era5.py

git mv src/seaice_forecast/data_processing/download_copernicus.py \
        src/seaice_forecast/data_processing/downloaders/copernicus.py

# Rename regridding module
git mv src/seaice_forecast/data_processing/regrid_environmental.py \
        src/seaice_forecast/data_processing/regridding.py

# Rename dataset files for clarity
git mv src/seaice_forecast/data_processing/dataset.py \
        src/seaice_forecast/data_processing/dataset_phase1.py

# dataset_phase2.py stays as-is
```

### File Moves - Models & Evaluation

```bash
# Move trainer from evaluation to training
git mv src/seaice_forecast/evaluation/trainer.py \
        src/seaice_forecast/training/trainer.py

# Create evaluation analyzers subdirectory
# (These will be created by extracting logic from scripts, not moved)
```

### File Moves - Scripts

```bash
# Data scripts
git mv scripts/download_data.py scripts/data/download_all.py
git mv scripts/prepare_data.py scripts/data/prepare_phase1.py
git mv scripts/prepare_environmental_data.py scripts/data/prepare_phase2.py

# Training scripts
git mv scripts/train.py scripts/training/train_phase1.py
git mv scripts/train_phase2.py scripts/training/train_phase2.py

# Evaluation scripts
git mv scripts/evaluate_baselines.py scripts/evaluation/evaluate_baselines.py
git mv scripts/evaluate_model.py scripts/evaluation/evaluate_model.py
git mv scripts/compare_phase1_phase2.py scripts/evaluation/compare_phases.py
git mv scripts/ablation_analysis.py scripts/evaluation/run_ablation.py
git mv scripts/seasonal_regional_analysis.py scripts/evaluation/analyze_seasonal.py

# Visualization scripts
git mv scripts/visualize_predictions.py scripts/visualization/visualize_predictions.py

# Benchmarking scripts
git mv scripts/benchmark_mps_training.py scripts/benchmarking/benchmark_mps.py

# Deprecated (already done)
# scripts/create_demo_data.py -> already deprecated in place
```

### File Moves - Documentation

```bash
# Model versions
git mv MODEL_VERSION.md docs/model_versions/phase1_v001.md
git mv MODEL_VERSION_PHASE2.md docs/model_versions/phase2_env_v001.md

# Guides
git mv GETTING_STARTED.md docs/guides/getting_started.md
git mv MPS_OPTIMIZATION_GUIDE.md docs/guides/mps_optimization.md
git mv MPS_OPTIMIZATION_SUMMARY.md docs/guides/mps_optimization_summary.md

# Phase reports
git mv PHASE1_CHECKLIST.md docs/phase_reports/phase1_checklist.md
git mv PHASE2_README.md docs/phase_reports/phase2_readme.md
git mv PHASE2_CONCLUSION.md docs/phase_reports/phase2_conclusion.md
git mv PHASE2_IMPLEMENTATION_SUMMARY.md docs/phase_reports/phase2_summary.md
git mv PHASE2_FILES_MANIFEST.md docs/phase_reports/phase2_manifest.md

# Keep BUILD_SUMMARY.md and README.md at root
```

### File Moves - Models & Checkpoints

```bash
# Move checkpoints from root models/ to checkpoints/
git mv models/*.pt checkpoints/
git mv models/*.json checkpoints/
```

---

## Phase 2b: Iceberg Drift Model Reorganization

### Directory Structure Creation

```bash
cd ../iceberg_drift_model
# Rename to match pattern
cd ..
git mv iceberg_drift_model iceberg_drift

cd iceberg_drift

# Create new subdirectories
mkdir -p data_processing/downloaders
mkdir -p training
mkdir -p scripts/data
mkdir -p scripts/training
mkdir -p scripts/inference
mkdir -p docs
mkdir -p checkpoints
```

### File Moves - Data Processing

```bash
# Extract real downloaders from download.py (contains synthetic generators)
# This requires code extraction, not just git mv

# Move trainer
git mv src/iceberg_drift/evaluation/trainer.py \
        src/iceberg_drift/training/trainer.py
```

### File Moves - Scripts

```bash
# Organize scripts
git mv scripts/train.py scripts/training/train.py
git mv scripts/train_with_real_data.py scripts/training/train_with_real_data.py
git mv scripts/tune.py scripts/training/tune.py
git mv scripts/predict.py scripts/inference/predict.py
```

---

## Phase 2c: Shared Utilities Extraction

```bash
cd ..
mkdir -p shared

# Extract common grid utilities (requires code extraction from both models)
# Extract common mask utilities
# Extract common visualization patterns
```

---

## Import Path Updates Required

After all moves, need to grep and replace:

### Sea-Ice Forecasting

```python
# Downloaders
from seaice_forecast.data_processing.download import -> from seaice_forecast.data_processing.downloaders.nsidc import
from seaice_forecast.data_processing.download_era5 import -> from seaice_forecast.data_processing.downloaders.era5 import
from seaice_forecast.data_processing.download_copernicus import -> from seaice_forecast.data_processing.downloaders.copernicus import

# Datasets
from seaice_forecast.data_processing.dataset import -> from seaice_forecast.data_processing.dataset_phase1 import

# Regridding
from seaice_forecast.data_processing.regrid_environmental import -> from seaice_forecast.data_processing.regridding import

# Trainer moved to training module
from seaice_forecast.evaluation.trainer import -> from seaice_forecast.training.trainer import
```

### Iceberg Drift

```python
# Trainer moved
from iceberg_drift.evaluation.trainer import -> from iceberg_drift.training.trainer import
```

---

## Verification Steps

After each major move:

1. Check imports still resolve: `python -c "import seaice_forecast; import iceberg_drift"`
2. Check scripts can be found: `python seaice_forecast/scripts/evaluation/evaluate_model.py --help`
3. Run baseline evaluation again to verify identical results

---

## Status Tracking

- [ ] Phase 2a: Sea-ice forecasting reorganization
  - [ ] Create directory structure
  - [ ] Move data_processing files
  - [ ] Move models & evaluation files
  - [ ] Move scripts
  - [ ] Move documentation
  - [ ] Move checkpoints
  - [ ] Update imports
  - [ ] Test imports

- [ ] Phase 2b: Iceberg drift reorganization
  - [ ] Rename directory
  - [ ] Create directory structure
  - [ ] Extract real downloaders
  - [ ] Move trainer
  - [ ] Move scripts
  - [ ] Update imports
  - [ ] Test imports

- [ ] Phase 2c: Extract shared utilities
  - [ ] Create shared/ directory
  - [ ] Extract grid_utils
  - [ ] Extract mask_utils
  - [ ] Update imports in both models

- [ ] Phase 3: Final verification
  - [ ] Re-run baseline evaluation
  - [ ] Compare metrics
  - [ ] Document any differences
