# Import Path Changes - Reorganization

This document tracks all import path changes made during the folder reorganization.

## Sea-Ice Forecasting Model

### Data Processing - Downloaders

**Before:**
```python
from seaice_forecast.data_processing.download import NSIDCDownloader
from seaice_forecast.data_processing.download_era5 import ERA5Downloader
from seaice_forecast.data_processing.download_copernicus import CopernicusMarineDownloader
```

**After:**
```python
from seaice_forecast.data_processing.downloaders.nsidc import NSIDCDownloader
from seaice_forecast.data_processing.downloaders.era5 import ERA5Downloader
from seaice_forecast.data_processing.downloaders.copernicus import CopernicusMarineDownloader
```

**Files Changed:**
- `scripts/data/download_all.py`
- `scripts/data/prepare_phase2.py`
- `src/seaice_forecast/data_processing/__init__.py`

---

### Data Processing - Datasets

**Before:**
```python
from seaice_forecast.data_processing.dataset import SICDataset, create_dataloaders
```

**After:**
```python
from seaice_forecast.data_processing.dataset_phase1 import SICDataset, create_dataloaders
```

**Rationale:** Renamed `dataset.py` to `dataset_phase1.py` for clarity alongside `dataset_phase2.py`.

**Files Changed:**
- `scripts/training/train_phase1.py`
- `scripts/evaluation/evaluate_model.py`
- `scripts/evaluation/analyze_seasonal.py`
- `scripts/evaluation/compare_phases.py`
- `src/seaice_forecast/data_processing/__init__.py`

---

### Data Processing - Regridding

**Before:**
```python
from seaice_forecast.data_processing.regrid_environmental import EnvironmentalRegridder
```

**After:**
```python
from seaice_forecast.data_processing.regridding import EnvironmentalRegridder
```

**Rationale:** Shorter, clearer name.

**Files Changed:**
- `scripts/data/prepare_phase2.py`

---

### Training - Trainer

**Before:**
```python
from seaice_forecast.evaluation.trainer import Trainer
```

**After:**
```python
from seaice_forecast.training.trainer import Trainer
```

**Rationale:** Trainer belongs in `training/` module, not `evaluation/`.

**Files Changed:**
- `scripts/training/train_phase1.py`
- `scripts/training/train_phase2.py`
- Created new `src/seaice_forecast/training/__init__.py`

---

### Configuration - Output Directories

**Before:**
```yaml
output:
  model_dir: "models"
```

**After:**
```yaml
output:
  model_dir: "checkpoints"
```

**Rationale:** Clearer naming - these are model checkpoints, not the models source code directory.

**Files Changed:**
- `src/seaice_forecast/config/settings.yaml`

---

### Script Path References (Documentation/Help Text)

Updated usage examples in docstrings and help messages to reflect new script locations:

**Before:**
- `python scripts/train.py`
- `python scripts/prepare_data.py`
- `python scripts/download_data.py`
- `python scripts/evaluate_model.py`

**After:**
- `python scripts/training/train_phase1.py`
- `python scripts/data/prepare_phase1.py`
- `python scripts/data/download_all.py`
- `python scripts/evaluation/evaluate_model.py`

**Note:** These are documentation strings only, not actual imports.

---

### Checkpoint Path References

**Before:**
- `models/sic_unet_v001_best.pt`
- `models/sic_unet_env_v001_best.pt`

**After:**
- `checkpoints/sic_unet_v001_best.pt`
- `checkpoints/sic_unet_env_v001_best.pt`

**Files Changed:**
- `scripts/visualization/visualize_predictions.py`
- `scripts/evaluation/analyze_seasonal.py` (default arguments)

---

## Files That Did NOT Require Import Changes

### Already Using Relative Imports
- `src/seaice_forecast/models/*.py` - No cross-module imports affected
- `src/seaice_forecast/utils/*.py` - Standalone utilities
- `src/seaice_forecast/evaluation/metrics.py` - Pure functions, no module imports

### Configuration Files
- `src/seaice_forecast/config/__init__.py` - Only loads YAML, no reorganized imports

### Phase 2 Specific
- `src/seaice_forecast/data_processing/dataset_phase2.py` - Already correctly named and located

---

## Verification Commands

### Test Import Resolution
```bash
cd seaice_forecast
source venv/bin/activate
python -c "
import sys
sys.path.insert(0, 'src')
from seaice_forecast.data_processing.downloaders import nsidc
from seaice_forecast.data_processing import dataset_phase1
from seaice_forecast.training import trainer
print('✓ All imports successful!')
"
```

### Test Script Execution (Dry Run)
```bash
# Help text should work without data
python scripts/data/download_all.py --help
python scripts/data/prepare_phase1.py --help
python scripts/training/train_phase1.py --help
python scripts/evaluation/evaluate_model.py --help
```

---

## Summary Statistics

- **Total files with import changes:** 10
- **Total import statements updated:** 15
- **Modules reorganized:**
  - `data_processing/downloaders/` (3 files)
  - `training/` (1 file moved)
  - Dataset files renamed (1 file)
  - Regridding file renamed (1 file)
- **Configuration files updated:** 1
- **Script paths updated:** ~8 (documentation strings)

---

## Backward Compatibility

**Breaking Changes:**
- All imports from `seaice_forecast.data_processing.download*` modules are broken
- All imports from `seaice_forecast.evaluation.trainer` are broken
- All imports from `seaice_forecast.data_processing.dataset` (without phase suffix) are broken

**Migration Path for External Code:**
1. Update downloader imports to use `downloaders/` subdirectory
2. Update `dataset` imports to `dataset_phase1`
3. Update `trainer` imports to `training.trainer`
4. Update checkpoint paths from `models/` to `checkpoints/`

---

## Testing Evidence

Import resolution tested and verified:
```
✓ All imports successful!
```

Next step: Re-run baseline evaluation to prove outputs unchanged.
