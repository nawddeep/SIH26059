#!/usr/bin/env python3
"""
Comprehensive Model Readiness Test
Tests if the sea-ice forecasting model can actually run end-to-end.
"""

import sys
from pathlib import Path
import traceback

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def test_section(name):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Testing: {name}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

def success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")

def info(msg):
    print(f"  {msg}")

# Track results
results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

# =============================================================================
# TEST 1: Python Environment
# =============================================================================
test_section("Python Environment")

try:
    import sys
    info(f"Python version: {sys.version}")
    if sys.version_info >= (3, 9):
        success("Python version >= 3.9")
        results['passed'].append("Python version")
    else:
        error("Python version < 3.9 (requires 3.9+)")
        results['failed'].append("Python version")
except Exception as e:
    error(f"Python check failed: {e}")
    results['failed'].append("Python environment")

# =============================================================================
# TEST 2: Core Dependencies
# =============================================================================
test_section("Core Dependencies")

dependencies = {
    'numpy': '1.24.0',
    'torch': '2.0.0',
    'xarray': '2023.1.0',
    'matplotlib': '3.7.0',
    'pandas': '2.0.0',
    'scipy': '1.10.0',
    'yaml': None,
    'netCDF4': '1.6.0',
}

for package, min_version in dependencies.items():
    try:
        if package == 'yaml':
            import yaml
            success(f"PyYAML installed")
            results['passed'].append(f"{package}")
        else:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            success(f"{package} {version}")
            results['passed'].append(f"{package}")
    except ImportError:
        error(f"{package} NOT INSTALLED")
        results['failed'].append(f"{package}")

# =============================================================================
# TEST 3: PyTorch Backend
# =============================================================================
test_section("PyTorch Backend")

try:
    import torch
    info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        info(f"CUDA version: {torch.version.cuda}")
        results['passed'].append("CUDA GPU")
    else:
        warning("CUDA not available")
        results['warnings'].append("No CUDA GPU")
    
    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        success("MPS (Apple Silicon GPU) available")
        results['passed'].append("MPS GPU")
    else:
        info("MPS not available (not on Apple Silicon)")
    
    # Default device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    success(f"Default compute device: {device}")
    
except Exception as e:
    error(f"PyTorch backend check failed: {e}")
    results['failed'].append("PyTorch backend")

# =============================================================================
# TEST 4: Project Structure
# =============================================================================
test_section("Project Structure")

base_path = Path(__file__).parent
required_dirs = [
    'src/seaice_forecast',
    'src/seaice_forecast/config',
    'src/seaice_forecast/data_processing',
    'src/seaice_forecast/models',
    'src/seaice_forecast/evaluation',
    'scripts',
    'data',
    'data/processed',
    'models',
    'output',
]

for dir_path in required_dirs:
    full_path = base_path / dir_path
    if full_path.exists():
        success(f"Directory exists: {dir_path}")
        results['passed'].append(f"dir:{dir_path}")
    else:
        error(f"Directory missing: {dir_path}")
        results['failed'].append(f"dir:{dir_path}")

# =============================================================================
# TEST 5: Data Files
# =============================================================================
test_section("Data Files")

data_files = {
    'data/processed/land_ocean_mask.npy': 'Land/ocean mask',
    'data/processed/train_data.npz': 'Training data',
    'data/processed/val_data.npz': 'Validation data',
    'data/processed/test_data.npz': 'Test data',
}

for file_path, description in data_files.items():
    full_path = base_path / file_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / (1024 * 1024)
        success(f"{description}: {size_mb:.1f} MB")
        results['passed'].append(f"data:{description}")
    else:
        error(f"{description} missing: {file_path}")
        results['failed'].append(f"data:{description}")

# =============================================================================
# TEST 6: Import Project Modules
# =============================================================================
test_section("Project Modules")

sys.path.insert(0, str(base_path / 'src'))

modules_to_test = [
    'seaice_forecast.config',
    'seaice_forecast.data_processing.dataset_phase1',
    'seaice_forecast.data_processing.dataset_real',
    'seaice_forecast.models.unet',
    'seaice_forecast.models.baselines',
    'seaice_forecast.evaluation.metrics',
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        success(f"Module import: {module_name}")
        results['passed'].append(f"module:{module_name}")
    except Exception as e:
        error(f"Module import failed: {module_name}")
        info(f"  Error: {str(e)}")
        results['failed'].append(f"module:{module_name}")

# =============================================================================
# TEST 7: Load and Test Data
# =============================================================================
test_section("Data Loading")

try:
    import numpy as np
    
    # Load mask
    mask_path = base_path / 'data/processed/land_ocean_mask.npy'
    if mask_path.exists():
        mask = np.load(mask_path)
        ocean_cells = np.sum(mask == 1)
        land_cells = np.sum(mask == 0)
        success(f"Mask loaded: shape={mask.shape}")
        info(f"  Ocean cells: {ocean_cells:,}")
        info(f"  Land cells: {land_cells:,}")
        results['passed'].append("Mask loading")
    
    # Load training data sample
    train_path = base_path / 'data/processed/train_data.npz'
    if train_path.exists():
        data = np.load(train_path)
        success(f"Training data loaded: {list(data.keys())}")
        if 'inputs' in data and 'targets' in data:
            info(f"  Input shape: {data['inputs'].shape}")
            info(f"  Target shape: {data['targets'].shape}")
            info(f"  Training samples: {len(data['inputs'])}")
            results['passed'].append("Training data loading")
        data.close()
    
except Exception as e:
    error(f"Data loading failed: {e}")
    traceback.print_exc()
    results['failed'].append("Data loading")

# =============================================================================
# TEST 7b: Data Freshness & Validity
# =============================================================================
test_section("Data Freshness & Validity")

try:
    import json

    MIN_TRAIN_SAMPLES = 200  # Real dataset should have hundreds; synthetic had 193
    MIN_OCEAN_CELLS = 70000  # Real mask has ~83k; a bad mask might have ~18k

    # --- Check training sample count ---
    train_meta_path = base_path / 'data/processed/train_metadata.json'
    if train_meta_path.exists():
        with open(train_meta_path, 'r') as f:
            train_meta = json.load(f)

        n_samples = train_meta.get('n_samples', 0)
        is_synthetic = train_meta.get('synthetic', None)
        date_range = train_meta.get('date_range', 'unknown')
        source = train_meta.get('source', 'unknown')

        info(f"  Training samples: {n_samples}")
        info(f"  Synthetic flag: {is_synthetic}")
        info(f"  Source: {source}")
        info(f"  Date range: {date_range}")

        if n_samples >= MIN_TRAIN_SAMPLES:
            success(f"Training sample count adequate: {n_samples} >= {MIN_TRAIN_SAMPLES}")
            results['passed'].append("Data: sample count")
        else:
            error(f"Training sample count TOO LOW: {n_samples} < {MIN_TRAIN_SAMPLES}")
            error("  Data may be stale/synthetic. Re-run: venv/bin/python scripts/data/build_real_dataset.py")
            results['failed'].append("Data: sample count")

        if is_synthetic is True:
            error("Data is marked as SYNTHETIC (synthetic=True in metadata)")
            error("  Rebuild from real NSIDC data: venv/bin/python scripts/data/build_real_dataset.py")
            results['failed'].append("Data: synthetic flag")
        elif is_synthetic is False:
            success("Data is marked as REAL (synthetic=False)")
            results['passed'].append("Data: synthetic flag")
        else:
            warning("Data metadata has no 'synthetic' field — cannot verify provenance")
            results['warnings'].append("Data: synthetic flag missing")
    else:
        error("Training metadata file not found: data/processed/train_metadata.json")
        results['failed'].append("Data: metadata file")

    # --- Check land-ocean mask quality ---
    mask_path = base_path / 'data/processed/land_ocean_mask.npy'
    if mask_path.exists():
        mask = np.load(mask_path)
        ocean_cells = int(np.sum(mask == 1))
        if ocean_cells >= MIN_OCEAN_CELLS:
            success(f"Mask ocean cells adequate: {ocean_cells:,} >= {MIN_OCEAN_CELLS:,}")
            results['passed'].append("Data: mask quality")
        else:
            error(f"Mask ocean cells TOO LOW: {ocean_cells:,} < {MIN_OCEAN_CELLS:,}")
            error("  Mask may have been built incorrectly (e.g., using SIC>0 instead of isfinite)")
            results['failed'].append("Data: mask quality")

    # --- Check date range in metadata ---
    if train_meta_path.exists():
        date_pairs = train_meta.get('date_pairs', [])
        if date_pairs:
            first_date = date_pairs[0][0]
            last_date = date_pairs[-1][-1]
            info(f"  Training period: {first_date} → {last_date}")
            # Sanity: training data should span at least 6 months
            from datetime import datetime
            try:
                d1 = datetime.strptime(first_date, "%Y-%m-%d")
                d2 = datetime.strptime(last_date, "%Y-%m-%d")
                span_days = (d2 - d1).days
                if span_days >= 180:
                    success(f"Training period spans {span_days} days (>= 180)")
                    results['passed'].append("Data: date range span")
                else:
                    warning(f"Training period only {span_days} days (< 180)")
                    results['warnings'].append("Data: short date range")
            except ValueError:
                warning("Could not parse dates from metadata")
                results['warnings'].append("Data: date parse")

except Exception as e:
    error(f"Data freshness check failed: {e}")
    traceback.print_exc()
    results['failed'].append("Data freshness")

# =============================================================================
# TEST 8: Model Instantiation
# =============================================================================
test_section("Model Instantiation")

try:
    import torch
    from seaice_forecast.models.unet import UNet
    
    # Create model
    model = UNet(input_channels=7, output_channels=1)
    success(f"U-Net model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    success(f"Total parameters: {total_params:,}")
    info(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(2, 7, 332, 316).to(device)  # Batch=2, 7 days, Antarctic grid
    
    with torch.no_grad():
        output = model(dummy_input)
    
    success(f"Forward pass successful: {dummy_input.shape} -> {output.shape}")
    
    # Check output properties
    if output.shape == (2, 1, 332, 316):
        success("Output shape correct")
    else:
        error(f"Output shape incorrect: expected (2, 1, 332, 316), got {output.shape}")
    
    if torch.all((output >= 0) & (output <= 1)):
        success("Output values in [0, 1] range (sigmoid working)")
    else:
        warning("Some output values outside [0, 1] range")
    
    results['passed'].append("Model instantiation")
    results['passed'].append("Forward pass")
    
except Exception as e:
    error(f"Model test failed: {e}")
    traceback.print_exc()
    results['failed'].append("Model instantiation")

# =============================================================================
# TEST 9: Training Script Availability
# =============================================================================
test_section("Training Scripts")

try:
    import os
    base_path = Path(__file__).parent
    
    training_scripts = {
        'scripts/training/train_phase1.py': 'Phase 1 training',
        'scripts/training/train_phase2.py': 'Phase 2 training',
        'scripts/training/train_phase3_convlstm.py': 'Phase 3 ConvLSTM training',
        'scripts/training/train_phase4_multi_horizon.py': 'Phase 4 multi-horizon training',
    }
    
    for script_path, description in training_scripts.items():
        full_path = base_path / script_path
        if full_path.exists():
            success(f"{description} script exists")
            results['passed'].append(f"script:{description}")
        else:
            warning(f"{description} script missing: {script_path}")
            results['warnings'].append(f"script:{description}")
    
except Exception as e:
    error(f"Training scripts check failed: {e}")
    traceback.print_exc()
    results['failed'].append("Training scripts")

# =============================================================================
# TEST 10: Basic Workflow Test
# =============================================================================
test_section("Basic Workflow Test")

try:
    from seaice_forecast.data_processing.dataset_phase1 import SICDataset
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    
    # Try to load actual data if it exists
    base_path = Path(__file__).parent
    train_path = base_path / 'data/processed/train_data.npz'
    mask_path = base_path / 'data/processed/land_ocean_mask.npy'
    
    if train_path.exists() and mask_path.exists():
        mask = np.load(mask_path)
        
        # Create dataset
        dataset = SICDataset(data_file=str(train_path), mask=mask)
        success(f"Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        success(f"DataLoader created")
        
        # Test one batch
        batch_x, batch_y, batch_mask = next(iter(dataloader))
        success(f"Loaded batch: x={batch_x.shape}, y={batch_y.shape}, mask={batch_mask.shape}")
        
        # Test forward pass with model
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        from seaice_forecast.models.unet import UNet
        model = UNet(input_channels=7, output_channels=1).to(device)
        
        with torch.no_grad():
            batch_x = batch_x.to(device)
            pred = model(batch_x)
        
        success(f"Forward pass with real data successful: {pred.shape}")
        
        # Compute loss
        batch_y = batch_y.to(device)
        mask_np = mask  # Keep numpy version for indexing
        
        # Simple MAE loss over ocean
        error_map = torch.abs(pred - batch_y)
        # Convert to numpy for masking
        error_np = error_map.cpu().numpy()
        masked_error = error_np[:, :, mask_np == 1].mean()
        
        success(f"Loss computed: MAE = {masked_error:.6f}")
        
        results['passed'].append("End-to-end workflow")
    else:
        warning("Training data not available, skipping workflow test")
        results['warnings'].append("No data for workflow test")
    
except Exception as e:
    error(f"Workflow test failed: {e}")
    traceback.print_exc()
    results['failed'].append("End-to-end workflow")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{BLUE}{'='*70}{RESET}")
print(f"{BLUE}FINAL SUMMARY{RESET}")
print(f"{BLUE}{'='*70}{RESET}\n")

print(f"{GREEN}✓ Passed: {len(results['passed'])} tests{RESET}")
print(f"{RED}✗ Failed: {len(results['failed'])} tests{RESET}")
print(f"{YELLOW}⚠ Warnings: {len(results['warnings'])} items{RESET}")

if results['failed']:
    print(f"\n{RED}Failed tests:{RESET}")
    for item in results['failed']:
        print(f"  - {item}")

if results['warnings']:
    print(f"\n{YELLOW}Warnings:{RESET}")
    for item in results['warnings']:
        print(f"  - {item}")

# Overall verdict
print(f"\n{BLUE}{'='*70}{RESET}")
if len(results['failed']) == 0:
    print(f"{GREEN}✓ MODEL IS READY TO RUN!{RESET}")
    print(f"{GREEN}  You can proceed with training and evaluation.{RESET}")
    sys.exit(0)
elif len(results['failed']) <= 3:
    print(f"{YELLOW}⚠ MODEL IS MOSTLY READY{RESET}")
    print(f"{YELLOW}  Fix the failed items above before running.{RESET}")
    sys.exit(1)
else:
    print(f"{RED}✗ MODEL IS NOT READY{RESET}")
    print(f"{RED}  Multiple critical issues need to be resolved.{RESET}")
    sys.exit(1)
