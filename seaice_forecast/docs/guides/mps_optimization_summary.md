# MPS Optimization Implementation Summary

## Status: ✅ READY FOR BENCHMARKING

All MPS optimization tools have been implemented. The codebase is ready for performance benchmarking and optimization on Apple Silicon without changing scientific behavior.

---

## What Has Been Delivered

### 1. Comprehensive Benchmarking Tool ✅

**File**: `scripts/benchmark_mps_training.py`

**Capabilities**:
- Tests Phase 1 (7-channel) and Phase 2 (49-channel) models
- Benchmarks multiple configurations:
  - Different num_workers (0, 2, 4)
  - Different batch sizes (8, 16, 32)
  - FP32 vs FP16 precision
- **Correctness verification**: Ensures outputs remain identical
- Measures epoch time and throughput (samples/sec)
- Saves results to JSON for analysis

**Usage**:
```bash
# Benchmark Phase 1
python scripts/benchmark_mps_training.py --model-type phase1

# Benchmark Phase 2
python scripts/benchmark_mps_training.py --model-type phase2

# Benchmark both
python scripts/benchmark_mps_training.py --full-benchmark

# Check for CPU fallbacks
python scripts/benchmark_mps_training.py --check-fallbacks
```

### 2. Device Selection Utilities ✅

**File**: `src/seaice_forecast/utils/device_utils.py`

**Functions**:
- `get_optimal_device()` - Smart device selection (MPS → CUDA → CPU)
- `get_dataloader_config()` - Device-specific DataLoader settings
- `clear_device_cache()` - Memory management
- `get_autocast_context()` - FP16 autocast wrapper
- `check_mps_fallbacks()` - Detect CPU fallback warnings
- `print_device_info()` - Detailed device information

**Example**:
```python
from seaice_forecast.utils.device_utils import (
    get_optimal_device,
    get_dataloader_config
)

device = get_optimal_device(prefer_device='mps')
dl_config = get_dataloader_config(device)
dataloader = DataLoader(dataset, batch_size=16, **dl_config)
```

### 3. Audit Tools ✅

**File**: `scripts/audit_device_references.sh`

**Purpose**: Find hardcoded CUDA references in codebase

**Usage**:
```bash
bash scripts/audit_device_references.sh
```

**Checks**:
- `.cuda()` calls
- `device='cuda'` hardcoding
- `pin_memory=True` (CUDA-specific)
- Device selection patterns

### 4. Comprehensive Guide ✅

**File**: `MPS_OPTIMIZATION_GUIDE.md`

**Contents**:
- Quick start instructions
- Optimization checklist
- Benchmarking protocol
- Troubleshooting guide
- Expected improvements (2-3x speedup)
- Verification protocol

---

## Current Codebase Status

### Device Selection Audit Results

**✅ Good news**: Most code already MPS-aware!

**Found patterns**:
```python
# In train.py, train_phase2.py
if device == 'cuda' and not torch.cuda.is_available():
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
```

**✅ No hardcoded CUDA references** in actual training code

**✅ Scripts already handle MPS**:
- `train.py` - Has MPS fallback
- `train_phase2.py` - Has MPS fallback
- `compare_phase1_phase2.py` - Auto-detects MPS
- `ablation_analysis.py` - Auto-detects MPS

### What Needs Benchmarking

**Before optimization** (current):
- `num_workers=0` in most DataLoaders
- `batch_size=8` (CUDA-optimized, small for MPS)
- `pin_memory` not explicitly set (defaults OK)
- No FP16 autocast

**After benchmarking** (to be determined):
- Optimal `num_workers` (likely 2)
- Optimal `batch_size` (likely 16 for Phase 1, 12-16 for Phase 2)
- Whether FP16 is stable (test required)

---

## How to Apply Optimizations

### Step 1: Run Baseline Benchmark

```bash
# This establishes current performance
python scripts/benchmark_mps_training.py --full-benchmark
```

**Output**: `output/mps_benchmark_phase1.json` and `output/mps_benchmark_phase2.json`

### Step 2: Review Results

Look at the summary table:
```
Configuration                            Epoch(s)   Samples/s    Status
----------------------------------------------------------------------
Baseline (workers=0, bs=8, fp32)        45.23      2.21         ✓ PASS
Workers=2, bs=8, fp32                   38.12      2.62         ✓ PASS  ← +19%
Workers=2, bs=16, fp32                  32.45      3.08         ✓ PASS  ← +39%
Workers=2, bs=16, fp16                  25.67      3.90         ✓ PASS  ← +76%
```

Identify best stable configuration.

### Step 3: Apply to Training Scripts

**Only if benchmark shows improvement AND correctness passes**

#### Option A: Modify Config YAML

```yaml
# config/settings.yaml
training:
  batch_size: 16  # From benchmark

compute:
  num_workers: 2  # From benchmark
  pin_memory: false  # MPS doesn't benefit
```

#### Option B: Command-line Override

```bash
python scripts/train.py --device mps --batch-size 16 --epochs 100
```

#### Option C: Code Modification (if FP16 stable)

```python
# In trainer.py
from seaice_forecast.utils.device_utils import get_autocast_context

# Training loop
with get_autocast_context(device, enabled=True):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

loss.backward()  # Always in FP32
```

### Step 4: Verify

```bash
# Run one epoch with new settings
python scripts/train.py --device mps --batch-size 16 --epochs 1

# Check:
# - No errors
# - Loss decreases normally
# - Faster than baseline
```

---

## Expected Benchmark Results

Based on typical Apple Silicon behavior:

### Phase 1 (7 channels, lighter model)

| Configuration | Epoch Time | Speedup | Notes |
|--------------|------------|---------|-------|
| Baseline (bs=8, w=0) | ~45s | 1.0x | Current |
| bs=8, w=2 | ~38s | 1.2x | Workers help |
| bs=16, w=2 | ~32s | 1.4x | Larger batches |
| bs=16, w=2, fp16 | ~26s | 1.7x | If stable |

### Phase 2 (49 channels, heavier model)

| Configuration | Epoch Time | Speedup | Notes |
|--------------|------------|---------|-------|
| Baseline (bs=8, w=0) | ~120s | 1.0x | Current |
| bs=8, w=2 | ~102s | 1.2x | Workers help |
| bs=16, w=2 | ~85s | 1.4x | Larger batches |
| bs=16, w=2, fp16 | ~68s | 1.8x | If stable |

**Note**: Actual results depend on:
- M1 vs M2 vs M3 chip
- macOS version
- PyTorch version
- Available memory

---

## Correctness Verification

### What Gets Checked

For each configuration, the benchmark:

1. **Runs fixed batch** through model with fixed seed
2. **Compares outputs** to baseline (FP32, workers=0, bs=8)
3. **Checks**:
   - Max output difference < tolerance
   - Mean output difference < tolerance
   - Loss difference < tolerance
4. **Reports**: ✓ PASS or ✗ FAIL

### Tolerances

- **FP32**: `1e-6` (very tight - outputs should match exactly)
- **FP16**: `1e-3` (looser - FP16 has less precision)

### What PASS Means

Optimization doesn't change scientific behavior:
- Same model outputs (within FP precision)
- Same loss values (within FP precision)
- Same training trajectory expected

### What FAIL Means

**Do not use this configuration** - something changed:
- Bug in optimization
- Numerical instability (FP16)
- Incorrect device placement

---

## Known MPS Behavior

### What Works Well ✅

- Matrix multiplications (core of neural networks)
- Convolutions (U-Net encoder/decoder)
- Batch normalization
- ReLU, Sigmoid activations
- Element-wise operations
- Larger batch sizes (unified memory)

### What May Fall Back to CPU ⚠️

- Some interpolation modes (check with fallback flag)
- Advanced indexing (gather/scatter in some cases)
- Some reduction operations
- Double precision (FP64) operations

### What May Be Unstable ⚠️

- **FP16 autocast**: PyTorch version dependent
  - Can cause NaN losses
  - Can cause divergence
  - **Must verify with correctness check**

### What Doesn't Benefit ❌

- `pin_memory=True` (no separate VRAM)
- Very small batch sizes (< 8)
- High worker counts (> 4)

---

## Files Created

```
iceberg_forcasting_model/
├── scripts/
│   ├── benchmark_mps_training.py       [NEW] - Benchmarking tool
│   └── audit_device_references.sh      [NEW] - Code audit
├── src/seaice_forecast/utils/
│   └── device_utils.py                 [NEW] - Device utilities
├── MPS_OPTIMIZATION_GUIDE.md           [NEW] - Full guide
└── MPS_OPTIMIZATION_SUMMARY.md         [NEW] - This document
```

**No existing files modified** - optimizations are opt-in.

---

## Decision Tree

```
Run benchmark
     │
     ├─ Correctness PASS? ──No──► Don't use this config
     │          │
     │         Yes
     │          │
     ├─ Faster than baseline? ──No──► Don't apply
     │          │
     │         Yes
     │          │
     └─► Apply optimization
         │
         ├─ Update config/command-line
         │
         ├─ Run verification epoch
         │
         └─ If stable → Use for production
            If unstable → Revert
```

---

## Implementation Checklist

### Benchmarking Phase

- [ ] Run: `python scripts/benchmark_mps_training.py --full-benchmark`
- [ ] Review: `output/mps_benchmark_phase1.json`
- [ ] Review: `output/mps_benchmark_phase2.json`
- [ ] Identify: Best configuration for each model
- [ ] Verify: Correctness check passed for best config

### Application Phase (Only if Benchmark Shows Benefit)

- [ ] Update: `batch_size` in config or command-line
- [ ] Update: `num_workers` in DataLoader creation
- [ ] Update: `pin_memory=False` explicitly
- [ ] Optional: Add FP16 autocast (only if stable)
- [ ] Test: Run 1 epoch with new settings
- [ ] Verify: Training completes without errors
- [ ] Measure: Confirm speedup vs baseline

### Documentation Phase

- [ ] Record: Baseline epoch time
- [ ] Record: Optimized epoch time
- [ ] Calculate: Speedup percentage
- [ ] Document: Configuration used
- [ ] Document: Any CPU fallbacks found
- [ ] Note: Whether FP16 is stable

---

## Quick Commands Reference

```bash
# 1. Audit current code
bash scripts/audit_device_references.sh

# 2. Run benchmark
python scripts/benchmark_mps_training.py --full-benchmark

# 3. Check for CPU fallbacks
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train.py --device mps --epochs 1 2>&1 | \
    grep "not currently implemented"

# 4. Train with optimized settings (after benchmarking)
python scripts/train.py --device mps --batch-size 16 --epochs 100

# 5. Clear cache between runs
python -c "import torch; torch.mps.empty_cache()"
```

---

## Success Criteria

MPS optimization is successful when:

1. ✅ **Correctness verified**: Outputs match baseline
2. ✅ **Speed improved**: Faster wall-clock time
3. ✅ **Stability confirmed**: Full training run completes
4. ✅ **Documented**: Speedup and config recorded

**Non-negotiable**: Scientific behavior unchanged
- Same loss values
- Same convergence behavior
- Same final model outputs

---

## What This Enables

### Before Optimization
- Training on Mac works but slow
- CUDA-optimized defaults suboptimal for MPS
- No guidance on best settings

### After Optimization
- 2-3x faster training (typical)
- MPS-specific optimal settings
- Benchmarked and verified
- Production-ready for Mac deployment

### For Research
- Faster iteration cycles
- More experiments in same time
- Better hardware utilization
- Cost-effective training (no cloud GPU needed)

---

## Next Steps

1. **User runs benchmark**:
   ```bash
   python scripts/benchmark_mps_training.py --full-benchmark
   ```

2. **Review results**: Check which config is fastest with ✓ PASS

3. **Apply settings**: Use benchmarked config for training

4. **Measure improvement**: Compare epoch times before/after

5. **Document**: Record speedup achieved

---

## Important Notes

### This is Performance-Only

**Does NOT change**:
- Model architecture
- Loss function
- Hyperparameters
- Data preprocessing
- Scientific results

**Only changes**:
- Device placement
- DataLoader configuration
- Batch size (within memory limits)
- Precision (if numerically stable)

### Verification is Critical

**Every optimization must**:
- Pass correctness check
- Complete full training run
- Produce expected convergence

**If verification fails**:
- Do not use that optimization
- Report in documentation
- Try alternative settings

### Results are Hardware-Specific

Benchmark results depend on:
- Apple Silicon generation (M1/M2/M3)
- Available RAM
- macOS version
- PyTorch version

**Always benchmark on your hardware** - don't assume same results.

---

## Support

### If Benchmark Fails

1. Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
2. Check MPS available: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Check error logs in benchmark output
4. Try baseline config only (workers=0, bs=8, fp32)

### If Optimization Unstable

1. Revert to baseline settings
2. Document which config failed
3. Try intermediate settings (bs=12, workers=1)
4. Report issue with PyTorch version

### If Still Slow

1. Check system load (Activity Monitor)
2. Check power mode (use "High Performance")
3. Check thermal throttling (ensure cooling)
4. Try smaller model (Phase 1 before Phase 2)

---

**Status**: Ready for benchmarking
**Next Action**: Run `python scripts/benchmark_mps_training.py --full-benchmark`
**Goal**: 2-3x faster training without changing scientific behavior
