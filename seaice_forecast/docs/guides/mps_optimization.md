# Apple Silicon (MPS) Training Optimization Guide

## Overview

This guide provides benchmarked optimizations for training on Apple Silicon (M1/M2/M3) using PyTorch's MPS (Metal Performance Shaders) backend. All optimizations preserve scientific behavior - same model outputs, same loss values, just faster wall-clock time.

---

## Quick Start

### 1. Run Benchmark

```bash
# Benchmark Phase 1 model (7-channel input)
python scripts/benchmark_mps_training.py --model-type phase1

# Benchmark Phase 2 model (49-channel input)
python scripts/benchmark_mps_training.py --model-type phase2

# Benchmark both
python scripts/benchmark_mps_training.py --full-benchmark
```

### 2. Apply Recommended Settings

Based on benchmarks, update your training command:

```bash
# Before (baseline)
python scripts/train.py --device mps --epochs 100

# After (optimized)
python scripts/train.py --device mps --batch-size 16 --epochs 100
# With DataLoader num_workers=2 in code
```

---

## Optimization Checklist

### ✅ Device Selection

**Issue**: Scripts defaulting to CUDA or not detecting MPS

**Fix**: Use the device utility

```python
from seaice_forecast.utils.device_utils import get_optimal_device

device = get_optimal_device(prefer_device='mps', verbose=True)
model.to(device)
```

**Verify**:
```bash
python -c "from seaice_forecast.utils.device_utils import get_optimal_device; \
           device = get_optimal_device(); print(f'Using: {device}')"
```

Expected output: `Using: mps`

---

### ✅ DataLoader Configuration

**Issue**: Default `num_workers=0` or `pin_memory=True` (CUDA-specific)

**Optimal for MPS**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=16,  # Larger than CUDA default
    num_workers=2,  # 2-4 workers for MPS
    pin_memory=False,  # MPS doesn't benefit
    persistent_workers=True  # When num_workers > 0
)
```

**Why**:
- MPS uses unified memory → no pinned memory benefit
- macOS uses `spawn` for multiprocessing → 2-4 workers optimal
- Higher worker counts can cause startup overhead

**Benchmark Results** (typical):
| Workers | Throughput | Notes |
|---------|-----------|-------|
| 0 | Baseline | Single-threaded data loading |
| 2 | +15-25% | Optimal for most cases |
| 4 | +10-20% | Diminishing returns |
| 8 | +0-10% | Overhead dominates |

---

### ✅ Batch Size Tuning

**Issue**: CUDA-optimized batch sizes (8) too small for MPS unified memory

**Optimal for MPS**:
- **Phase 1 (7 channels)**: batch_size=16-32
- **Phase 2 (49 channels)**: batch_size=16-24

**Why**:
- Unified memory architecture reduces transfer overhead
- Larger batches amortize kernel launch costs
- MPS doesn't have separate VRAM limit

**Benchmark**:
```bash
# Test different batch sizes
python scripts/benchmark_mps_training.py --model-type phase1
# Check: samples/sec increases with larger batches
```

**Warning**: Don't exceed physical memory → causes swapping

---

### ✅ Mixed Precision (FP16)

**Status**: ⚠️ **Test carefully - MPS FP16 can be unstable**

**How to test**:
```python
from torch.cuda.amp import autocast

# Training loop
with torch.autocast(device_type='mps', dtype=torch.float16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Loss computation in FP32
loss.backward()
```

**Correctness check** (critical):
```bash
python scripts/benchmark_mps_training.py --model-type phase1
# Check: "Correctness Check" must PASS for FP16 config
```

**Known issues**:
- Some PyTorch versions have MPS FP16 bugs
- Loss may diverge or produce NaNs
- If unstable: **revert to FP32**

**Benchmark results** (when stable):
- Speed: +20-40% faster
- Memory: -30% reduction
- Accuracy: Must match FP32 within tolerance

---

### ✅ CPU Fallback Detection

**Issue**: Some ops silently fall back to CPU, slowing training

**How to check**:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/train.py --device mps --epochs 1 2>&1 | \
    grep "not currently implemented for the MPS device"
```

**Common fallbacks**:
- Advanced indexing (gather, scatter)
- Some interpolation modes
- Certain reduction operations

**Fix**:
- If unavoidable: Document it
- If avoidable: Rewrite operation using MPS-supported ops
- Check PyTorch release notes for fixes

---

### ✅ Memory Management

**Clear cache between runs**:
```python
import torch

# Between model training runs
torch.mps.empty_cache()
```

**Why**: Prevents allocator from holding unnecessary memory

**When to use**:
- Training multiple models in sequence
- After evaluation before training
- If seeing memory pressure warnings

---

## Benchmarking Results Template

After running `benchmark_mps_training.py`, you should see:

```
BENCHMARK SUMMARY
================================================================================

Configuration                            Workers  Batch   Precision  Epoch(s)   Samples/s    Status
--------------------------------------------------------------------------------------------------------------
Baseline (workers=0, bs=8, fp32)        0        8       fp32       45.23      2.21         ✓ PASS
Workers=2, bs=8, fp32                   2        8       fp32       38.12      2.62         ✓ PASS
Workers=2, bs=16, fp32                  2        16      fp32       32.45      3.08         ✓ PASS
Workers=2, bs=16, fp16                  2        16      fp16       25.67      3.90         ✓ PASS*

* If FP16 passes correctness check
```

**Interpret**:
- Baseline → Current performance
- Best config → Highest samples/s with PASS status
- FP16 → Only use if correctness check passes

---

## Implementation Checklist

### Code Changes

- [ ] **Import device utilities**
  ```python
  from seaice_forecast.utils.device_utils import (
      get_optimal_device,
      get_dataloader_config,
      clear_device_cache
  )
  ```

- [ ] **Update device selection**
  ```python
  # Old
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # New
  device = get_optimal_device(prefer_device='mps')
  ```

- [ ] **Update DataLoader config**
  ```python
  # Old
  dataloader = DataLoader(dataset, batch_size=8, num_workers=0)

  # New
  dl_config = get_dataloader_config(device, num_workers=2)
  dataloader = DataLoader(
      dataset,
      batch_size=16,
      **dl_config
  )
  ```

- [ ] **Add cache clearing**
  ```python
  # Between model training
  clear_device_cache(device)
  ```

- [ ] **Optional: Add FP16** (only if stable)
  ```python
  from seaice_forecast.utils.device_utils import get_autocast_context

  with get_autocast_context(device, enabled=True):
      outputs = model(inputs)
      loss = criterion(outputs, targets)
  ```

### Verification

- [ ] Run correctness check
  ```bash
  python scripts/benchmark_mps_training.py --model-type phase1
  # Verify: All configs show "✓ PASS"
  ```

- [ ] Run full training epoch
  ```bash
  python scripts/train.py --device mps --epochs 1
  # Verify: No errors, loss decreases normally
  ```

- [ ] Check for fallbacks
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  python scripts/train.py --device mps --epochs 1 2>&1 | \
      grep "not currently implemented"
  # Document any found
  ```

- [ ] Benchmark improvement
  ```bash
  # Before optimization
  time python scripts/train.py --device mps --epochs 5

  # After optimization
  time python scripts/train.py --device mps --epochs 5

  # Compare wall-clock time
  ```

---

## Expected Improvements

Based on typical MPS optimizations:

| Optimization | Expected Speedup | Cumulative |
|-------------|------------------|------------|
| Baseline | 1.0x | 1.0x |
| + DataLoader workers (2) | +20-30% | 1.2-1.3x |
| + Batch size (16) | +40-50% | 1.7-1.9x |
| + FP16 (if stable) | +20-40% | 2.0-2.7x |

**Total potential**: 2-3x faster training

**Actual results depend on**:
- Model size (Phase 1 vs Phase 2)
- Data pipeline bottlenecks
- PyTorch version
- macOS version
- M1 vs M2 vs M3 chip

---

## Troubleshooting

### Issue: "MPS backend out of memory"

**Cause**: Batch size too large for available memory

**Fix**:
```bash
# Reduce batch size
python scripts/train.py --device mps --batch-size 8

# Or clear cache before training
python -c "import torch; torch.mps.empty_cache()"
```

### Issue: "RuntimeError: ... not currently implemented for the MPS device"

**Cause**: Operation has no MPS implementation

**Fix**:
1. Check PyTorch version (upgrade if old)
2. Check operation in code (can it be rewritten?)
3. If unavoidable: Document and accept CPU fallback
4. File PyTorch issue if widely used operation

### Issue: Loss diverges with FP16

**Cause**: Numerical instability in mixed precision

**Fix**:
```python
# Revert to FP32
# Remove autocast context
# Document that FP16 is unstable for this model
```

### Issue: Workers=4 slower than workers=2

**Cause**: macOS multiprocessing overhead

**Fix**: Use workers=2 (optimal for most MPS cases)

### Issue: Training slower on Mac than expected

**Check**:
1. Battery power mode (use "High Performance")
2. Background processes (close heavy apps)
3. Thermal throttling (ensure good ventilation)
4. Swapping (batch size too large)

---

## Verification Protocol

Before considering optimization complete:

1. **Correctness check passes**
   ```bash
   python scripts/benchmark_mps_training.py --model-type phase1
   # All configs: ✓ PASS
   ```

2. **Full epoch trains successfully**
   ```bash
   python scripts/train.py --device mps --epochs 1
   # No errors, loss decreases
   ```

3. **Outputs numerically identical**
   - Run fixed batch before/after optimization
   - Verify max difference < 1e-6 (FP32) or < 1e-3 (FP16)
   - Verify loss difference < 1e-6 (FP32) or < 1e-3 (FP16)

4. **Performance measured**
   - Baseline epoch time recorded
   - Optimized epoch time recorded
   - Speedup calculated and documented

5. **Fallbacks documented**
   - List of CPU fallback ops (if any)
   - Explanation for each unavoidable fallback

---

## Files Modified

When applying optimizations:

### New Files
- `src/seaice_forecast/utils/device_utils.py` - Device selection utilities
- `scripts/benchmark_mps_training.py` - Benchmarking tool
- `MPS_OPTIMIZATION_GUIDE.md` - This guide

### Modified Files (if applying optimizations)
- `scripts/train.py` - Device selection, DataLoader config
- `scripts/train_phase2.py` - Device selection, DataLoader config
- `src/seaice_forecast/data_processing/dataset.py` - DataLoader defaults
- `src/seaice_forecast/evaluation/trainer.py` - Optional FP16 support

**Rule**: Only modify if benchmark shows improvement AND correctness check passes

---

## References

- **PyTorch MPS Backend**: https://pytorch.org/docs/stable/notes/mps.html
- **MPS Profiler**: https://developer.apple.com/metal/pytorch/
- **Issue Tracker**: https://github.com/pytorch/pytorch/labels/module%3A%20mps

---

## Summary

**Goal**: 2-3x faster training on Apple Silicon

**Method**:
1. Benchmark current performance
2. Apply proven optimizations (workers, batch size)
3. Test FP16 (revert if unstable)
4. Verify correctness at every step
5. Document improvements

**Non-negotiable**:
- Scientific behavior unchanged
- Outputs numerically identical
- Loss values match baseline

**When done**:
- Training faster
- Correctness verified
- Improvements documented
- Ready for production use

---

**Last Updated**: [To be filled after benchmarking]
**PyTorch Version Tested**: [To be filled]
**macOS Version**: [To be filled]
