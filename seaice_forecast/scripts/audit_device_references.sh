#!/bin/bash
# Audit codebase for device-specific references

echo "============================================================"
echo "DEVICE REFERENCE AUDIT"
echo "============================================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Searching in: $PROJECT_ROOT"
echo ""

echo "1. CUDA-specific references (.cuda(), device='cuda', etc.)"
echo "------------------------------------------------------------"
grep -rn "\.cuda()" "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  ✓ None found"
grep -rn "device='cuda'" "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  ✓ None found"
grep -rn 'device="cuda"' "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  ✓ None found"
grep -rn "\.to('cuda')" "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  ✓ None found"
grep -rn '\.to("cuda")' "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  ✓ None found"

echo ""
echo "2. Device selection patterns"
echo "------------------------------------------------------------"
grep -rn "torch.cuda.is_available()" "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null | head -10

echo ""
echo "3. MPS-aware device selection"
echo "------------------------------------------------------------"
grep -rn "torch.backends.mps.is_available()" "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null | head -10

echo ""
echo "4. DataLoader pin_memory settings"
echo "------------------------------------------------------------"
grep -rn "pin_memory.*True" "$PROJECT_ROOT/src/" "$PROJECT_ROOT/scripts/" 2>/dev/null || echo "  ✓ None found"

echo ""
echo "5. Recommended device selection pattern"
echo "------------------------------------------------------------"
echo "  from seaice_forecast.utils.device_utils import get_optimal_device"
echo "  device = get_optimal_device(prefer_device='mps')"
echo ""

echo "============================================================"
echo "AUDIT COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review any CUDA-specific code found above"
echo "  2. Replace with device-agnostic patterns"
echo "  3. Run: python scripts/benchmark_mps_training.py"
