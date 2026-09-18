#!/usr/bin/env bash
# Verify every claim this repository makes, from a clone.
#
#     ./verify.sh
#
# Runs the test suites, loads each exported model artifact and calls predict()
# on it, checks that every figure quoted in the documentation matches its source
# file, and prints the held-out evaluation.
#
#     ./verify.sh                 check documented figures against stored results
#     ./verify.sh --recompute     additionally re-derive the evaluation from the
#                                 checkpoint and compare it to the stored result
#                                 (slower, and the only check that proves the
#                                 committed numbers are what the model produces)
#
# Exit code is non-zero if anything fails.

set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

PY="${PY:-$ROOT/seaice_forecast/venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "!! No model environment at $PY"
  echo "   The models need Python 3.9 with torch, joblib, scikit-learn, pandas."
  echo "   Set PY=/path/to/python and re-run."
  exit 1
fi

RECOMPUTE=0
[ "${1:-}" = "--recompute" ] && RECOMPUTE=1

fail=0
hr() { printf '\n%s\n' "------------------------------------------------------------"; }
step() { printf '\n== %s\n' "$1"; }

step "Environment"
"$PY" - <<'PYEOF'
import sys
print(f"  python      {sys.version.split()[0]}")
for m in ("numpy", "torch", "sklearn", "joblib", "pandas"):
    try:
        mod = __import__(m)
        print(f"  {m:11s} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {m:11s} MISSING")
PYEOF

step "Test suite: seaice_forecast"
(cd seaice_forecast && "$PY" -m pytest tests/ -q --no-header 2>&1 | tail -1) || fail=1

step "Test suite: iceberg-drift"
(cd iceberg-drift && "$PY" -m pytest tests/ -q --no-header 2>&1 | tail -1) || fail=1

step "End-to-end integration: telemetry -> shore -> backend -> models -> route"
"$PY" -m pytest integration/ -q --no-header 2>&1 | tail -1 || fail=1

step "Exported model artifacts: load from disk and run"
"$PY" - <<'PYEOF' || exit 1
import hashlib, json, sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "shipNavigation/backend"))   # class definitions
sys.path.insert(0, str(root / "seaice_forecast/src"))      # sea-ice architecture

import joblib
import numpy as np

manifest = json.loads((root / "model_exports/manifest.json").read_text())
bad = 0

for entry in manifest["artifacts"]:
    path = root / "model_exports" / entry["file"]
    if not path.exists():
        print(f"  {entry['file']:24s} MISSING"); bad += 1; continue

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ok_hash = digest == entry.get("sha256")

    try:
        obj = joblib.load(path)
        key = entry["key"]
        if key == "polaris_risk":
            out = f"risk={float(obj.predict(0.70, 'PC4')):.4f} at SIC 0.70, PC4"
        elif key == "fuel_consumption":
            r = obj.predict(0.70, "PC4", 25.0)
            out = f"{r['fuel_tonnes']:.3f} t, {r['speed_knots']:.3f} kn over 25 km"
        elif key == "iceberg_drift":
            out = f"{len(obj.features)} features, 2-stage gate + u/v regressors"
        else:
            out = f"{obj.arch['in_channels']}ch x {obj.arch['seq_len']}d -> 332x316"
        print(f"  {entry['file']:24s} loads  hash={'ok' if ok_hash else 'CHANGED'}  {out}")
        if not ok_hash:
            bad += 1
    except Exception as exc:
        print(f"  {entry['file']:24s} FAILED {type(exc).__name__}: {exc}")
        bad += 1

sys.exit(1 if bad else 0)
PYEOF
[ $? -ne 0 ] && fail=1

step "Documented figures vs their source files"
"$PY" scripts/check_claims.py || fail=1

if [ "$RECOMPUTE" -eq 1 ]; then
  step "Re-deriving the evaluation from the checkpoint (a few minutes)"
  (cd seaice_forecast && "$PY" scripts/evaluation/rollout_comparison.py \
      --horizons 1 --max-samples 120 --stride 5 --out /tmp/recomputed.json 2>&1 \
      | grep -E "^\+") || fail=1
  "$PY" scripts/compare_recomputed.py /tmp/recomputed.json || fail=1
fi

step "Held-out evaluation (read from the files the training runs wrote)"
"$PY" - <<'PYEOF'
import json
from pathlib import Path

sea = Path("seaice_forecast/output/evaluation/baseline_comparison.json")
if sea.exists():
    print("  Sea ice - MAE over ice-zone cells, lower is better:")
    print(f"    {'lead':>6} {'model':>9} {'persist':>9} {'clim':>9}   verdict")
    for h, b in json.loads(sea.read_text()).items():
        m, p, c = b.get("model"), b.get("persistence"), b.get("climatology")
        if not (m and p and c):
            continue
        v = "beats persistence" if m["mae_ice"] < p["mae_ice"] else "BEATEN by persistence"
        print(f"    {h:>6} {m['mae_ice']:9.4f} {p['mae_ice']:9.4f} {c['mae_ice']:9.4f}   {v}")
    print("    -> beats climatology everywhere; persistence is stronger at every")
    print("       lead tested. See seaice_forecast/OPEN_PROBLEM.md.")

drift = Path("iceberg-drift/output/drift_twostage_metrics.json")
if drift.exists():
    print("\n  Iceberg drift - 24 h position error against held-out NIC fixes:")
    print(f"    {'variant':>18} {'RMS km':>8} {'median':>8} {'<10km':>7} {'skill':>7}")
    for name, m in json.loads(drift.read_text()).items():
        print(f"    {name:>18} {m['pos_err_24h_km_rms']:8.3f} "
              f"{m['pos_err_24h_km_median']:8.3f} {m['within_10km_pct']:6.1f}% "
              f"{m['skill_vs_constant']:7.3f}")
    print("    -> two_stage is the shipped model.")
PYEOF

hr
if [ "$fail" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED (see above)"
fi
exit "$fail"
