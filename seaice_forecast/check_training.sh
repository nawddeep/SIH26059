#!/bin/bash
# Quick training status. Run from seaice_forecast/:   ./check_training.sh
cd "$(dirname "$0")"
LOG=logs/train_run9.log
CKPT=models/checkpoints/phase3

echo "=============================================="
if pgrep -f "train_phase3_convlstm" > /dev/null; then
    PID=$(pgrep -f train_phase3_convlstm | head -1)
    echo " STATUS:  ✅ TRAINING RUNNING"
    echo " pid $PID   elapsed $(ps -p $PID -o etime= | tr -d ' ')"
else
    echo " STATUS:  ⛔ NOT RUNNING (finished, or stopped)"
fi
echo "=============================================="

DONE=$(grep -cE "Epoch +[0-9]+/" "$LOG" 2>/dev/null || echo 0)
echo " epochs completed: $DONE / 15"

echo ""
echo " --- last 3 epoch results ---"
grep -E "Epoch +[0-9]+/|val_loss|Best" "$LOG" 2>/dev/null | tail -3 || echo " (none yet - epoch 1 takes ~6 min)"

echo ""
echo " --- checkpoints ---"
if ls "$CKPT"/*.pt >/dev/null 2>&1; then
    ls -lh "$CKPT"/*.pt | awk '{print "  ",$9,"  ",$5,"  ",$6,$7,$8}'
else
    echo "   none yet"
fi

echo ""
if grep -qiE "Traceback|Error|Killed" "$LOG" 2>/dev/null; then
    echo " ⚠️  ERRORS FOUND:"
    grep -iE "Traceback|Error|Killed" "$LOG" | tail -3
else
    echo " no errors in log ✅"
fi

echo ""
echo " TRAINING IS OVER when: process NOT RUNNING *and* epochs = 15"
echo "=============================================="
