#!/bin/bash
# Monitor training progress

LOG_FILE=$(ls -t logs/training_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No training log found!"
    exit 1
fi

echo "📊 Monitoring: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if training is running
if ps aux | grep "train_phase1" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    RUNNING=true
else
    echo "⏸️  Training has STOPPED"
    RUNNING=false
fi

echo ""
echo "📈 Latest Progress:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -15 "$LOG_FILE" | grep "Epoch"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Show best model info
if grep -q "New best model" "$LOG_FILE"; then
    BEST_LOSS=$(grep "New best model" "$LOG_FILE" | tail -1 | sed -n 's/.*val_loss: \([0-9.]*\).*/\1/p')
    BEST_EPOCH=$(grep "New best model" "$LOG_FILE" | tail -1 | sed -n 's/.*Epoch *\([0-9]*\).*/\1/p')
    echo "🏆 Best Model:"
    echo "   Epoch: $BEST_EPOCH"
    echo "   Val Loss: $BEST_LOSS"
    echo ""
fi

# Show checkpoint info
if [ -d "models/checkpoints/phase1" ]; then
    echo "💾 Saved Checkpoints:"
    ls -lht models/checkpoints/phase1/*.pt 2>/dev/null | head -3 | awk '{print "   " $9, "(" $5 ")"}'
    echo ""
fi

if [ "$RUNNING" = true ]; then
    echo "💡 To watch live: tail -f $LOG_FILE"
    echo "💡 To stop: killall -9 python"
else
    echo "💡 Training complete or stopped!"
    echo "💡 Check models/checkpoints/phase1/ for saved models"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
