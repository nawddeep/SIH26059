#!/usr/bin/env bash
# Launches the gateway: ingest and sender together, shutting both down cleanly.
#
# The two run as separate processes on purpose. Ingestion has to keep buffering
# while the sender is stuck retrying a dead satellite link, and that guarantee
# only holds if neither can block the other.
#
# Usage:  ./run.sh           start both (tmux panes, or background + logs)
#         ./run.sh stop      stop both
#         ./run.sh status    show what is running and the queue depth
#         ./run.sh reset     stop, then wipe the queue for a clean demo run

set -uo pipefail
cd "$(dirname "$0")"

SESSION="gateway"
PYTHON="${PYTHON:-python3}"
PROCS=("ingest:ingest.py" "sender:sender.py")

stop_all() {
  echo "[run] stopping gateway..."
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION" && echo "[run]   tmux session '$SESSION' killed"
  fi
  for entry in "${PROCS[@]}"; do
    script="${entry#*:}"
    # Anchored so the pattern cannot match the shell running this script -
    # an unanchored 'pkill -f ingest.py' kills its own parent.
    if pkill -f "^${PYTHON} ${script}$" 2>/dev/null; then
      echo "[run]   stopped $script"
    fi
  done
  sleep 1
  echo "[run] done."
}

show_status() {
  echo "[run] gateway processes:"
  for entry in "${PROCS[@]}"; do
    script="${entry#*:}"
    if pgrep -f "^${PYTHON} ${script}$" >/dev/null 2>&1; then
      printf "        %-12s RUNNING (pid %s)\n" "$script" "$(pgrep -f "^${PYTHON} ${script}$" | tr '\n' ' ')"
    else
      printf "        %-12s stopped\n" "$script"
    fi
  done
  echo "[run] queue:"
  "$PYTHON" buffer.py 2>/dev/null | sed -n '1,2p' | sed 's/^/        /'
}

case "${1:-start}" in
  stop)   stop_all; exit 0 ;;
  status) show_status; exit 0 ;;
  reset)
    stop_all
    rm -f buffer.db buffer.db-wal buffer.db-shm ingest.log sender.log
    echo "[run] queue and logs wiped - next start begins from empty."
    exit 0 ;;
  start)  ;;
  *)      echo "usage: $0 [start|stop|status|reset]"; exit 1 ;;
esac

for entry in "${PROCS[@]}"; do
  script="${entry#*:}"
  if pgrep -f "^${PYTHON} ${script}$" >/dev/null 2>&1; then
    echo "[run] ERROR: $script is already running. Run './run.sh stop' first."
    exit 1
  fi
done

echo "[run] starting gateway from $(pwd)"
echo "[run]   ship feeds : ${SHIP_HOST:-127.0.0.1}:5001-5004"
echo "[run]   shore uplink: ${SHORE_HOST:-127.0.0.1}:${SHORE_PORT:-6000}"

if command -v tmux >/dev/null 2>&1; then
  echo "[run] display mode: tmux (session '$SESSION')"
  tmux new-session -d -s "$SESSION" -n gateway "$PYTHON ingest.py"
  tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
  # Give ingest a moment to open the database before the sender reads it.
  sleep 2
  tmux split-window -t "$SESSION:gateway" "$PYTHON sender.py"
  tmux select-layout -t "$SESSION:gateway" even-vertical >/dev/null
  sleep 1
  echo "[run] both panes started."
  echo "[run] attach with:  tmux attach -t $SESSION      (detach: Ctrl-b then d)"
  echo "[run] stop with:    ./run.sh stop"
  exit 0
fi

echo "[run] display mode: foreground (Ctrl+C stops both)"
"$PYTHON" ingest.py & INGEST_PID=$!
sleep 2
"$PYTHON" sender.py & SENDER_PID=$!

cleanup() {
  echo ""
  echo "[run] shutting down..."
  # SIGTERM, not SIGKILL: both processes have handlers that drain and close
  # the database cleanly. Killing them outright risks leaving a stale WAL.
  kill -TERM "$INGEST_PID" "$SENDER_PID" 2>/dev/null
  wait "$INGEST_PID" "$SENDER_PID" 2>/dev/null
  echo "[run] both stopped."
  exit 0
}
trap cleanup INT TERM

echo "[run] ingest pid $INGEST_PID, sender pid $SENDER_PID"
wait
