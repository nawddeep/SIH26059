#!/usr/bin/env bash
# Launches the shore-side receiver.
#
# The four simulated instrument emitters that used to run alongside it have
# been removed. Point the gateway's ingest.py at a real instrument feed with
# SHIP_HOST; there is no simulated ship any more.
#
# Three display modes, picked automatically in this order:
#   tmux        - five tiled panes in one session. Works over SSH, survives a
#                 dropped connection, and gives the "ship's bridge" look.
#   lxterminal  - one window per instrument. Needs a desktop session.
#   background  - nohup with one log file per feed. Always works.
#
# Usage:  ./run_all.sh          start everything
#         ./run_all.sh stop     stop everything
#         ./run_all.sh status   show what is running

set -uo pipefail
cd "$(dirname "$0")"

SESSION="shore"
LOGDIR="logs"
PYTHON="${PYTHON:-python3}"

FEEDS=(
  "shore:shore_listener.py"
)

# --- stop / status --------------------------------------------------------
stop_all() {
  echo "[run_all] stopping ship-sim..."
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION" && echo "[run_all]   tmux session '$SESSION' killed"
  fi
  for entry in "${FEEDS[@]}"; do
    script="${entry#*:}"
    if pkill -f "$PYTHON .*$script" 2>/dev/null; then
      echo "[run_all]   stopped $script"
    fi
  done
  echo "[run_all] done."
}

show_status() {
  echo "[run_all] shore processes:"
  local found=0
  for entry in "${FEEDS[@]}"; do
    script="${entry#*:}"
    if pgrep -f "$PYTHON .*$script" >/dev/null 2>&1; then
      printf "           %-22s RUNNING (pid %s)\n" "$script" "$(pgrep -f "$PYTHON .*$script" | tr '\n' ' ')"
      found=1
    else
      printf "           %-22s stopped\n" "$script"
    fi
  done
  [ "$found" -eq 0 ] && echo "[run_all]   nothing running"
  echo "[run_all] listening ports:"
  ss -ltn 2>/dev/null | grep -E ':(500[1-4]|6000)' || echo "           none"
}

case "${1:-start}" in
  stop)   stop_all; exit 0 ;;
  status) show_status; exit 0 ;;
  start)  ;;
  *)      echo "usage: $0 [start|stop|status]"; exit 1 ;;
esac

# --- refuse to double-start ----------------------------------------------
for entry in "${FEEDS[@]}"; do
  script="${entry#*:}"
  if pgrep -f "$PYTHON .*$script" >/dev/null 2>&1; then
    echo "[run_all] ERROR: $script is already running."
    echo "[run_all]        run './run_all.sh stop' first, or './run_all.sh status' to look."
    exit 1
  fi
done

echo "[run_all] starting shore receiver from $(pwd)"

# --- mode 1: tmux ---------------------------------------------------------
if command -v tmux >/dev/null 2>&1; then
  echo "[run_all] display mode: tmux (session '$SESSION')"

  first="${FEEDS[0]}"
  tmux new-session -d -s "$SESSION" -n bridge "$PYTHON ${first#*:}"
  # Keep a pane visible if its command dies, so a crash is diagnosable
  # instead of the pane simply vanishing mid-demo.
  tmux set-option -t "$SESSION" remain-on-exit on >/dev/null

  for entry in "${FEEDS[@]:1}"; do
    tmux split-window -t "$SESSION:bridge" "$PYTHON ${entry#*:}"
    tmux select-layout -t "$SESSION:bridge" tiled >/dev/null
  done
  tmux select-layout -t "$SESSION:bridge" tiled >/dev/null

  sleep 1
  echo "[run_all] shore receiver started."
  echo "[run_all] attach with:   tmux attach -t $SESSION"
  echo "[run_all] detach with:   Ctrl-b then d"
  echo "[run_all] stop with:     ./run_all.sh stop"
  exit 0
fi

# --- mode 2: lxterminal (desktop session) --------------------------------
if command -v lxterminal >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  echo "[run_all] display mode: lxterminal (one window per instrument)"
  for entry in "${FEEDS[@]}"; do
    name="${entry%%:*}"; script="${entry#*:}"
    lxterminal --title="${name^^} CONSOLE" -e "$PYTHON $script" &
    sleep 0.3
  done
  echo "[run_all] shore receiver window opened."
  exit 0
fi

# --- mode 3: background with log files -----------------------------------
echo "[run_all] display mode: background (no terminal multiplexer available)"
mkdir -p "$LOGDIR"
for entry in "${FEEDS[@]}"; do
  name="${entry%%:*}"; script="${entry#*:}"
  nohup "$PYTHON" "$script" > "$LOGDIR/$name.log" 2>&1 &
  echo "[run_all]   $script -> $LOGDIR/$name.log (pid $!)"
  sleep 0.3
done
echo "[run_all] follow a feed with:  tail -f $LOGDIR/gps.log"
echo "[run_all] stop with:           ./run_all.sh stop"
