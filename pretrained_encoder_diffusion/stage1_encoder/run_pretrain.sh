#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run MRI Encoder Pretraining in background with nohup
# Usage:
#   bash run_pretrain.sh          # start / resume pretraining
#   bash run_pretrain.sh status   # check if running
#   bash run_pretrain.sh tail     # tail the live log
#   bash run_pretrain.sh stop     # kill the background process
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/pretrain_encoder.log"
PID_FILE="$SCRIPT_DIR/logs/pretrain_encoder.pid"

mkdir -p "$SCRIPT_DIR/logs"

# ── Helpers ──────────────────────────────────────────────────────────────────
is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# ── Sub-commands ─────────────────────────────────────────────────────────────
case "${1:-start}" in

  start)
    if is_running; then
        echo "Already running (PID $(cat "$PID_FILE")). Use 'bash run_pretrain.sh tail' to follow logs."
        exit 0
    fi

    echo "Starting MRI encoder pretraining..."
    echo "Log file : $LOG_FILE"
    echo "PID file : $PID_FILE"

    cd "$SCRIPT_DIR" || exit 1

    nohup python3 pretrain_mri_encoder.py >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"

    echo ""
    echo "Pretraining started  (PID $!)"
    echo "Monitor progress:"
    echo "  tail -f $LOG_FILE"
    echo "  tensorboard --logdir $SCRIPT_DIR/runs_mri_encoder --port 6007"
    ;;

  status)
    if is_running; then
        echo "Running — PID $(cat "$PID_FILE")"
        echo "Log: $LOG_FILE"
        echo "Last 5 lines:"
        tail -5 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
    else
        echo "Not running."
        if [ -f "$LOG_FILE" ]; then
            echo "Last 5 lines of log:"
            tail -5 "$LOG_FILE"
        fi
    fi
    ;;

  tail)
    echo "Following $LOG_FILE (Ctrl+C to stop)..."
    tail -f "$LOG_FILE"
    ;;

  stop)
    if is_running; then
        kill "$(cat "$PID_FILE")" && echo "Stopped PID $(cat "$PID_FILE")." && rm "$PID_FILE"
    else
        echo "Not running."
    fi
    ;;

  *)
    echo "Usage: bash run_pretrain.sh [start|status|tail|stop]"
    ;;

esac
