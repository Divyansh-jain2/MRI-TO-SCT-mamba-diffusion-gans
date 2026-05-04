#!/bin/bash

# Inference script for hybrid model (runs in background with nohup)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/inference_${TIMESTAMP}.log"

echo "Starting hybrid inference..."
echo "Log file: $LOG_FILE"
echo "Monitor with: tail -f $LOG_FILE"
echo ""

nohup python3 inference_hybrid.py --mc_runs 5 > "$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "logs/inference_${TIMESTAMP}.pid"

echo "Inference started with PID: $PID"
echo ""
echo "Useful commands:"
echo "  - tail -f $LOG_FILE                          (monitor progress)"
echo "  - ps aux | grep inference_hybrid              (check if running)"
echo "  - kill \$(cat logs/inference_${TIMESTAMP}.pid)  (stop inference)"
