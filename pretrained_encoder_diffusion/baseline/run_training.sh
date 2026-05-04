#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Run the Python script in the background and log output
echo "Starting training script..."
echo "Log file: $LOG_FILE"
echo "Monitor with: tail -f $LOG_FILE"
echo ""

nohup python3 main.py > "$LOG_FILE" 2>&1 &

# Get the PID of the background process
PID=$!
echo "Training process started with PID: $PID"
echo "PID saved to: logs/training_${TIMESTAMP}.pid"

# Save PID to file for easy reference
echo $PID > "logs/training_${TIMESTAMP}.pid"

echo "Process running in background. Use 'tail -f $LOG_FILE' to monitor progress."
