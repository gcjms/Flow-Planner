#!/bin/bash
# Overnight Risk Network Pipeline
# Usage: bash run_overnight.sh

set -e
export PYTHONUNBUFFERED=1

echo "[$(date +%H:%M:%S)] Starting overnight pipeline..."

# Activate conda
source /home/gcjms/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

cd /home/gcjms/Flow-Planner
python3 -u run_overnight.py

echo "[$(date +%H:%M:%S)] Pipeline finished."
