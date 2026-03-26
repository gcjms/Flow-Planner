#!/bin/bash
# 夜间自动验证链：Best-of-N → 官方权重开环
# Best-of-N 已在后台运行，这个脚本等它结束后自动启动官方权重评估

set -e
source /home/gcjms/miniconda3/etc/profile.d/conda.sh
conda activate flow_planner

echo "[$(date '+%H:%M:%S')] Waiting for Best-of-N validation (PID 372685) to finish..."

# Wait for Best-of-N to finish
while kill -0 372685 2>/dev/null; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Best-of-N done. Starting official weights evaluation..."
python3 -u /home/gcjms/Flow-Planner/run_official_eval.py

echo "[$(date '+%H:%M:%S')] All validations complete!"
