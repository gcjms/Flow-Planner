#!/bin/bash
# ============================================================
# 监控训练进程，训练完成后自动执行批量 eval
# 用法: nohup bash monitor_and_eval.sh &
# ============================================================

LOG_FILE="/home/gcjms/Flow-Planner/training_output/boston_resume_train.log"
EVAL_SCRIPT="/home/gcjms/Flow-Planner/eval_all_ckpts.sh"
MONITOR_LOG="/home/gcjms/Flow-Planner/training_output/monitor.log"

echo "[$(date)] 开始监控训练进程..." | tee "$MONITOR_LOG"

# 等待训练完成
while true; do
    # 检查训练进程是否还在运行
    if ! pgrep -f "trainer.py.*flow_planner_standard" > /dev/null 2>&1; then
        echo "[$(date)] 训练进程已结束" | tee -a "$MONITOR_LOG"
        break
    fi
    
    # 每 5 分钟输出一次进度
    LAST_LINE=$(tail -n 1 "$LOG_FILE" 2>/dev/null)
    echo "[$(date)] 训练中... $LAST_LINE" | tee -a "$MONITOR_LOG"
    sleep 300
done

# 检查训练是否成功完成
if grep -q "Training finished" "$LOG_FILE" 2>/dev/null; then
    echo "[$(date)] ✅ 训练成功完成，开始批量评估..." | tee -a "$MONITOR_LOG"
    
    # 执行批量评估
    bash "$EVAL_SCRIPT" 2>&1 | tee -a "$MONITOR_LOG"
    
    echo "[$(date)] ✅ 批量评估完成！" | tee -a "$MONITOR_LOG"
else
    echo "[$(date)] ❌ 训练未正常完成，请检查日志: $LOG_FILE" | tee -a "$MONITOR_LOG"
fi
