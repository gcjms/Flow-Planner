#!/bin/bash
# Multi-City Data Processing & Training Pipeline
# Saved to persistent location to survive WSL restarts
set -e

MAP_PATH="/mnt/d/nuplan_data/dataset/maps"
SAVE_PATH="/home/gcjms/nuplan/dataset/processed_all_npz"
FLOW_DIR="/home/gcjms/Flow-Planner"
LOG="/home/gcjms/Flow-Planner/pipeline.log"

PITTSBURGH_DB="/mnt/f/data/cache/train_pittsburgh"
VEGAS_DB="/mnt/f/data/cache/train_vegas_1"
VAL_DB="/mnt/f/data/cache/val"

mkdir -p "$SAVE_PATH"

echo "Pipeline started: $(date)" | tee "$LOG"

eval "$(conda shell.bash hook)"
conda activate nuplan
cd "$FLOW_DIR"

# Pittsburgh
echo "[$(date +%H:%M)] Processing Pittsburgh (52G)..." | tee -a "$LOG"
python data_process.py --data_path "$PITTSBURGH_DB" --map_path "$MAP_PATH" --save_path "$SAVE_PATH" --total_scenarios 1000000 --shuffle_scenarios 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M)] Pittsburgh done. NPZ: $(ls $SAVE_PATH/*.npz | wc -l)" | tee -a "$LOG"

# Vegas
echo "[$(date +%H:%M)] Processing Vegas (237G)..." | tee -a "$LOG"
python data_process.py --data_path "$VEGAS_DB" --map_path "$MAP_PATH" --save_path "$SAVE_PATH" --total_scenarios 1000000 --shuffle_scenarios 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M)] Vegas done. NPZ: $(ls $SAVE_PATH/*.npz | wc -l)" | tee -a "$LOG"

# Val
VAL_SAVE="/home/gcjms/nuplan/dataset/processed_val_npz"
mkdir -p "$VAL_SAVE"
echo "[$(date +%H:%M)] Processing Val (152G)..." | tee -a "$LOG"
python data_process.py --data_path "$VAL_DB" --map_path "$MAP_PATH" --save_path "$VAL_SAVE" --total_scenarios 1000000 --shuffle_scenarios 2>&1 | tee -a "$LOG"
echo "[$(date +%H:%M)] Val done. NPZ: $(ls $VAL_SAVE/*.npz | wc -l)" | tee -a "$LOG"

# Shuffled splits
echo "[$(date +%H:%M)] Creating splits..." | tee -a "$LOG"
python3 -c "
import os, json, random
train_dir='$SAVE_PATH'
train_files=[f for f in os.listdir(train_dir) if f.endswith('.npz')]
random.shuffle(train_files)
for n in ['flow_planner_train.json','flow_planner_training.json']:
    with open(os.path.join(train_dir,n),'w') as f: json.dump(train_files,f,indent=2)
print(f'Train: {len(train_files)}')
val_dir='$VAL_SAVE'
val_files=[f for f in os.listdir(val_dir) if f.endswith('.npz')]
random.shuffle(val_files)
with open(os.path.join(val_dir,'flow_planner_val.json'),'w') as f: json.dump(val_files,f,indent=2)
print(f'Val: {len(val_files)}')
" 2>&1 | tee -a "$LOG"

# Training
echo "[$(date +%H:%M)] Launching training..." | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=null HYDRA_FULL_ERROR=1
export PROJECT_ROOT=/home/gcjms/Flow-Planner
export SAVE_DIR=/home/gcjms/Flow-Planner/training_output_multicity
export TENSORBOARD_LOG_PATH=$SAVE_DIR/tb_logs
export TRAINING_DATA="$SAVE_PATH"
export TRAINING_JSON="$SAVE_PATH/flow_planner_training.json"
mkdir -p "$SAVE_DIR" "$TENSORBOARD_LOG_PATH"
cd "$FLOW_DIR/flow_planner/run_script"
python -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone \
    ../trainer.py --config-name flow_planner_standard \
    train.batch_size=32 train.epoch=200 2>&1 | tee -a "$LOG"

echo "Pipeline completed: $(date)" | tee -a "$LOG"
