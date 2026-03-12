export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT=/home/gcjms/Flow-Planner
export SAVE_DIR=/home/gcjms/Flow-Planner/training_output
export TENSORBOARD_LOG_PATH=/home/gcjms/Flow-Planner/training_output/tb_logs
export TRAINING_DATA=/home/gcjms/nuplan/dataset/processed_npz
export TRAINING_JSON=/home/gcjms/nuplan/dataset/processed_npz/flow_planner_training.json
export TORCH_LOGS="dynamic,recompiles"

python -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone ../trainer.py --config-name flow_planner_standard
