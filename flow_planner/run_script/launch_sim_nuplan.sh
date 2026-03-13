export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT=/home/gcjms/nuplan-devkit # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT=/home/gcjms/nuplan/dataset # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT=/home/gcjms/nuplan/dataset/maps # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT=/home/gcjms/Flow-Planner/testing_output # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT=debug_2

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE=closed_loop_nonreactive_agents # e.g., "closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=flow_planner_release
CONFIG_FILE=/home/gcjms/Flow-Planner/training_output/outputs/FlowPlannerTraining/flow_planner_standard/2026-03-11_21-40-45/.hydra/config.yaml # path of .hydra/config in ckpt folder
CKPT_FILE=/home/gcjms/Flow-Planner/training_output/outputs/FlowPlannerTraining/flow_planner_standard/2026-03-11_21-40-45/latest.pth # path to the .pth of checkpoint

if [ "$SPLIT" == "val14" ] || [ "$SPLIT" == "one_continuous_log" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

/home/gcjms/miniconda3/envs/flow_planner/bin/python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_builder.data_root=$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=4 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=1.0 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"