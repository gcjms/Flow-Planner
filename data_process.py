"""
Data Processing Script for Flow-Planner
Converts nuplan .db data into .npz format for training.
Based on Flow-Planner's built-in DataProcessor.
"""
import os
import argparse
import json

from flow_planner.data.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping


def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):
    scenario_types = None
    scenario_tokens = scenario_tokens
    log_names = log_names
    map_names = None

    num_scenarios_per_type = num_scenarios_per_type
    limit_total_scenarios = limit_total_scenarios
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None

    expand_scenarios = True
    remove_invalid_goals = False
    shuffle = shuffle

    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (scenario_types, scenario_tokens, log_names, map_names,
            num_scenarios_per_type, limit_total_scenarios,
            timestamp_threshold_s, ego_displacement_minimum_m,
            expand_scenarios, remove_invalid_goals, shuffle,
            ego_start_speed_threshold, ego_stop_speed_threshold,
            speed_noise_tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flow-Planner Data Processing')
    parser.add_argument('--data_path', type=str, required=True, help='path to nuplan .db files')
    parser.add_argument('--map_path', type=str, required=True, help='path to nuplan maps')
    parser.add_argument('--save_path', type=str, required=True, help='path to save processed npz data')
    parser.add_argument('--total_scenarios', type=int, default=1000000, help='limit total number of scenarios')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--shuffle_scenarios', action='store_true', default=True, help='shuffle scenarios')
    args = parser.parse_args()

    # Create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(
        args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios
    ))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # Process data
    del worker, builder, scenario_filter

    processor = DataProcessor(args.save_path)
    processor.work(scenarios)

    # Generate JSON file list
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    json_output_path = os.path.join(args.save_path, 'flow_planner_training.json')
    with open(json_output_path, 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names to {json_output_path}")
