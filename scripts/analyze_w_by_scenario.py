"""
Per-Scenario-Type CFG Weight Analysis
======================================
读取多个 w 值的闭环仿真 metrics，按场景类型分组计算 NR Score，
输出对比报告。

用法:
    python scripts/analyze_w_by_scenario.py \
        --results_dir /root/autodl-tmp/w_grid_search_output \
        --output_file /root/autodl-tmp/w_grid_search_output/w_grid_search_report.md
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime


# val14 的 14 种场景类型
SCENARIO_TYPES = [
    'starting_left_turn',
    'starting_right_turn',
    'starting_straight_traffic_light_intersection_traversal',
    'stopping_with_lead',
    'high_lateral_acceleration',
    'high_magnitude_speed',
    'low_magnitude_speed',
    'traversing_pickup_dropoff',
    'waiting_for_pedestrian_to_cross',
    'behind_long_vehicle',
    'stationary_in_traffic',
    'near_multiple_vehicles',
    'changing_lane',
    'following_lane_with_lead',
]

# 场景类型的中文名称
SCENARIO_NAMES_CN = {
    'starting_left_turn': '左转起步',
    'starting_right_turn': '右转起步',
    'starting_straight_traffic_light_intersection_traversal': '直行过信号灯',
    'stopping_with_lead': '前车减速停车',
    'high_lateral_acceleration': '高横向加速度',
    'high_magnitude_speed': '高速行驶',
    'low_magnitude_speed': '低速行驶',
    'traversing_pickup_dropoff': '路过上下客区',
    'waiting_for_pedestrian_to_cross': '等行人过马路',
    'behind_long_vehicle': '跟大车',
    'stationary_in_traffic': '拥堵静止',
    'near_multiple_vehicles': '周围多车',
    'changing_lane': '变道',
    'following_lane_with_lead': '跟车行驶',
}


def extract_scenario_type(filename):
    """从 metric 文件名提取场景类型"""
    basename = os.path.basename(filename)
    # 格式: {scenario_type}_{token}_diffusion_planner.pickle[.temp]
    clean = basename.replace('_diffusion_planner.pickle.temp', '')
    clean = clean.replace('_diffusion_planner.pickle', '')

    for st in SCENARIO_TYPES:
        if clean.startswith(st + '_'):
            token = clean[len(st) + 1:]
            return st, token
    return None, None


def extract_score_from_metric(data):
    """
    从 nuPlan metric pickle 数据中提取场景得分。

    nuPlan 的 metric pickle 文件包含 MetricStatisticsDataFrame 或类似对象。
    这里尝试多种方式解析：
    1. 如果是包含 metric_statistics 的对象列表
    2. 如果是 pandas DataFrame
    3. 如果是嵌套的 dict/list 结构
    """
    try:
        # 方式 1: nuPlan MetricFileResult 列表
        # 每个元素有 metric_statistics 属性，里面是 MetricStatistics 列表
        if isinstance(data, list):
            all_scores = []
            for metric_result in data:
                # MetricFileResult → metric_statistics: List[MetricStatistics]
                stats = None
                if hasattr(metric_result, 'metric_statistics'):
                    stats = metric_result.metric_statistics
                elif hasattr(metric_result, 'statistics'):
                    stats = metric_result.statistics

                if stats is not None:
                    for stat in (stats if isinstance(stats, list) else [stats]):
                        # MetricStatistics → metric_score / value
                        score = None
                        if hasattr(stat, 'metric_score'):
                            score = stat.metric_score
                        elif hasattr(stat, 'value'):
                            score = stat.value
                        elif hasattr(stat, 'score'):
                            score = stat.score

                        if score is not None and isinstance(score, (int, float)):
                            all_scores.append(float(score))

            if all_scores:
                return np.mean(all_scores)

        # 方式 2: 单个 MetricFileResult
        if hasattr(data, 'metric_statistics'):
            scores = []
            for stat in data.metric_statistics:
                if hasattr(stat, 'metric_score'):
                    scores.append(stat.metric_score)
            if scores:
                return np.mean(scores)

        # 方式 3: DataFrame
        if hasattr(data, 'to_dict') or hasattr(data, 'values'):
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    if 'score' in data.columns:
                        return data['score'].mean()
                    elif 'metric_score' in data.columns:
                        return data['metric_score'].mean()
            except ImportError:
                pass

        # 方式 4: 递归搜索 score 字段
        score = _recursive_score_search(data, depth=0)
        if score is not None:
            return score

    except Exception as e:
        pass

    return None


def _recursive_score_search(obj, depth=0):
    """递归搜索对象中的 score/metric_score 字段"""
    if depth > 5:
        return None

    if isinstance(obj, (int, float)):
        return None

    # 检查常见的 score 属性
    for attr in ['metric_score', 'score', 'value', 'result']:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, (int, float)) and 0 <= val <= 1:
                return float(val)

    # 检查 dict
    if isinstance(obj, dict):
        for key in ['metric_score', 'score', 'value']:
            if key in obj and isinstance(obj[key], (int, float)):
                return float(obj[key])

    return None


def load_metrics(metrics_dir):
    """加载一个 w 值的所有 metric pickle 文件，返回 {(scenario_type, token): score}"""
    results = {}
    pattern = os.path.join(metrics_dir, '*.pickle*')
    files = glob.glob(pattern)

    loaded = 0
    failed = 0
    no_score = 0

    for f in files:
        scenario_type, token = extract_scenario_type(f)
        if scenario_type is None:
            continue

        try:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)

            score = extract_score_from_metric(data)
            if score is not None:
                results[(scenario_type, token)] = score
                loaded += 1
            else:
                no_score += 1
                # Debug: 打印第一个无法解析的结构
                if no_score == 1:
                    print(f"  DEBUG: Could not extract score from {os.path.basename(f)}")
                    print(f"    Data type: {type(data)}")
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        print(f"    Item[0] type: {type(item)}")
                        if hasattr(item, '__dict__'):
                            print(f"    Item[0] attrs: {list(item.__dict__.keys())}")
                            # 打印每个属性的类型和部分值
                            for k, v in list(item.__dict__.items())[:5]:
                                print(f"      {k}: {type(v)} = {str(v)[:100]}")

        except Exception as e:
            failed += 1
            if failed == 1:
                print(f"  Warning: Failed to load {os.path.basename(f)}: {e}")

    print(f"  Results: {loaded} scored, {no_score} no-score, {failed} failed, {len(files)} total")
    return results


def generate_report(grouped_scores, w_values, output_file):
    """生成分析报告"""
    lines = []
    lines.append("# Per-Scenario-Type CFG Weight Grid Search Report")
    lines.append("")
    lines.append(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**W values tested**: {', '.join(f'{w:.1f}' for w in sorted(w_values))}")
    lines.append("")

    # 主表格
    lines.append("## 1. NR Score by Scenario Type and W")
    lines.append("")

    w_sorted = sorted(w_values)
    header = "| 场景类型 | 中文名 | N | " + " | ".join(f"w={w:.1f}" for w in w_sorted) + " | 最优w | Δ |"
    separator = "|" + "|".join(["---"] * (5 + len(w_sorted))) + "|"
    lines.append(header)
    lines.append(separator)

    summary_data = []

    for st in SCENARIO_TYPES:
        if st not in grouped_scores:
            continue

        cn_name = SCENARIO_NAMES_CN.get(st, st)
        w_means = {}
        n = 0

        for w in w_sorted:
            scores = grouped_scores[st].get(w, [])
            if scores:
                w_means[w] = np.mean(scores) * 100
                n = max(n, len(scores))

        if not w_means:
            continue

        best_w = max(w_means, key=w_means.get)
        worst = min(w_means.values())
        gap = w_means[best_w] - worst

        score_strs = []
        for w in w_sorted:
            s = w_means.get(w)
            if s is None:
                score_strs.append("N/A")
            elif w == best_w:
                score_strs.append(f"**{s:.1f}**")
            else:
                score_strs.append(f"{s:.1f}")

        row = f"| {st} | {cn_name} | {n} | " + " | ".join(score_strs) + f" | {best_w:.1f} | {gap:.1f} |"
        lines.append(row)
        summary_data.append({'type': st, 'cn_name': cn_name, 'best_w': best_w, 'gap': gap})

    # 汇总
    lines.append("")
    lines.append("## 2. 结论")
    lines.append("")

    if summary_data:
        summary_data.sort(key=lambda x: x['gap'], reverse=True)
        lines.append("### 场景对 w 的敏感度排名")
        lines.append("")
        for i, d in enumerate(summary_data, 1):
            emoji = "🔴" if d['gap'] > 3 else ("🟡" if d['gap'] > 1 else "🟢")
            lines.append(f"{i}. {emoji} **{d['cn_name']}** ({d['type']}): Δ={d['gap']:.1f}, 最优 w={d['best_w']:.1f}")

        large_gap = [d for d in summary_data if d['gap'] > 3]
        lines.append("")
        if large_gap:
            lines.append(f"### ✅ {len(large_gap)} 种场景类型的最优 w 差异 > 3 分，自适应 w 有价值")
        else:
            lines.append("### ❌ 所有场景 Δ ≤ 3 分，w 影响微弱，建议投入 Best-of-N")
    else:
        lines.append("### ⚠️ 无法提取分数")
        lines.append("metric pickle 格式可能不兼容，请手动检查 pickle 文件结构。")
        lines.append("运行以下命令查看格式：")
        lines.append("```python")
        lines.append("import pickle")
        lines.append("with open('<metric_file>', 'rb') as f:")
        lines.append("    data = pickle.load(f)")
        lines.append("print(type(data))")
        lines.append("if isinstance(data, list): print(type(data[0]), dir(data[0]))")
        lines.append("```")

    report = "\n".join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {output_file}")
    print("\n" + "=" * 60)
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze per-scenario-type NR Score for different w values')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Root dir containing w_0.5/, w_1.0/, ... subdirectories')
    parser.add_argument('--output_file', type=str, default='w_grid_search_report.md',
                        help='Output report file path')
    args = parser.parse_args()

    # 发现所有 w 值目录
    w_dirs = sorted(glob.glob(os.path.join(args.results_dir, 'w_*')))
    if not w_dirs:
        print(f"Error: No w_* directories found in {args.results_dir}")
        sys.exit(1)

    # {scenario_type: {w: [scores]}}
    grouped = defaultdict(lambda: defaultdict(list))
    w_values = set()

    for w_dir in w_dirs:
        try:
            w = float(os.path.basename(w_dir).split('_')[1])
        except (IndexError, ValueError):
            continue

        # 递归查找 metrics 目录
        metrics_dir = None
        for root, dirs, files in os.walk(w_dir):
            if os.path.basename(root) == 'metrics':
                pickle_files = [f for f in files if '.pickle' in f]
                if pickle_files:
                    metrics_dir = root
                    break

        if metrics_dir is None:
            print(f"Warning: No metrics for w={w}")
            continue

        print(f"\nLoading metrics for w={w} from {metrics_dir}...")
        results = load_metrics(metrics_dir)

        if results:
            w_values.add(w)
            for (st, token), score in results.items():
                grouped[st][w].append(score)

    if not w_values:
        print("Error: No valid metrics loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(w_values)} w values: {sorted(w_values)}")
    generate_report(grouped, w_values, args.output_file)


if __name__ == '__main__':
    main()
