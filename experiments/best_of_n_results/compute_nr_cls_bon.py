#!/usr/bin/env python3
"""Compute NR-CLS score for Best-of-N experiment results and compare with baseline."""
import pandas as pd
import os

BON_METRICS = '/root/autodl-tmp/best_of_n_output/metrics'
BASELINE_METRICS = '/root/autodl-tmp/testing_output/metrics'

NR_WEIGHTS = {
    'no_ego_at_fault_collisions': 5,
    'drivable_area_compliance': 5,
    'driving_direction_compliance': 5,
    'ego_is_comfortable': 2,
    'ego_is_making_progress': 5,
    'ego_progress_along_expert_route': 5,
    'time_to_collision_within_bound': 5,
    'speed_limit_compliance': 2,
}

def compute_score(metrics_dir, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if not os.path.exists(metrics_dir):
        print(f"  Metrics directory not found: {metrics_dir}")
        return None
    metric_means = {}
    total_w = sum(NR_WEIGHTS.values())
    for name, weight in NR_WEIGHTS.items():
        fp = os.path.join(metrics_dir, name + '.parquet')
        if not os.path.exists(fp):
            print(f"  WARNING: {name}.parquet NOT FOUND")
            continue
        df = pd.read_parquet(fp, columns=['scenario_name', 'metric_score'])
        avg = df['metric_score'].mean()
        passing = (df['metric_score'] >= 1.0).sum()
        n = len(df)
        metric_means[name] = avg
        print(f"  {name:45s}: {avg:.4f}  ({passing}/{n} pass)")
    if not metric_means:
        return None
    weighted_sum = sum(NR_WEIGHTS[k] * v for k, v in metric_means.items())
    score_wa = weighted_sum / total_w * 100
    product = 1.0
    for v in metric_means.values():
        product *= v
    score_prod = product * 100
    print(f"\n  Weighted Average: {score_wa:.2f}%")
    print(f"  Product:          {score_prod:.2f}%")
    return {'weighted_avg': score_wa, 'product': score_prod, 'means': metric_means}

def main():
    bon = compute_score(BON_METRICS, "Best-of-N (N=5)")
    baseline = compute_score(BASELINE_METRICS, "Baseline (N=1)")
    if bon and baseline:
        print(f"\n{'='*60}")
        print(f"  COMPARISON: Best-of-N vs Baseline")
        print(f"{'='*60}")
        print(f"  Baseline WA:   {baseline['weighted_avg']:.2f}%")
        print(f"  Best-of-N WA:  {bon['weighted_avg']:.2f}%")
        print(f"  Improvement:   {bon['weighted_avg'] - baseline['weighted_avg']:+.2f}%")
        print()
        print(f"  Per-metric improvement:")
        for name in NR_WEIGHTS:
            if name in bon['means'] and name in baseline['means']:
                diff = (bon['means'][name] - baseline['means'][name]) * 100
                print(f"    {name:45s}: {diff:+.2f}%")
    output_file = '/root/autodl-tmp/best_of_n_output/nr_cls_comparison.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        if bon:
            f.write(f"Best-of-N Weighted Average: {bon['weighted_avg']:.2f}%\n")
            f.write(f"Best-of-N Product: {bon['product']:.2f}%\n")
        if baseline:
            f.write(f"Baseline Weighted Average: {baseline['weighted_avg']:.2f}%\n")
            f.write(f"Baseline Product: {baseline['product']:.2f}%\n")
        if bon and baseline:
            f.write(f"Improvement (WA): {bon['weighted_avg'] - baseline['weighted_avg']:+.2f}%\n")
    print(f"\nResults saved to {output_file}")

if __name__ == '__main__':
    main()
