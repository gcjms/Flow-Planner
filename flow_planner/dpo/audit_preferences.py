"""
Quick audit tool for DPO preference files.

Purpose:
  - summarize pair / scene / dim-label distributions
  - inspect score-gap quality when available
  - surface suspicious low-gap examples for manual review
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_details(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected detail JSON list, got {type(payload)!r}")
    return payload


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit preference dataset quality")
    parser.add_argument("--preference_path", type=str, required=True)
    parser.add_argument("--details_json", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.preference_path, allow_pickle=True)
    scenario_ids = data["scenario_ids"]

    summary: Dict[str, Any] = {
        "preference_path": args.preference_path,
        "pairs": int(len(scenario_ids)),
        "unique_scenes": int(len(set(scenario_ids.tolist()))),
    }

    if "dim_labels" in data:
        summary["dim_labels"] = dict(Counter(str(x) for x in data["dim_labels"].tolist()))

    if "score_gaps" in data:
        gaps = np.asarray(data["score_gaps"], dtype=np.float32)
        summary["score_gap_stats"] = _percentiles(gaps)
        summary["low_gap_counts"] = {
            "<=0.0": int(np.sum(gaps <= 0.0)),
            "<=0.1": int(np.sum(gaps <= 0.1)),
            "<=0.5": int(np.sum(gaps <= 0.5)),
            "<=1.0": int(np.sum(gaps <= 1.0)),
        }

    suspicious: List[Dict[str, Any]] = []
    if args.details_json:
        details = _load_details(args.details_json)
        summary["details_count"] = len(details)
        summary["method_counts"] = dict(
            Counter(str(item.get("method", "unknown")) for item in details)
        )
        if any("dim_label" in item for item in details):
            summary["detail_dim_labels"] = dict(
                Counter(str(item.get("dim_label", "unknown")) for item in details)
            )
        if any("score_gap" in item for item in details):
            detail_gaps = np.asarray(
                [float(item.get("score_gap", 0.0)) for item in details],
                dtype=np.float32,
            )
            summary["details_score_gap_stats"] = _percentiles(detail_gaps)

        def _sort_key(item: Dict[str, Any]) -> tuple:
            gap = float(item["score_gap"]) if "score_gap" in item else float("inf")
            legacy = str(item.get("dim_label", "")).startswith("legacy")
            return (abs(gap), 0 if legacy else 1)

        suspicious = sorted(details, key=_sort_key)[: args.top_k]

    report = {
        "summary": summary,
        "suspicious_examples": suspicious,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
