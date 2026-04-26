#!/usr/bin/env python3
"""Append a structured anchor experiment record to a markdown log.

The script intentionally stays dependency-free so it can run on the AutoDL
machine from any branch snapshot.  It appends a human-readable block with the
same fields we want for paper reuse: purpose, data, method, artifacts, results,
interpretation, and next action.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable, List


DEFAULT_DOC = "docs/experiments/anchor_conditioned.md"


def _lines(title: str, values: Iterable[str]) -> List[str]:
    rows = [str(value).strip() for value in values if str(value).strip()]
    if not rows:
        return []
    out = [f"- {title}:"]
    out.extend(f"  - {row}" for row in rows)
    return out


def _format_eval_json(path: str) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    fields = {
        "collision": summary.get("collision_rate"),
        "progress": summary.get("avg_progress"),
        "route": summary.get("avg_route"),
        "collision_score": summary.get("avg_collision_score"),
        "scenes": summary.get("scenes_evaluated"),
        "failed": summary.get("scenes_failed"),
    }
    formatted = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            formatted.append(f"{key} {value:.4f}")
        else:
            formatted.append(f"{key} {value}")
    return f"`{path}` -> " + ", ".join(formatted)


def build_record(args: argparse.Namespace) -> str:
    today = args.date or dt.datetime.now().strftime("%Y%m%d")
    title = args.title.strip()
    if args.include_date and today not in title:
        title = f"{title} {today}"

    lines: List[str] = [f"## Experiment: {title}", ""]
    if args.status:
        lines.append(f"- Status: {args.status}")
    lines.extend(_lines("Goal", args.goal))
    lines.extend(_lines("Data", args.data))
    lines.extend(_lines("Method", args.method))
    lines.extend(_lines("Artifacts", args.artifact))
    lines.extend(_lines("Results", args.result))

    eval_results = [_format_eval_json(path) for path in args.eval_json]
    lines.extend(_lines("Eval JSON results", eval_results))
    lines.extend(_lines("Interpretation", args.interpretation))
    lines.extend(_lines("Next", args.next))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", default=DEFAULT_DOC, help=f"Markdown log path. Default: {DEFAULT_DOC}")
    parser.add_argument("--title", required=True, help="Experiment title after '## Experiment:'.")
    parser.add_argument("--date", default=None, help="Override date tag, e.g. 20260426.")
    parser.add_argument("--no-date", dest="include_date", action="store_false", help="Do not append date to title.")
    parser.set_defaults(include_date=True)
    parser.add_argument("--status", default="", help="Optional status, e.g. running / completed / negative.")
    parser.add_argument("--goal", action="append", default=[], help="Purpose of the experiment. Repeatable.")
    parser.add_argument("--data", action="append", default=[], help="Dataset / manifest / sample count. Repeatable.")
    parser.add_argument("--method", action="append", default=[], help="Method or key hyperparameters. Repeatable.")
    parser.add_argument("--artifact", action="append", default=[], help="Output path, log, checkpoint, patch. Repeatable.")
    parser.add_argument("--result", action="append", default=[], help="Important numeric result. Repeatable.")
    parser.add_argument("--eval-json", action="append", default=[], help="Eval JSON to summarize. Repeatable.")
    parser.add_argument("--interpretation", action="append", default=[], help="What the result means. Repeatable.")
    parser.add_argument("--next", action="append", default=[], help="Next action. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print record without writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = build_record(args)
    if args.dry_run:
        print(record)
        return

    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    existing = doc.read_text(encoding="utf-8") if doc.exists() else ""
    separator = "\n" if existing.endswith("\n") or not existing else "\n\n"
    doc.write_text(existing + separator + record, encoding="utf-8")
    print(f"Appended experiment record to {doc}")


if __name__ == "__main__":
    main()
