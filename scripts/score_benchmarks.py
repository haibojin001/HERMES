#!/usr/bin/env python3
"""Aggregate official post-run scores for the three manifest-based benchmarks."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes.manifest_runner import load_manifest


def read_jsonl(path: Path) -> list[dict]:
    if path.is_dir():
        return [json.loads(file.read_text()) for file in sorted(
            path.glob("*/result.json"))]
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def summarize(manifest: list[dict], runs: list[dict], scores: list[dict],
              benchmark: str, expected_repeats: int) -> dict:
    tasks = {row["task_id"]: row for row in manifest
             if row["benchmark"] == benchmark}
    if not tasks:
        raise ValueError(f"manifest contains no {benchmark} tasks")
    done = {(r["task_id"], r["run_index"]) for r in runs
            if r.get("benchmark") == benchmark}
    values = {}
    for row in scores:
        if row.get("benchmark") != benchmark:
            continue
        key = row["task_id"], row["run_index"]
        if key in values:
            raise ValueError(f"duplicate score: {key}")
        if key[0] not in tasks:
            raise ValueError(f"score for unknown task: {key}")
        if key not in done:
            raise ValueError(f"score has no matching run: {key}")
        if not row.get("source"):
            raise ValueError(f"score lacks official grader source: {key}")
        value = row["score"]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"score must be in [0, 1]: {key}")
        if benchmark in ("terminal_bench", "devops_gym") and value not in (0, 1):
            raise ValueError(f"{benchmark} requires binary outcomes: {key}")
        values[key] = value
    expected = {(task_id, index) for task_id in tasks
                for index in range(expected_repeats)}
    if done & expected != expected:
        raise ValueError(f"missing runs: {sorted(expected - done)[:10]}")
    if values.keys() != expected:
        raise ValueError(f"missing official scores: {sorted(expected - values.keys())[:10]}")
    result = {"benchmark": benchmark, "tasks": len(tasks),
              "repeats": expected_repeats, "graded": len(values)}
    if benchmark == "terminal_bench":
        per_run = [100 * statistics.mean(values[task_id, index] for task_id in tasks)
                   for index in range(expected_repeats)]
        result["resolution_pct"] = statistics.mean(per_run)
        result["std_pct"] = statistics.stdev(per_run) if len(per_run) > 1 else 0.0
        result["per_run_pct"] = per_run
    elif benchmark == "swe_refactor":
        result["composite_pct"] = 100 * statistics.mean(values.values())
    else:
        groups = defaultdict(list)
        for (task_id, _), value in values.items():
            groups[tasks[task_id]["category"]].append(value)
        if len(groups) != 4:
            raise ValueError("DevOps-Gym requires all four categories")
        category_rates = {name: 100 * statistics.mean(group)
                          for name, group in sorted(groups.items())}
        result["category_success_pct"] = category_rates
        result["unweighted_average_pct"] = statistics.mean(category_rates.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("swe_refactor", "terminal_bench",
                                                "devops_gym"), required=True)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repeats = args.repeats or (5 if args.benchmark == "terminal_bench" else 1)
    manifest = load_manifest(args.manifest)
    count = sum(row["benchmark"] == args.benchmark for row in manifest)
    expected_tasks = {"swe_refactor": 20, "terminal_bench": 66}.get(args.benchmark)
    if expected_tasks and count != expected_tasks:
        parser.error(f"paper protocol needs {expected_tasks} tasks; manifest has {count}")
    if args.benchmark == "terminal_bench" and repeats != 5:
        parser.error("paper protocol needs five Terminal-Bench runs per task")
    report = summarize(manifest, read_jsonl(args.runs), read_jsonl(args.scores),
                       args.benchmark, repeats)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
