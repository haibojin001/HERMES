#!/usr/bin/env python3
"""Read official Harbor or legacy Terminal-Bench verifier results after runs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path


CATEGORIES = ("build", "monitor", "issue_resolving", "test_generation")
CATEGORY_LABELS = {
    "build": "build_configuration",
    "monitor": "monitoring",
    "issue_resolving": "issue_resolving",
    "test_generation": "test_generation",
}


def _task_ids(root: Path, category: str | None = None) -> set[str]:
    task_root = root / "tasks"
    if category:
        task_root /= category
    extension = "task.yaml" if category else "task.toml"
    return {path.parent.name for path in task_root.glob(f"*/{extension}")}


def _token_usage(row: dict) -> tuple[int | None, int | None]:
    context = row.get("agent_result") or {}
    return context.get("n_input_tokens"), context.get("n_output_tokens")


def read_harbor(results_root: Path, benchmark: str, tag: str,
                benchmark_root: Path, repeats: int) -> tuple[list[dict], dict]:
    expected = _task_ids(benchmark_root)
    expected_count = {"swe_refactor": 20, "terminal_bench": 66}[benchmark]
    if len(expected) != expected_count:
        raise ValueError(
            f"{benchmark}: expected {expected_count} pinned task directories, "
            f"found {len(expected)}")
    records = []
    input_tokens = output_tokens = 0
    missing_usage = 0
    for index in range(repeats):
        job = results_root / f"hermes_{benchmark}_{tag}_r{index}"
        if not job.is_dir():
            raise ValueError(f"missing official Harbor job: {job}")
        seen = set()
        for file in sorted(job.rglob("result.json")):
            row = json.loads(file.read_text())
            if "task_name" not in row:  # job-level result.json
                continue
            if row.get("exception_info"):
                raise ValueError(f"failed trial: {file}")
            task_id = row["task_name"].rsplit("/", 1)[-1]
            if task_id not in expected:
                raise ValueError(f"unknown task in {file}: {task_id}")
            if task_id in seen:
                raise ValueError(f"duplicate task in {job}: {task_id}")
            seen.add(task_id)
            rewards = (row.get("verifier_result") or {}).get("rewards") or {}
            if benchmark == "swe_refactor" and rewards.get("valid") != 1:
                raise ValueError(f"invalid SWE Refactor grading: {file}")
            reward = rewards.get("reward")
            if (not isinstance(reward, (int, float)) or isinstance(reward, bool)
                    or not 0 <= reward <= 1):
                raise ValueError(f"missing or invalid official reward: {file}")
            tokens_in, tokens_out = _token_usage(row)
            if tokens_in is None or tokens_out is None:
                missing_usage += 1
            else:
                input_tokens += tokens_in
                output_tokens += tokens_out
            records.append({
                "benchmark": benchmark, "task_id": task_id,
                "run_index": index,
                "score": reward if benchmark == "swe_refactor"
                else int(reward == 1),
                "raw_reward": reward,
                "source": str(file.resolve()),
            })
        if seen != expected:
            raise ValueError(
                f"{job}: missing {len(expected - seen)} official task results; "
                f"examples: {sorted(expected - seen)[:5]}")
    usage = {"input_tokens": input_tokens if not missing_usage else None,
             "output_tokens": output_tokens if not missing_usage else None,
             "trials_missing_usage": missing_usage}
    return records, usage


def read_devops(results_root: Path, tag: str, benchmark_root: Path
                ) -> tuple[list[dict], dict]:
    records = []
    input_tokens = output_tokens = 0
    missing_usage = 0
    counts = {}
    for category in CATEGORIES:
        expected = _task_ids(benchmark_root, category)
        if not expected:
            raise ValueError(f"no {category} tasks in pinned DevOps-Gym")
        counts[CATEGORY_LABELS[category]] = len(expected)
        file = (results_root / f"hermes_{category}_{tag}" / "results.json")
        if not file.is_file():
            raise ValueError(f"missing official legacy TB result: {file}")
        rows = json.loads(file.read_text()).get("results")
        if not isinstance(rows, list):
            raise ValueError(f"invalid legacy TB result: {file}")
        seen = set()
        for row in rows:
            task_id = row["task_id"]
            if task_id not in expected or task_id in seen:
                raise ValueError(f"unknown or duplicate task: {task_id} in {file}")
            seen.add(task_id)
            resolved = row.get("is_resolved")
            if not isinstance(resolved, bool):
                raise ValueError(f"ungraded DevOps-Gym task: {task_id}")
            tokens_in = row.get("total_input_tokens")
            tokens_out = row.get("total_output_tokens")
            if tokens_in is None or tokens_out is None:
                missing_usage += 1
            else:
                input_tokens += tokens_in
                output_tokens += tokens_out
            records.append({
                "benchmark": "devops_gym", "category": CATEGORY_LABELS[category],
                "task_id": task_id, "run_index": 0, "score": int(resolved),
                "source": str(file.resolve()),
            })
        if seen != expected:
            raise ValueError(
                f"{category}: missing {len(expected - seen)} official task results")
    usage = {"input_tokens": input_tokens if not missing_usage else None,
             "output_tokens": output_tokens if not missing_usage else None,
             "trials_missing_usage": missing_usage,
             "category_task_counts": counts}
    return records, usage


def summarize(records: list[dict], benchmark: str, repeats: int,
              usage: dict) -> dict:
    report = {"benchmark": benchmark, "tasks": len(records) // repeats,
              "repeats": repeats, "graded_trials": len(records), **usage}
    if benchmark == "swe_refactor":
        runs = [[r["score"] for r in records if r["run_index"] == index]
                for index in range(repeats)]
        means = [100 * statistics.mean(run) for run in runs]
        report["composite_pct"] = statistics.mean(means)
        report["per_run_pct"] = means
        if repeats > 1:
            report["std_pct"] = statistics.stdev(means)
    elif benchmark == "terminal_bench":
        means = [100 * statistics.mean(
            r["score"] for r in records if r["run_index"] == index)
            for index in range(repeats)]
        report["resolution_pct"] = statistics.mean(means)
        report["std_pct"] = statistics.stdev(means)
        report["per_run_pct"] = means
    else:
        groups = defaultdict(list)
        for row in records:
            groups[row["category"]].append(row["score"])
        rates = {category: 100 * statistics.mean(groups[category])
                 for category in CATEGORY_LABELS.values()}
        report["category_success_pct"] = rates
        report["unweighted_average_pct"] = statistics.mean(rates.values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark",
                        choices=("swe_refactor", "terminal_bench", "devops_gym"),
                        required=True)
    parser.add_argument("--results", type=Path, required=True,
                        help="Harbor jobs-dir or legacy tb output-path")
    parser.add_argument("--benchmark-root", type=Path,
                        help="pinned official benchmark checkout")
    parser.add_argument("--tag", default="paper", help="HERMES_RUN_TAG used for runs")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark_home = Path(os.environ.get(
        "HERMES_BENCHMARKS",
        Path(__file__).resolve().parent.parent / "hermes_data" / "benchmarks"))
    root = args.benchmark_root or benchmark_home / args.benchmark
    repeats = args.repeats or (5 if args.benchmark == "terminal_bench" else 1)
    if repeats < 1 or (args.benchmark == "terminal_bench" and repeats != 5):
        parser.error("Terminal-Bench requires five runs; repeats must be positive")
    if args.benchmark == "devops_gym":
        if repeats != 1:
            parser.error("DevOps-Gym report expects one run per category")
        rows, usage = read_devops(args.results, args.tag, root)
    else:
        rows, usage = read_harbor(args.results, args.benchmark, args.tag,
                                  root, repeats)
    report = summarize(rows, args.benchmark, repeats, usage)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix(".scores.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
