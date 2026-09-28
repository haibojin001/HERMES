"""Run HERMES on isolated, task-visible workspaces described by a JSONL manifest.

This is a transport for benchmark snapshots other than SWE-bench Verified.
The manifest is supplied by the benchmark adapter. It contains only task-visible
information; official graders run separately after this process has exited.
Run this module inside a benchmark-provided container or sandbox.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from hermes import pipeline as p
from hermes import trajectory as traj
from hermes.artifacts import edit_component


BENCHMARKS = {"swe_refactor", "terminal_bench", "devops_gym"}
CATEGORIES = {"build_configuration", "monitoring", "issue_resolving",
              "test_generation"}


def load_manifest(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        for key in ("task_id", "benchmark", "workspace", "issue", "visible_commands"):
            if key not in row:
                raise ValueError(f"{path}:{number}: missing {key}")
        if any(key in row for key in ("grader", "hidden_tests", "official_score",
                                     "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")):
            raise ValueError(f"{path}:{number}: hidden grader data in task manifest")
        if row["benchmark"] not in BENCHMARKS:
            raise ValueError(f"{path}:{number}: unknown benchmark")
        if not isinstance(row["visible_commands"], list) or not row["visible_commands"]:
            raise ValueError(f"{path}:{number}: visible_commands must be nonempty")
        if not all(isinstance(cmd, str) and cmd.strip()
                   for cmd in row["visible_commands"]):
            raise ValueError(f"{path}:{number}: invalid visible command")
        if row["benchmark"] == "devops_gym" and row.get("category") not in CATEGORIES:
            raise ValueError(f"{path}:{number}: invalid DevOps-Gym category")
        task_key = (row["benchmark"], row["task_id"])
        if task_key in seen:
            raise ValueError(f"{path}:{number}: duplicate task {task_key}")
        seen.add(task_key)
        workspace = Path(row["workspace"]).expanduser()
        if not workspace.is_dir():
            raise ValueError(f"{path}:{number}: workspace missing: {workspace}")
        row["workspace"] = str(workspace.resolve())
        rows.append(row)
    return rows


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def materialize(row: dict, out_dir: Path) -> Path:
    """Create a fresh per-task copy; the supplied benchmark snapshot is read only."""
    source = Path(row["workspace"])
    target = out_dir / "workspace"
    if source == out_dir or source in out_dir.parents:
        raise ValueError("output directory must be outside the task workspace")
    if target.exists():
        raise FileExistsError(f"task workspace already exists: {target}")
    if (source / ".git").exists():
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks",
                        str(source), str(target)], check=True)
    else:
        shutil.copytree(source, target,
                        ignore=shutil.ignore_patterns("__pycache__", ".venv",
                                                       "node_modules", ".git"))
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        subprocess.run(["git", "add", "-A"], cwd=target, check=True)
        subprocess.run(["git", "-c", "user.name=HERMES",
                        "-c", "user.email=hermes@invalid.example",
                        "commit", "-qm", "task snapshot"], cwd=target, check=True)
    return target


def _portable_command(command: str) -> str:
    if os.name == "nt" and command.startswith("python3 "):
        return '"' + sys.executable + '"' + command[len("python3"):]
    return command


def execute_visible(repo_dir: Path, commands: list[str], timeout: int) -> dict:
    """Run only commands included in the task-visible manifest."""
    records = []
    for command in commands:
        run_command = _portable_command(command)
        try:
            result = subprocess.run(run_command, cwd=repo_dir, shell=True,
                                    capture_output=True, text=True,
                                    timeout=timeout)
            rc = result.returncode
            output = (result.stdout + result.stderr)[-12000:]
        except subprocess.TimeoutExpired as exc:
            rc = 124
            output = f"timed out after {timeout}s: {exc}"
        records.append({"command": run_command, "exit_code": rc, "output": output})
    combined = "\n\n".join(
        f"$ {rec['command']}\nexit={rec['exit_code']}\n{rec['output']}"
        for rec in records)
    failures = [rec for rec in records if rec["exit_code"] != 0]
    return {"shell": combined[:12000], "test": combined[:12000],
            "runtime": "\n".join(rec["output"][-3000:] for rec in failures)[:4000],
            "trace": "", "repro_output": "", "repro_rc": None,
            "repo_test_cmd": "; ".join(commands), "repo_test_rc": int(bool(failures)),
            "setup_rc": 0, "container_rc": 0, "command_results": records}


def _patch(repo_dir: Path) -> str:
    # Only artifacts explicitly marked by a primitive enter the diff. Baseline
    # or validation commands can create untracked caches and build byproducts.
    return subprocess.run(["git", "diff", "--binary", "--no-color"], cwd=repo_dir,
                          capture_output=True, text=True, check=True).stdout


def run_task(row: dict, output_root: Path, run_index: int, max_iterations: int,
             timeout: int) -> dict:
    label = f"{_safe_id(row['benchmark'])}_{_safe_id(row['task_id'])}_r{run_index}"
    out_dir = output_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = materialize(row, out_dir)
    start = time.time()
    allow_tests = row["benchmark"] == "devops_gym" and row.get("category") == "test_generation"
    p.GENERIC_COMPONENTS = True
    p.ALLOW_TEST_EDITS = allow_tests
    recorder = traj.Trajectory(label, repo=row["benchmark"],
                               problem_statement=row["issue"],
                               out_root=output_root / "trajectories",
                               config={"benchmark": row["benchmark"],
                                       "category": row.get("category"),
                                       "run_index": run_index,
                                       "model": p.THINKING_MODEL,
                                       "max_iterations": max_iterations,
                                       "visible_commands": row["visible_commands"]})
    traj.set_trajectory(recorder)
    rounds = 0
    accepted = False
    try:
        traj.set_phase("1.0_ISSUE_ANALYSIS")
        architecture = p.discover_architecture(repo_dir, row["issue"])
        traj.set_phase(p.TRIAGE_PHASE)
        relevant = p.locate_files(repo_dir, row["issue"], architecture)
        if not relevant:
            raise RuntimeError("activation found no repository components")
        if len(relevant) > p.MAX_RELEVANT_FILES and not p.ABLATE_ON_DEMAND:
            traj.set_phase("1.1_BUG_LOCALIZATION_RANK")
            choices = sorted(relevant, key=lambda r: r["file"])[:p.RANK_CANDIDATES]
            ranked = p.llm_json(
                f"Task: {row['issue'][:1000]}\nRank the task-relevant files "
                f"for modification or inspection: {[x['file'] for x in choices]}\n"
                f"Return JSON {{\"files\": [\"path\", ...]}} with at most "
                f"{p.MAX_RELEVANT_FILES} existing paths.",
                p.role_model("planner"), 1024, no_think=True).get("files", [])
            by_path = {r["file"]: r for r in choices}
            relevant = [by_path[name] for name in ranked
                        if isinstance(name, str) and name in by_path]
            if not relevant:
                relevant = choices[:p.MAX_RELEVANT_FILES]
        pool = {item["file"]: item.get("reason", "") for item in relevant}
        traj.set_phase("1.2_TASK_DECOMPOSITION")
        tasks = p.decompose_tasks(repo_dir, row["issue"], architecture, relevant)
        traj.set_phase("1.3_DEPENDENCY_ANALYSIS")
        deps = p.analyze_dependencies(repo_dir, row["issue"], architecture, tasks)
        for name in deps["missing_files"]:
            if name not in {task["file"] for task in tasks}:
                tasks.append({"file": name, "role": "Other",
                              "task": "dependency implicated by activation",
                              "changes": True})
        traj.set_phase("1.4_EDIT_PLANNING")
        plan_obj = p.plan_edits(repo_dir, row["issue"], architecture, tasks, deps)
        baseline = (execute_visible(repo_dir, row["visible_commands"], timeout)
                    if not p.ABLATE_EXECUTION_FEEDBACK else
                    {"repo_test_rc": None, "command_results": []})
        traj.log("baseline", command_results=baseline["command_results"])
        feedback = ""
        report = None
        for rounds in range(1, max_iterations + 1):
            scope = [{"file": task["file"], "role": task.get("role", "Other")}
                     for task in tasks]
            roles = {task["file"]: task.get("role", "Other") for task in tasks}
            plan = p.render_plan(plan_obj)
            traj.set_phase(f"2_INTER_FILE_COMMUNICATION (round {rounds})")
            if p.ABLATE_COMMUNICATION:
                intents, adjusted = {}, set()
            else:
                intents, adjusted = p.negotiate(
                    repo_dir, row["issue"], architecture, plan, scope,
                    feedback, rounds, deps, roles)
            edits = plan_obj.get("edits", {})
            writers = [item["file"] for item in scope
                       if item["file"] in edits or item["file"] in adjusted]
            traj.set_phase(f"2.5_SELF_MODIFY (round {rounds})")
            for filename in writers:
                if (repo_dir / filename).is_file():
                    objective = (edits.get(filename) or {}).get("change", "")
                    changed = edit_component(
                        repo_dir, filename, row["issue"], objective,
                        intents.get(filename, ""), feedback, rounds,
                        architecture=architecture, plan=plan,
                        role=roles.get(filename, "Other"),
                        must_not_break=(edits.get(filename) or {}).get(
                            "must_not_break", ""))
                    persistent = [name for name in changed
                                  if (repo_dir / name).is_file()]
                    if persistent:
                        subprocess.run(["git", "add", "-N", "--", *persistent],
                                       cwd=repo_dir, check=True)
            patch = _patch(repo_dir)
            traj.set_phase(f"3_EXECUTION (round {rounds})")
            obs = (execute_visible(repo_dir, row["visible_commands"], timeout)
                   if not p.ABLATE_EXECUTION_FEEDBACK else
                   {"shell": "", "test": "", "runtime": "", "trace": "",
                    "repro_output": "", "repro_rc": None,
                    "repo_test_cmd": None, "repo_test_rc": None,
                    "setup_rc": None, "container_rc": None,
                    "command_results": []})
            traj.log("verify", round=rounds, command_results=obs["command_results"])
            traj.set_phase(f"4_CRITIC (round {rounds})")
            if p.ABLATE_CRITIC:
                report = p.no_critic_report(
                    obs, patch, rounds, baseline["repo_test_rc"])
            else:
                report = p.critique(row["issue"], plan_obj, obs, patch, writers,
                                    rounds, baseline_repo_test_rc=baseline["repo_test_rc"],
                                    prev_report=report, inactive_candidates=pool)
            if report["status"] == "PASS":
                accepted = True
                break
            if rounds < max_iterations:
                feedback = (p.render_critic_report(report)
                            if not p.ABLATE_CRITIC else obs["test"])
                traj.set_phase(f"1.3_ACTIVE_SET_REVISION (round {rounds})")
                if not p.ABLATE_ON_DEMAND and not p.ABLATE_CRITIC:
                    tasks = p.revise_active_set(repo_dir, tasks, report, rounds,
                                                localized_pool=pool)
                traj.set_phase(f"1.4_EDIT_PLANNING (replan after round {rounds})")
                plan_obj = p.plan_edits(repo_dir, row["issue"], architecture,
                                        tasks, deps,
                                        critic=None if p.ABLATE_CRITIC else report,
                                        round_idx=rounds)
        patch = _patch(repo_dir)
        result = {"task_id": row["task_id"], "benchmark": row["benchmark"],
                  "category": row.get("category"), "run_index": run_index,
                  "critic_accepted": accepted, "rounds": rounds,
                  "patch": str(out_dir / "patch.diff"),
                  "workspace": str(repo_dir),
                  "trajectory": str(recorder.out_dir),
                  "official_score": None}
        (out_dir / "patch.diff").write_text(patch)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        recorder.close(resolved=None, rounds=rounds,
                       elapsed_s=time.time() - start, final_patch=patch)
        return result
    except BaseException:
        recorder.close(resolved=None, rounds=rounds,
                       elapsed_s=time.time() - start, final_patch=_patch(repo_dir))
        raise
    finally:
        traj.set_trajectory(None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS))
    parser.add_argument("--paper-protocol", action="store_true",
                        help="require the task count and repeat count in the paper")
    parser.add_argument("--no-resume", action="store_true",
                        help="error if a task result already exists")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--run-index", type=int,
                        help="execute only this repeat index (for Slurm arrays)")
    parser.add_argument("--no-runs-file", action="store_true",
                        help="write per-task result.json only, avoiding concurrent appends")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--command-timeout", type=int, default=900)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--ollama", metavar="MODEL")
    model.add_argument("--vllm", metavar="MODEL")
    model.add_argument("--model", metavar="LITELLM_ID")
    parser.add_argument("--ollama-url")
    parser.add_argument("--vllm-url")
    parser.add_argument("--planner-model")
    parser.add_argument("--critic-model")
    parser.add_argument("--primitive-model")
    parser.add_argument("--ablate-communication", action="store_true")
    parser.add_argument("--ablate-on-demand", action="store_true")
    parser.add_argument("--ablate-execution-feedback", action="store_true")
    parser.add_argument("--ablate-critic", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.max_iterations < 1:
        parser.error("--repeat and --max-iterations must be positive")
    if args.run_index is not None and not 0 <= args.run_index < args.repeat:
        parser.error("--run-index must be between 0 and --repeat-1")
    if args.ollama:
        p.use_ollama(args.ollama, url=args.ollama_url)
    elif args.vllm:
        p.use_vllm(args.vllm, url=args.vllm_url)
    else:
        p.THINKING_MODEL = args.model
        p.TRIAGE_MODEL = args.model
    p.PLANNER_MODEL = args.planner_model
    p.PRIMITIVE_MODEL = args.primitive_model
    p.CRITIC_MODEL = args.critic_model
    p.ABLATE_COMMUNICATION = args.ablate_communication
    p.ABLATE_ON_DEMAND = args.ablate_on_demand
    p.ABLATE_EXECUTION_FEEDBACK = args.ablate_execution_feedback
    p.ABLATE_CRITIC = args.ablate_critic
    rows = load_manifest(args.manifest)
    if args.benchmark:
        rows = [row for row in rows if row["benchmark"] == args.benchmark]
    if args.paper_protocol:
        if not args.benchmark:
            parser.error("--paper-protocol requires --benchmark")
        count = len(rows)
        expected = {"swe_refactor": 20, "terminal_bench": 66}.get(args.benchmark)
        if expected and count != expected:
            parser.error(f"paper protocol needs {expected} tasks; found {count}")
        if args.benchmark == "terminal_bench" and args.repeat != 5:
            parser.error("paper protocol needs five runs per Terminal-Bench task")
        if args.benchmark == "devops_gym" and {
                row.get("category") for row in rows} != CATEGORIES:
            parser.error("paper protocol needs all four DevOps-Gym categories")
    if args.task_id:
        wanted = set(args.task_id)
        rows = [row for row in rows if row["task_id"] in wanted]
        if wanted - {row["task_id"] for row in rows}:
            parser.error("task ID not found in manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "runs.jsonl"
    handle = None if args.no_runs_file else output_file.open("a")
    try:
        for row in rows:
            indexes = ([args.run_index] if args.run_index is not None
                       else range(args.repeat))
            for run_index in indexes:
                label = (f"{_safe_id(row['benchmark'])}_"
                         f"{_safe_id(row['task_id'])}_r{run_index}")
                if (args.output / label / "result.json").exists():
                    if args.no_resume:
                        parser.error(f"result already exists: {label}")
                    print(f"skipping completed {label}")
                    continue
                result = run_task(row, args.output, run_index,
                                  args.max_iterations, args.command_timeout)
                if handle:
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
                print(json.dumps(result))
    finally:
        if handle:
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
