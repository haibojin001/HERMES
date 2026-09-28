# Terminal-Bench 4.0

The locked `v4.0.0` checkout contains 66 tasks. The official runner starts each task environment with Harbor and performs five **independent full runs** so the mean and standard deviation are computed across complete 66-task evaluations:

```bash
python benchmarks/fetch.py terminal_bench
pip install harbor
HERMES_BACKEND=ollama HERMES_MODEL=qwen3:8b \
  experiments/terminal_bench_4/run_official.sh all
python scripts/report_official.py --benchmark terminal_bench \
  --results hermes_data/official_runs/terminal_bench \
  --output hermes_data/reports/terminal.json
```

Pass a task directory name instead of `all` for a smoke run. The reporter requires 66 graded tasks in each of the five jobs and treats an official `reward` of exactly 1 as resolved; partial rewards are preserved in its per-trial score output. The benchmark's verifier runs only after the agent. Do not feed its results into the Critic.

The adapter activates persistent task artifacts as Dev-Primitives: repository files under `/workspace/repo` and, for other tasks, independently editable scripts, configuration files, service definitions, build files and required output files. Task-visible shell and service commands provide observations for the Critic. The alternative `run.sh MANIFEST OUTPUT` runs isolated filesystem snapshots for debugging and does not run Harbor or reproduce service state.
