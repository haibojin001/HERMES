# SWE-bench Verified

The built-in SWE-bench loader and held-out evaluator cover all 500 Verified tasks.
The solver sees the issue, repository snapshot and task-visible tests. It only
loads `FAIL_TO_PASS`, `PASS_TO_PASS` and the benchmark test patch after the
activation-modification-execution-diagnosis loop terminates.

```bash
python scripts/prepare_dataset.py
python scripts/build_env_images.py
BACKEND_ARGS="--ollama qwen3:8b --ollama-num-ctx 32768" \
  scripts/run_swebench.sh
```

`MAX_ITER=4` is the paper's revision budget `B=3`.
The environment images must be available in the selected container runtime.
An unavailable image now produces an empty, failed prediction before any LLM
calls; it is never treated as a completed HERMES trajectory.

`scripts/aggregate_results.py` reads the prediction file, final evaluation
results and trajectories. Run one backbone per configuration directory. These
scripts do not supply the baseline harness results in the paper; those require
separate matched baseline runs or documented leaderboard provenance.
