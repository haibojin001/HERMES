# SWE Refactor Bench

The paper scores 20 whole-repository migration tasks using the benchmark's official composite score. Fetch the pinned source and run Harbor with HERMES as the agent:

```bash
python benchmarks/fetch.py swe_refactor
pip install harbor
HERMES_BACKEND=ollama HERMES_MODEL=qwen3:8b \
  experiments/swe_refactor/run_official.sh all
python scripts/report_official.py --benchmark swe_refactor \
  --results hermes_data/official_runs/swe_refactor \
  --output hermes_data/reports/refactor.json
```

`all` can be replaced by a task directory name for a smoke run. `HERMES_ATTEMPTS=3` runs three independent complete evaluations, as in the manuscript's variation study; pass `--repeats 3` to the reporter. The scorer rejects missing tasks and invalid verifier verdicts. It reads `reward` from the official Harbor result, after HERMES ends. `critic_status` is diagnostic only.

Harbor provides `/workspace/repo` as the editable artifact and runs its audit, behavioural and verification stages in separate evaluator environments. Reproducing a score requires those images and the scorer's model credentials. A filesystem-only manifest cannot replace that protocol. `run.sh MANIFEST OUTPUT` is available for task-visible snapshot development; it writes patches but no official scores.
