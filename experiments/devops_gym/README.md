# DevOps-Gym

The paper reports build and configuration, monitoring, issue resolving, and test generation success rates plus their **unweighted** mean. This benchmark uses legacy Terminal-Bench `tb`, not Harbor v4. Fetch both pinned checkouts and the required Git LFS task assets, then install `tb` from its checkout:

```bash
python benchmarks/fetch.py devops_gym terminal_bench_legacy --with-assets
pip install -e hermes_data/benchmarks/terminal_bench_legacy
for category in build monitor issue_resolving test_generation; do
  HERMES_BACKEND=ollama HERMES_MODEL=qwen3:8b \
    experiments/devops_gym/run_official.sh "$category"
done
python scripts/report_official.py --benchmark devops_gym \
  --results hermes_data/official_runs/devops_gym \
  --output hermes_data/reports/devops.json
```

The reporter reads each category's official `results.json` and requires every task to have a boolean `is_resolved` verdict. Monitoring tasks can require sustained observation and a newly created report file. The report file is a persistent Dev-Primitive; monitoring commands and their output are execution observations routed to the Critic. The hidden benchmark tests are run by `tb` only after the agent returns.

`run.sh MANIFEST OUTPUT` is a filesystem snapshot development path and does not run the official verifier.
