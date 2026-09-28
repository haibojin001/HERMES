# HERMES: anonymous research code

HERMES activates task-relevant Dev-Primitives, edits persistent artifacts, executes task-visible checks, and uses Critic feedback for up to `B` revision rounds. `--max-iterations 4` means the paper default `B=3`.

| Experiment | Entry point | Official evaluator |
|---|---|---|
| SWE-bench Verified | `scripts/run_swebench.sh` | `hermes.evaluate` |
| SWE Refactor Bench | `experiments/swe_refactor/run_official.sh` | Harbor and the benchmark's three-stage scorer |
| Terminal-Bench 4.0 | `experiments/terminal_bench_4/run_official.sh` | Harbor task verifier, five independent runs |
| DevOps-Gym | `experiments/devops_gym/run_official.sh` for each of four categories | legacy Terminal-Bench `tb` verifier |

The `run.sh` files under the last three experiment directories are for benchmark-provided task-visible **snapshot manifests**. They write patches and trajectories but do not themselves run official graders. Use `run_official.sh` for scored experiments.

## Install and fetch

Use Python 3.12 or newer for the official Harbor and legacy `tb` runners, a working container runtime, and a reachable model endpoint. From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install 'harbor==0.23.0'
python benchmarks/fetch.py swe_refactor terminal_bench devops_gym terminal_bench_legacy
pip install -e hermes_data/benchmarks/terminal_bench_legacy
```

The pinned public benchmark commits are in `benchmarks/lock.json`. DevOps-Gym uses Git LFS; add `--with-assets` to the fetch command before running image-backed tasks. Fetching the benchmark repositories does not install their Docker images. The official runners verify commit hashes and expect the checkouts under `hermes_data/benchmarks` unless `HERMES_BENCHMARKS` is set.

For hosted OpenAI or Claude models, install `pip install -e '.[hosted]'` and set the corresponding API key in the environment. Run `python scripts/check_models.py --backend hosted --model PROVIDER/MODEL_ID` before starting the harness. Model routing and effort options are in [the backbone guide](configs/backbones.md).

## Official examples

```bash
export HERMES_BACKEND=ollama HERMES_MODEL=qwen3:8b
experiments/swe_refactor/run_official.sh all
experiments/terminal_bench_4/run_official.sh all
for category in build monitor issue_resolving test_generation; do
  experiments/devops_gym/run_official.sh "$category"
done
python scripts/report_official.py --benchmark swe_refactor \
  --results hermes_data/official_runs/swe_refactor
python scripts/report_official.py --benchmark terminal_bench \
  --results hermes_data/official_runs/terminal_bench
python scripts/report_official.py --benchmark devops_gym \
  --results hermes_data/official_runs/devops_gym
```

Run the SWE-bench solver only where its environment images are available:

```bash
python scripts/prepare_dataset.py
python scripts/build_env_images.py
BACKEND_ARGS='--ollama qwen3:8b --ollama-num-ctx 32768' scripts/run_swebench.sh
```

The scored reports require one official verifier result for every expected task and run. The Critic's own PASS judgement is never used as a benchmark score. Official output, fetched benchmarks, workspaces, and trajectories belong under ignored `hermes_data/`.

## Model and analysis settings

The manuscript's open-weight condition uses Ollama Qwen3-8B, 32K context, temperature 0.6, top-p 0.95, and top-k 20. The Ollama code path sets these values. Frontier-model table rows require the exact available API IDs and effort settings; use `HERMES_REASONING_EFFORT` for official Harbor and `tb` runs or `--reasoning-effort` for SWE-bench. `configs/backbones.md` describes model slots. SWE-bench ablations, role sweeps, and revision-budget sweep are in `scripts/`; the other benchmark adapters expose the core loop and official scoring but do not claim to reproduce every appendix control.

## Checks

```bash
python -B -m unittest discover -s tests -q
python scripts/check_anonymity.py
```
