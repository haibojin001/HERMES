# Experiment map

The four primary benchmarks have separate environments and graders. `--max-iterations 4` corresponds to the paper's `B=3`; hold the model, effort, task snapshot and environment fixed when comparing rows. The official runners retain benchmark-verifier output separately from the agent trajectory.

| Manuscript result | Launcher | Scored output |
|---|---|---|
| SWE-bench Verified resolution | `scripts/run_swebench.sh` | `hermes.evaluate` |
| SWE Refactor Bench composite, 20 tasks | `experiments/swe_refactor/run_official.sh all` | Harbor `reward` averaged by `scripts/report_official.py` |
| Terminal-Bench 4.0 resolution, 66 tasks x 5 runs | `experiments/terminal_bench_4/run_official.sh all` | Five Harbor jobs; full reward 1 means resolved |
| DevOps-Gym four category success rates | `experiments/devops_gym/run_official.sh CATEGORY` | legacy `tb` `is_resolved`; unweighted category mean |

Fetch exact upstream versions with `benchmarks/fetch.py`; hashes are in `benchmarks/lock.json`. The three newer benchmarks need their own official task images. The DevOps-Gym checkout also needs Git LFS assets. `run_official.sh` starts official graders **after** the agent returns; `run.sh` in each directory is a development runner over task-visible JSONL manifests and is not a scored substitute.

## SWE-bench analyses

- `scripts/run_ablations.sh`: the complete system and four mechanism removals.
- `scripts/run_replanning_budget.sh`: `B=0,1,2,3,5`, mapped to `--max-iterations 1,2,3,4,6`.
- `scripts/run_backbone_sweep.sh --roles`: change Planner, Dev-Primitive or Critic model separately.
- `scripts/aggregate_results.py`: grade and trajectory metrics. `--price` prints a one-rate estimate only when every model call has token usage and exactly one model was used. Mixed-model sweeps need per-model pricing to compute cost.

The appendix's central-editor and compute-matched controls, reference-component selection recall, and matched frontier-model effort sweeps do not have complete launchers in this tree. Running the official baseline launchers under Ollama Qwen3-8B is a new condition; it is not a reproduction of those frontier-model rows.

## Completeness rules

`report_official.py` rejects missing or ungraded tasks and prints the denominators it actually observed. For Terminal-Bench it computes each full run's 66-task rate and then reports the mean and sample standard deviation of five rates. For DevOps-Gym it averages category rates without weighting by category size.

Task-visible execution output may be sent to the Critic. Official grader output must never be inserted into a task manifest, prompt, or revision round. All official score files are produced after agent termination.
