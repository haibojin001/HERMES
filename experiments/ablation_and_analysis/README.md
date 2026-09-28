# Ablations and analysis

`scripts/run_ablations.sh` implements the four mechanism removals for SWE-bench Verified. `scripts/run_replanning_budget.sh` runs `B=0,1,2,3,5`, and `scripts/run_backbone_sweep.sh --roles` varies one model role at a time. `--max-iterations` is `B+1`.

`manifest_runner.py` accepts `--ablate-communication`, `--ablate-on-demand`, `--ablate-execution-feedback`, and `--ablate-critic` for task-visible filesystem snapshots. That runner does not call an official verifier, so these flags alone cannot produce the non-SWE-bench ablation table. The Harbor and legacy official adapters do not yet expose all four ablations.

The appendix's central-editor, compute-matched, and reference-component selection analyses need separate policies or benchmark-specific labels. No output for those rows is fabricated here. Use the same task snapshot, model, effort, revision budget and scorer for each implemented comparison and keep rows in separate output directories.
