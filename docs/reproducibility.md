# Reproducibility status

This code provides an anonymous HERMES source tree, pinned upstream benchmark checkouts, task-environment adapters, independent output paths, and post-run score readers.

## What is wired end to end

- SWE-bench Verified: solver, task-visible execution, official held-out evaluation, five ablation configurations and model/budget sweeps.
- SWE Refactor Bench: Harbor adapter edits `/workspace/repo`; Harbor later runs the official composite scorer. The reporter requires all 20 valid trial rewards.
- Terminal-Bench 4.0: Harbor adapter acts in the official terminal environment; five separate complete jobs allow run-level mean and standard deviation over 66 tasks. The reporter uses reward 1 as resolution and retains any partial raw reward.
- DevOps-Gym: the pinned legacy Terminal-Bench runner calls a HERMES agent before its verifier. The four category jobs and reporter require every official `is_resolved` verdict.
- The local Qwen3-8B Ollama route uses 32K context, temperature 0.6, top-p 0.95 and top-k 20. The trajectory counts localization calls and token usage when the backend reports usage.

## Limits that affect a paper comparison

1. **Model conditions.** Native OpenAI and Claude API interfaces and a configurable reasoning-effort parameter are available, but exact model IDs, account access, effort support and inference prices must be supplied for a named manuscript row. The Ollama commands run the open-weight condition only. The legacy `--vllm` path defaults to deterministic decoding, so it is a distinct condition from the paper's Ollama Qwen path.
2. **Environment and grading.** Official task images, Docker runtime, scorer model access, and DevOps-Gym LFS assets are external prerequisites. A JSONL manifest run writes a patch and Critic result but cannot produce a benchmark number without the official runtime and verifier.
3. **Analysis coverage.** The central-editor controls, full appendix multi-benchmark ablations, benchmark-specific reference-component labels, and per-model cost pricing are not completed. The released script set must not be described as reproducing those tables.
4. **Token gaps.** Localization usage is counted. A backend response without token usage is marked in `summary.json`; it does not provide a complete cost. Single-rate `--price` is suppressed when usage is missing or multiple models were used.
5. **Repository communication timing.** SWE-bench Verified and SWE Refactor Bench still use a separate intent negotiation before local edits. The Terminal-Bench and DevOps-Gym adapters send addressed messages during local editing. The appendix describes interleaved communication for all tasks, so the repository adapters should not be treated as an exact implementation of that timing.

## Isolation

SWE-bench `FAIL_TO_PASS` and `PASS_TO_PASS` reach only the post-solve grader. Harbor provides separate verifier environments for the other two Harbor benchmarks. The DevOps-Gym `tb` verifier runs after `perform_task` returns. The agent instructions require task-visible checks only; neither a hidden verifier result nor a grader reward is used to revise an answer.

## Verification possible here

Unit tests and syntax checks run without a model or containers. They verify accounting, data isolation, score aggregation and error handling, but cannot establish official benchmark scores. Run a single pinned task in each official harness before scheduling a full experiment and inspect its recorded `result.json` and trajectory.
