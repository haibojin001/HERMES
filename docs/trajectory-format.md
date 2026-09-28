# Trajectory records

Each run writes `trajectory.jsonl`, `trajectory.md`, `summary.json`, and `final_patch.diff` in its run directory. Official Harbor and legacy `tb` results are separate files written after the agent returns. `critic_status` is an internal diagnosis, not the official score.

Every JSONL record contains a monotonic `seq`, elapsed time `t`, and `kind`. Repository runs include `architecture`, `triage`, `task_decomposition`, `dependency_analysis`, `plan`, `negotiate`, `artifact_edit`, `verify`, `critic`, `active_set`, `replan`, `patch`, and `run_end` records. The terminal adapters additionally log addressed `message` and `terminal_command` records.

`summary.json` records `llm_calls`, `triage_llm_calls`, `prompt_tokens`, `completion_tokens`, `usage_missing_calls`, and `model_usage` by backbone. Localization calls are included in these totals. Their full prompts and responses are omitted from the JSONL by default to keep files manageable; set `trajectory.LOG_TRIAGE_CALLS=True` when those details are needed. A missing backend usage value increases `usage_missing_calls` and makes token totals incomplete.

The manuscript's revision budget is `B=max_iterations-1`; `rounds` in the trajectory is the number actually used. Held-out grader results never enter `critic` or `replan` records. `held_out_evaluation` appears in SWE-bench trajectories only after the edit loop; the Harbor and legacy graders store their result under the harness's official job directory.
