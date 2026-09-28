# Manuscript mechanisms and code

The implementations below are entry points, not assertions that a published score is reproduced. The official grader runs after the agent finishes.

| Manuscript mechanism | Code path |
|---|---|
| Issue analysis and activated set `A` | `pipeline.discover_architecture`, `locate_files`, `decompose_tasks`, `analyze_dependencies`, `plan_edits` for repository tasks; task-component planning in `harbor_agent` and `legacy_tb_agent` for interactive tasks |
| Artifact-owned Dev-Primitive | `artifacts.edit_component` and `apply_artifact_edit`; all task adapters use this shared structured editor. Its prompt and response include the manuscript's `component`, `action`, `summary`, `changes`, `updated_artifact`, `messages`, and `validation_notes` fields. The runtime also supports exact replacement anchors for large files. |
| Addressed inter-primitive communication `C_i` | `pipeline.negotiate` for repository tasks; `artifacts.edit_component` emits addressed messages during editing in the terminal adapters |
| Task-visible execution observation `o` | `ContainerRunner.run_task_visible`, `manifest_runner.execute_visible`, Harbor `environment.exec`, or legacy `TmuxSession` |
| Critic verdict `v` and feedback `phi=(e,c,u)` | `pipeline.critique`; a failing task-visible test or reproduction forces FAIL |
| Revision of objectives and active set | `pipeline.revise_active_set` followed by `plan_edits`; interactive adapters reselect components from Critic feedback |
| Held-out grading | `hermes.evaluate`, Harbor verifier, or legacy `tb` verifier, invoked after the solve loop |
| Revision budget `B=3` | `--max-iterations 4`; first edit round plus three possible revisions |

A new artifact created by `apply_artifact_edit` persists across rounds. On SWE-bench, the editor marks the created path as intent-to-add so it enters the final patch. In Harbor, the adapter uploads the created file into the agent environment before execution. A later Critic can activate it through the revised component pool.

## Experiment rows

SWE-bench Verified has launchers for the full system, four mechanism removals, `B=0,1,2,3,5`, and role-model sweeps. The three other benchmarks have official task and scorer entry points plus a snapshot development runner. Their full appendix controls, central-editor baseline, and all matched frontier-model effort settings are not in the current code. See `docs/experiments.md` for launch commands and `docs/reproducibility.md` for the deviations that matter to a paper comparison.

In Terminal-Bench and DevOps-Gym, shell scripts, configuration files, service definitions, build files and required output files are eligible Dev-Primitives. The terminal adapters select their paths, create or modify them through the shared editor, and route task-visible command output to the Critic as an observation. Runtime processes and terminal actions do not appear in the activated component set.
