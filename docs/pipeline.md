# HERMES solve loop

For a repository task, `pipeline.discover_architecture` reads the issue and tree layout. `locate_files` activates task-relevant candidate components. `decompose_tasks`, `analyze_dependencies`, and `plan_edits` assign component-local objectives. The selected Dev-Primitives exchange addressed requirements through `negotiate`; `artifacts.edit_component` applies each owner's structured edit and may create new artifacts.

The solver then runs task-visible commands in the task environment. The resulting shell, test and runtime output goes to `pipeline.critique`. A deterministic failing reproduction, test, setup or container signal cannot be turned into PASS by the Critic. On FAIL, `revise_active_set` may activate an existing or newly created component, and `plan_edits` rewrites objectives. The loop stops on PASS or at `B+1` total edit rounds.

SWE-bench Verified executes its held-out tests only after that loop. SWE Refactor Bench and Terminal-Bench run in Harbor; DevOps-Gym uses legacy Terminal-Bench. Those harnesses start the official verifier after the HERMES agent returns. The snapshot manifest runner deliberately writes an ungraded patch, so it is useful for development but not a substitute for an official benchmark trial.

For Terminal-Bench and DevOps-Gym, an activated primitive owns a persistent task artifact, including a shell script, configuration file, service definition, build file or new report file. The adapters stage each selected artifact, apply its owner's edit, and sync the result to the task container. Commands used to inspect or interact with services produce execution observations for the Critic. A command, process, terminal output or transient system state is not a Dev-Primitive.
