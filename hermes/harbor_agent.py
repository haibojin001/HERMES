"""Harbor adapter for official SWE Refactor Bench and Terminal-Bench tasks.

Harbor owns environment creation and the hidden verifier. This agent reads only
the agent environment, edits persistent task artifacts, and uses task-visible
commands for execution feedback. It never reads the verifier environment.
"""

import asyncio
import json
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage

from hermes import pipeline as p
from hermes import trajectory as traj
from hermes.artifacts import edit_component
from hermes.terminal_components import components, selection_prompt


def _remote_path(workspace: str, rel: str) -> str:
    if (not rel or Path(rel).is_absolute() or ".." in Path(rel).parts
            or ".git" in Path(rel).parts):
        raise ValueError(f"invalid component path: {rel}")
    return workspace.rstrip("/") + "/" + rel


class HermesHarborAgent(BaseAgent):
    """HERMES file primitives running against a Harbor-provided task sandbox."""

    def __init__(self, *args, workspace="auto", backend="hosted",
                 ollama_url=None, vllm_url=None, max_iterations=4, command_timeout=900,
                 terminal_mode=False, planner_model=None,
                 primitive_model=None, critic_model=None, triage_model=None,
                 reasoning_effort=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.workspace = workspace
        self.backend = backend
        self.ollama_url = ollama_url
        self.vllm_url = vllm_url
        self.max_iterations = int(max_iterations)
        self.command_timeout = int(command_timeout)
        self.terminal_mode = str(terminal_mode).lower()
        self.planner_model = planner_model
        self.primitive_model = primitive_model
        self.critic_model = critic_model
        self.triage_model = triage_model
        self.reasoning_effort = reasoning_effort
        if self.terminal_mode not in ("true", "false", "auto"):
            raise ValueError("terminal_mode must be true, false or auto")
        if self.max_iterations < 1 or self.command_timeout < 1:
            raise ValueError("iteration budget and command timeout must be positive")

    @staticmethod
    def name() -> str:
        return "hermes"

    def version(self) -> str:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        # All reasoning runs in the Harbor host process. The task image is
        # provided by the benchmark, and no network installer runs in it.
        return None

    async def _workspace(self, instruction: str,
                         environment: BaseEnvironment) -> str:
        if self.workspace != "auto":
            if not await environment.is_dir(self.workspace):
                raise FileNotFoundError(f"task workspace missing: {self.workspace}")
            return self.workspace
        candidates = ["/workspace/repo"]
        candidates += re.findall(r"/(?:workspace|app)/[A-Za-z0-9_./-]+",
                                 instruction)
        result = await environment.exec("pwd", timeout_sec=10)
        if result.return_code == 0 and result.stdout:
            candidates.append(result.stdout.strip())
        candidates += ["/app", "/workspace"]
        for path in candidates:
            if await environment.is_dir(path):
                return path
        raise FileNotFoundError(
            "could not locate task workspace; pass -ak workspace=/task/path")

    async def _download(self, environment: BaseEnvironment, workspace: str,
                        local: Path) -> None:
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        await environment.download_dir(workspace, local)

    async def _sync(self, environment: BaseEnvironment, workspace: str,
                    local: Path, changed: list[str]) -> None:
        for rel in changed:
            remote = _remote_path(workspace, rel)
            path = local / rel
            if path.is_file():
                parent = remote.rsplit("/", 1)[0]
                result = await environment.exec(
                    f"mkdir -p {shlex.quote(parent)}", timeout_sec=20)
                if result.return_code != 0:
                    raise RuntimeError(f"cannot create remote parent: {parent}")
                await environment.upload_file(path, remote)
            else:
                result = await environment.exec(
                    f"rm -f {shlex.quote(remote)}", timeout_sec=20)
                if result.return_code != 0:
                    raise RuntimeError(f"cannot delete remote component: {remote}")

    async def _observe(self, environment: BaseEnvironment, workspace: str,
                       commands: list[str]) -> dict:
        items = []
        for cmd in commands:
            result = await environment.exec(
                command=cmd, cwd=workspace, timeout_sec=self.command_timeout)
            items.append({"command": cmd, "rc": result.return_code,
                          "output": ((result.stdout or "") + (result.stderr or ""))[-12000:]})
        text = "\n\n".join(
            f"$ {item['command']}\nexit={item['rc']}\n{item['output']}"
            for item in items)
        return {"shell": text[:12000], "test": text[:12000], "runtime": "",
                "trace": "", "repro_output": "", "repro_rc": None,
                "repo_test_cmd": "; ".join(commands),
                "repo_test_rc": int(any(item["rc"] != 0 for item in items)),
                "setup_rc": 0, "container_rc": 0}

    def _configure_model(self) -> None:
        model = self.model_name or "qwen3:8b"
        if self.backend == "ollama":
            p.use_ollama(model, url=self.ollama_url)
        elif self.backend == "vllm":
            p.use_vllm(model, url=self.vllm_url)
        elif self.backend == "hosted":
            p.use_hosted(model, reasoning_effort=self.reasoning_effort)
        else:
            raise ValueError(f"unknown model backend: {self.backend}")
        if self.ollama_url:
            p.OLLAMA_URL = self.ollama_url
        if self.vllm_url:
            p.VLLM_URL = self.vllm_url
        if self.backend != "hosted" and self.reasoning_effort:
            p.HOSTED_REASONING_EFFORT = self.reasoning_effort
        p.PLANNER_MODEL = self.planner_model
        p.PRIMITIVE_MODEL = self.primitive_model
        p.CRITIC_MODEL = self.critic_model
        if self.triage_model:
            p.TRIAGE_MODEL = self.triage_model
        p.GENERIC_COMPONENTS = True
        p.ALLOW_TEST_EDITS = False

    async def _run_terminal_task(self, instruction: str,
                                 environment: BaseEnvironment,
                                 context: AgentContext) -> None:
        """Solve terminal tasks with persistent artifact owners and shell feedback."""
        workspace = await self._workspace(instruction, environment)
        start = time.time()
        recorder = traj.Trajectory(
            self.logs_dir.name, repo="terminal_bench",
            problem_statement=instruction, out_root=self.logs_dir,
            config={"workspace": workspace, "model": self.model_name,
                    "backend": self.backend, "max_iterations": self.max_iterations,
                    "primitive_mode": "persistent_artifact"})
        traj.set_trajectory(recorder)
        rounds = 0
        report = None
        try:
            traj.set_phase("1_TASK_ANALYSIS")
            initial = await environment.exec(
                "pwd; ls -la | head -40", cwd=workspace, timeout_sec=30)
            prompt = selection_prompt(instruction, initial.stdout or "")
            selected = await asyncio.to_thread(
                p.llm_json, prompt, p.role_model("planner"), 2048)
            active = components(selected.get("components"), workspace)
            if not active:
                retry = await asyncio.to_thread(
                    p.llm_json,
                    selection_prompt(instruction, initial.stdout or "",
                                     "Select a concrete persistent file path."),
                    p.role_model("planner"), 2048)
                active = components(retry.get("components"), workspace)
            if not active:
                raise RuntimeError("no persistent task artifact selected")
            feedback = ""
            registered = set()
            communication = {}
            with tempfile.TemporaryDirectory(prefix="hermes-terminal-") as tmp:
                local = Path(tmp)
                for rounds in range(1, self.max_iterations + 1):
                    command_prompt = (
                        f"Task: {instruction}\nArtifacts: {active}\n"
                        f"Critic feedback: {feedback[-3000:]}\n"
                        "Choose up to three task-visible commands to observe "
                        "the environment before editing and up to three commands "
                        "to check behavior after editing. Commands may interact "
                        "with services. Persistent artifact edits are handled "
                        "by their file owners. Never call hidden tests or the "
                        "benchmark verifier. Return JSON "
                        '{"observe":["command"],"check":["command"]}.')
                    command_plan = await asyncio.to_thread(
                        p.llm_json, command_prompt, p.role_model("planner"), 2048)
                    if not isinstance(command_plan, dict):
                        command_plan = {}
                    before = [c for c in (command_plan.get("observe") or [])[:3]
                              if isinstance(c, str) and c.strip()]
                    checks = [c for c in (command_plan.get("check") or [])[:3]
                              if isinstance(c, str) and c.strip()]
                    traj.set_phase(f"3_EXECUTION (round {rounds}, observation)")
                    pre_obs = await self._observe(environment, workspace, before)
                    traj.log("terminal_observation", round=rounds,
                             commands=before, output=pre_obs["shell"])
                    changed = []
                    inbox = {item["name"]: list(
                        communication.get(item["name"], [])) for item in active}
                    queue = list(active)
                    calls = {item["name"]: 0 for item in active}
                    while queue:
                        item = queue.pop(0)
                        owner = item["owner"]
                        remote = item["name"]
                        staged = local / owner
                        if calls[remote] >= 2:
                            continue
                        if calls[remote] == 0:
                            if await environment.is_dir(remote):
                                traj.log("artifact_edit", round=rounds, file=remote,
                                         error="selected path is a directory")
                                continue
                            exists = await environment.is_file(remote)
                            if exists:
                                staged.parent.mkdir(parents=True, exist_ok=True)
                                await environment.download_file(remote, staged)
                            elif staged.exists():
                                staged.unlink()
                        else:
                            exists = staged.is_file()
                        calls[remote] += 1
                        traj.set_phase(f"2_INTER_PRIMITIVE_COLLABORATION (round {rounds})")
                        outgoing = []
                        edited = await asyncio.to_thread(
                            edit_component, local, owner, instruction,
                            item["objective"], "\n".join(inbox[remote])[-4000:],
                            feedback + "\nTask-visible observation:\n"
                            + pre_obs["shell"][-4000:], rounds,
                            new_file_path_rule=(
                                "For this terminal task, each new_files path "
                                "is relative to the container filesystem root "
                                "(for example, workspace/new.conf)."),
                            peer_paths=[peer["name"] for peer in active
                                        if peer["name"] != remote],
                            outgoing=outgoing)
                        changed.extend(edited)
                        for message in outgoing:
                            target = message["to"]
                            text = f"From {remote}: {message['message']}"
                            inbox[target].append(text)
                            communication.setdefault(target, []).append(text)
                            traj.log("message", round=rounds, sender=remote,
                                     recipient=target,
                                     content=message["message"])
                            if (calls[target] == 1 and
                                    not any(peer["name"] == target for peer in queue)):
                                queue.append(next(peer for peer in active
                                                  if peer["name"] == target))
                        for name in edited:
                            if name != owner or not exists:
                                artifact = "/" + name
                                if artifact not in registered:
                                    registered.add(artifact)
                                    traj.log("component_registered",
                                             round=rounds, file=artifact)
                    await self._sync(environment, "/", local, changed)
                    traj.set_phase(f"3_EXECUTION (round {rounds}, check)")
                    post_obs = await self._observe(environment, workspace, checks)
                    traj.log("terminal_check", round=rounds,
                             commands=checks, output=post_obs["shell"])
                    obs = dict(post_obs)
                    obs["shell"] = (pre_obs["shell"] + "\n" + post_obs["shell"])[-12000:]
                    obs["runtime"] = pre_obs["shell"][-4000:]
                    if not checks:
                        obs["repo_test_rc"] = None
                        obs["repo_test_cmd"] = None
                    plan = {"edits": {item["name"]: {
                        "role": "Other", "change": item["objective"]}
                        for item in active}}
                    local_diff = "\n".join(
                        f"{name}\n{(local / name).read_text(errors='replace')[:1200]}"
                        if (local / name).is_file() else f"{name} [deleted]"
                        for name in changed)
                    traj.set_phase(f"4_CRITIC (round {rounds})")
                    report = await asyncio.to_thread(
                        p.critique, instruction, plan, obs, local_diff,
                        [item["name"] for item in active], rounds,
                        feedback, None, report,
                        {name: "created task artifact"
                         for name in registered})
                    if not changed and not before and not checks:
                        report["status"] = "FAIL"
                    if report["status"] == "PASS":
                        break
                    feedback = p.render_critic_report(report)
                    if rounds < self.max_iterations:
                        traj.set_phase(f"1.4_REPLANNING (round {rounds})")
                        revision = await asyncio.to_thread(
                            p.llm_json,
                            selection_prompt(
                                instruction,
                                obs["shell"] + "\nRegistered new artifacts: "
                                + str(sorted(registered)), feedback),
                            p.role_model("planner"), 2048)
                        active = components(revision.get("components"),
                                            workspace) or active
            summary = recorder.close(
                resolved=None, rounds=rounds, elapsed_s=time.time() - start)
            context.n_input_tokens = summary["prompt_tokens"]
            context.n_output_tokens = summary["completion_tokens"]
            context.model_usage = {
                model: ModelUsage(n_input_tokens=stats["prompt_tokens"],
                                  n_output_tokens=stats["completion_tokens"])
                for model, stats in summary["model_usage"].items()}
            context.metadata = {"critic_status": report["status"] if report else None,
                                "active_workspace": workspace,
                                "primitive_mode": "persistent_artifact"}
            (self.logs_dir / "hermes-result.json").write_text(
                json.dumps({"critic_status": context.metadata["critic_status"],
                            "rounds": rounds, "workspace": workspace},
                           indent=2))
        except BaseException:
            recorder.close(resolved=None, rounds=rounds,
                           elapsed_s=time.time() - start)
            raise
        finally:
            traj.set_trajectory(None)

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        self._configure_model()
        if (self.terminal_mode == "true" or
                (self.terminal_mode == "auto" and
                 (await self._workspace(instruction, environment)).rstrip("/")
                 != "/workspace/repo")):
            return await self._run_terminal_task(instruction, environment, context)
        workspace = await self._workspace(instruction, environment)
        start = time.time()
        with tempfile.TemporaryDirectory(prefix="hermes-harbor-") as tmp:
            local = Path(tmp) / "workspace"
            await self._download(environment, workspace, local)
            if self.terminal_mode == "auto" and not p.dev_primitive_files(local):
                return await self._run_terminal_task(
                    instruction, environment, context)
            recorder = traj.Trajectory(
                self.logs_dir.name, repo="harbor-task",
                problem_statement=instruction, out_root=self.logs_dir,
                config={"workspace": workspace, "model": self.model_name,
                        "backend": self.backend,
                        "max_iterations": self.max_iterations})
            traj.set_trajectory(recorder)
            rounds = 0
            try:
                traj.set_phase("1.0_ISSUE_ANALYSIS")
                architecture = await asyncio.to_thread(
                    p.discover_architecture, local, instruction)
                traj.set_phase(p.TRIAGE_PHASE)
                relevant = await asyncio.to_thread(
                    p.locate_files, local, instruction, architecture)
                if not relevant:
                    raise RuntimeError("no task artifacts localized")
                if len(relevant) > p.MAX_RELEVANT_FILES:
                    prompt = (f"Task: {instruction[:1200]}\nSelect up to "
                              f"{p.MAX_RELEVANT_FILES} relevant files from "
                              f"{[r['file'] for r in relevant[:p.RANK_CANDIDATES]]}. "
                              "Return JSON {\"files\":[\"path\", ...]}.")
                    ranked = await asyncio.to_thread(
                        p.llm_json, prompt, p.role_model("planner"), 1024)
                    pool = {r["file"]: r for r in relevant}
                    relevant = [pool[name] for name in ranked.get("files", [])
                                if isinstance(name, str) and name in pool]
                    if not relevant:
                        relevant = list(pool.values())[:p.MAX_RELEVANT_FILES]
                pool = {r["file"]: r.get("reason", "") for r in relevant}
                traj.set_phase("1.2_TASK_DECOMPOSITION")
                tasks = await asyncio.to_thread(
                    p.decompose_tasks, local, instruction, architecture, relevant)
                traj.set_phase("1.3_DEPENDENCY_ANALYSIS")
                deps = await asyncio.to_thread(
                    p.analyze_dependencies, local, instruction, architecture, tasks)
                traj.set_phase("1.4_EDIT_PLANNING")
                plan_obj = await asyncio.to_thread(
                    p.plan_edits, local, instruction, architecture, tasks, deps)
                command_prompt = (
                    f"Task instruction: {instruction[:4000]}\n"
                    f"Visible files: {[r['file'] for r in relevant]}\n"
                    "Choose one to three shell commands available in the agent "
                    "environment that check the requested behavior. Do not invoke "
                    "a benchmark verifier or hidden tests. Return JSON "
                    "{\"commands\":[\"shell command\", ...]}.")
                command_plan = await asyncio.to_thread(
                    p.llm_json, command_prompt, p.role_model("planner"), 1024)
                commands = [s for s in command_plan.get("commands", [])
                            if isinstance(s, str) and s.strip()][:3]
                if not commands:
                    commands = ["git diff --check"]
                baseline = await self._observe(environment, workspace, commands)
                report = None
                feedback = ""
                for rounds in range(1, self.max_iterations + 1):
                    scope = [{"file": task["file"],
                              "role": task.get("role", "Other")}
                             for task in tasks]
                    roles = {item["file"]: item["role"] for item in scope}
                    plan = p.render_plan(plan_obj)
                    traj.set_phase(f"2_INTER_FILE_COMMUNICATION (round {rounds})")
                    intents, adjusted = await asyncio.to_thread(
                        p.negotiate, local, instruction, architecture, plan,
                        scope, feedback, rounds, deps, roles)
                    edits = plan_obj.get("edits", {})
                    writers = [item["file"] for item in scope
                               if item["file"] in edits or item["file"] in adjusted]
                    traj.set_phase(f"2.5_SELF_MODIFY (round {rounds})")
                    changed = []
                    for owner in writers:
                        objective = (edits.get(owner) or {}).get("change", "")
                        additions = await asyncio.to_thread(
                            edit_component, local, owner, instruction, objective,
                            intents.get(owner, ""), feedback, rounds)
                        changed.extend(additions)
                    for name in changed:
                        if (local / name).is_file() and name not in pool:
                            pool[name] = "created by an activated primitive"
                    await self._sync(environment, workspace, local, changed)
                    traj.set_phase(f"3_EXECUTION (round {rounds})")
                    obs = await self._observe(environment, workspace, commands)
                    traj.log("verify", round=rounds, commands=commands,
                             output=obs["test"])
                    traj.set_phase(f"4_CRITIC (round {rounds})")
                    local_diff = "\n".join(
                        f"{name}\n{(local / name).read_text(errors='replace')[:1200]}"
                        if (local / name).is_file() else f"{name} [deleted]"
                        for name in changed)
                    report = await asyncio.to_thread(
                        p.critique, instruction, plan_obj, obs, local_diff,
                        [item["file"] for item in scope], rounds, feedback,
                        baseline["repo_test_rc"], report, pool)
                    if not changed and report["status"] == "PASS":
                        report["status"] = "FAIL"
                        report["summary"] = "No persistent artifact changed."
                    if report["status"] == "PASS":
                        break
                    if rounds < self.max_iterations:
                        feedback = p.render_critic_report(report)
                        await self._download(environment, workspace, local)
                        traj.set_phase(f"1.3_ACTIVE_SET_REVISION (round {rounds})")
                        tasks = p.revise_active_set(
                            local, tasks, report, rounds, localized_pool=pool)
                        traj.set_phase(f"1.4_EDIT_PLANNING (round {rounds})")
                        plan_obj = await asyncio.to_thread(
                            p.plan_edits, local, instruction, architecture,
                            tasks, deps, report, rounds)
                traj.set_phase("5_REFINED_PATCH")
                summary = recorder.close(
                    resolved=None, rounds=rounds, elapsed_s=time.time() - start)
                context.n_input_tokens = summary["prompt_tokens"]
                context.n_output_tokens = summary["completion_tokens"]
                context.model_usage = {
                    model: ModelUsage(n_input_tokens=stats["prompt_tokens"],
                                      n_output_tokens=stats["completion_tokens"])
                    for model, stats in summary["model_usage"].items()}
                context.metadata = {"critic_status": report["status"] if report else None,
                                    "active_workspace": workspace}
                (self.logs_dir / "hermes-result.json").write_text(
                    json.dumps({"critic_status": context.metadata["critic_status"],
                                "rounds": rounds, "workspace": workspace},
                               indent=2))
            except BaseException:
                recorder.close(resolved=None, rounds=rounds,
                               elapsed_s=time.time() - start)
                raise
            finally:
                traj.set_trajectory(None)
