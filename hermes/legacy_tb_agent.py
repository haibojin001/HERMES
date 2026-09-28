"""HERMES artifact primitives in DevOps-Gym's legacy Terminal-Bench runner."""

import json
import re
import tempfile
import time
import uuid
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.terminal.tmux_session import TmuxSession

from hermes import pipeline as p
from hermes import trajectory as traj
from hermes.artifacts import edit_component
from hermes.terminal_components import components, selection_prompt


class HermesTerminalBenchAgent(BaseAgent):
    """A task-environment adapter for DevOps-Gym's official `tb run`."""

    def __init__(self, model="qwen3:8b", backend="ollama", ollama_url=None,
                 vllm_url=None,
                 max_iterations=4, command_timeout=540,
                 planner_model=None, primitive_model=None, critic_model=None,
                 triage_model=None, reasoning_effort=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.backend = backend
        self.ollama_url = ollama_url
        self.vllm_url = vllm_url
        self.max_iterations = int(max_iterations)
        self.command_timeout = int(command_timeout)
        self.planner_model = planner_model
        self.primitive_model = primitive_model
        self.critic_model = critic_model
        self.triage_model = triage_model
        self.reasoning_effort = reasoning_effort
        if self.max_iterations < 1 or self.command_timeout < 1:
            raise ValueError("iteration and command limits must be positive")

    @staticmethod
    def name() -> str:
        return "hermes"

    @property
    def version(self) -> str:
        return "0.2.0"

    def _configure(self) -> None:
        if self.backend == "ollama":
            p.use_ollama(self.model, url=self.ollama_url)
        elif self.backend == "vllm":
            p.use_vllm(self.model, url=self.vllm_url)
        elif self.backend == "hosted":
            p.use_hosted(self.model, reasoning_effort=self.reasoning_effort)
        else:
            raise ValueError(f"unknown backend: {self.backend}")
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

    def _execute(self, session: TmuxSession, command: str) -> dict:
        marker = "__HERMES_EXIT_" + uuid.uuid4().hex + "__"
        wrapped = f"({command}) ; printf '{marker}:%s\\n' \"$?\""
        session.send_keys([wrapped, "Enter"], block=True,
                          max_timeout_sec=self.command_timeout)
        screen = session.capture_pane(capture_entire=True)
        matches = re.findall(re.escape(marker) + r":(\d+)", screen)
        return {"command": command,
                "exit_code": int(matches[-1]) if matches else 124,
                "output": screen[-12000:]}

    @staticmethod
    def _observation(records: list[dict]) -> dict:
        text = "\n\n".join(
            f"$ {r['command']}\nexit={r['exit_code']}\n{r['output']}"
            for r in records)
        return {"shell": text[-12000:], "test": text[-12000:],
                "runtime": text[-4000:], "trace": "",
                "repro_output": "", "repro_rc": None,
                "repo_test_cmd": None, "repo_test_rc": None,
                "setup_rc": 0, "container_rc": 0}

    @staticmethod
    def _container_run(session: TmuxSession, command: list[str]):
        return session.container.exec_run(command)

    def _stage_artifact(self, session: TmuxSession, remote: str,
                        staged: Path) -> bool:
        """Mirror one persistent task file; return False for a missing file."""
        kind = self._container_run(
            session, ["sh", "-c", 'test -f "$1"', "sh", remote])
        if kind.exit_code != 0:
            return False
        result = self._container_run(
            session, ["sh", "-c", 'cat "$1"', "sh", remote])
        if result.exit_code != 0:
            raise RuntimeError(f"cannot read task artifact: {remote}")
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(result.output)
        mode = self._container_run(
            session, ["sh", "-c", 'stat -c %a "$1"', "sh", remote])
        if mode.exit_code == 0:
            try:
                staged.chmod(int(mode.output.strip(), 8))
            except ValueError:
                pass
        return True

    def _sync_artifacts(self, session: TmuxSession, local: Path,
                        changed: list[str]) -> None:
        for owner in changed:
            remote = "/" + owner
            staged = local / owner
            if staged.is_file():
                session.copy_to_container(
                    staged, container_dir=str(Path(remote).parent),
                    container_filename=Path(remote).name)
            else:
                result = self._container_run(
                    session, ["sh", "-c", 'rm -f "$1"', "sh", remote])
                if result.exit_code != 0:
                    raise RuntimeError(f"cannot delete task artifact: {remote}")

    def perform_task(self, instruction: str, session: TmuxSession,
                     logging_dir: Path | None = None) -> AgentResult:
        self._configure()
        start = time.time()
        out_root = Path(logging_dir) if logging_dir else Path("hermes_data/legacy_tb")
        out_root.mkdir(parents=True, exist_ok=True)
        recorder = traj.Trajectory(
            "hermes", repo="devops_gym", problem_statement=instruction,
            out_root=out_root,
            config={"model": self.model, "backend": self.backend,
                    "max_iterations": self.max_iterations})
        traj.set_trajectory(recorder)
        rounds = 0
        commands_run = []
        report = None
        try:
            traj.set_phase("1_TASK_ANALYSIS")
            initial = self._execute(session, "pwd; ls -la | head -40")
            pwd = self._container_run(session, ["pwd"])
            workspace = pwd.output.decode(errors="replace").strip() if pwd.exit_code == 0 else "/"
            response = p.llm_json(
                selection_prompt(instruction, initial["output"]), p.role_model("planner"), 2048)
            active = components(response.get("components"), workspace)
            if not active:
                retry = p.llm_json(
                    selection_prompt(instruction, initial["output"],
                                     "Select a concrete persistent file path."),
                    p.role_model("planner"), 2048)
                active = components(retry.get("components"), workspace)
            if not active:
                raise RuntimeError("no persistent task artifact selected")
            feedback = ""
            registered = set()
            communication = {}
            with tempfile.TemporaryDirectory(prefix="hermes-devops-") as tmp:
                local = Path(tmp)
                for rounds in range(1, self.max_iterations + 1):
                    command_plan = p.llm_json(
                        f"Task: {instruction}\nArtifacts: {active}\n"
                        f"Critic feedback: {feedback[-3000:]}\n"
                        "Choose up to three task-visible commands to observe "
                        "the environment before editing and up to three commands "
                        "to check behavior after editing. Commands may interact "
                        "with services. Persistent artifact edits are handled "
                        "by their file owners. Never call hidden tests or the "
                        "benchmark verifier. Return JSON "
                        '{"observe":["command"],"check":["command"]}.',
                        p.role_model("planner"), 2048)
                    if not isinstance(command_plan, dict):
                        command_plan = {}
                    before = [c for c in (command_plan.get("observe") or [])[:3]
                              if isinstance(c, str) and c.strip()]
                    checks = [c for c in (command_plan.get("check") or [])[:3]
                              if isinstance(c, str) and c.strip()]
                    traj.set_phase(f"3_EXECUTION (round {rounds}, observation)")
                    pre_records = [self._execute(session, command)
                                   for command in before]
                    for record in pre_records:
                        commands_run.append(record["command"])
                        traj.log("terminal_observation", round=rounds, **record)
                    pre_obs = self._observation(pre_records)
                    changed = []
                    inbox = {item["name"]: list(
                        communication.get(item["name"], [])) for item in active}
                    queue = list(active)
                    calls = {item["name"]: 0 for item in active}
                    while queue:
                        item = queue.pop(0)
                        remote, owner = item["name"], item["owner"]
                        staged = local / owner
                        if calls[remote] >= 2:
                            continue
                        if calls[remote] == 0:
                            is_dir = self._container_run(
                                session, ["sh", "-c", 'test -d "$1"', "sh", remote])
                            if is_dir.exit_code == 0:
                                traj.log("artifact_edit", round=rounds,
                                         file=remote,
                                         error="selected path is a directory")
                                continue
                            exists = self._stage_artifact(session, remote, staged)
                            if not exists and staged.exists():
                                staged.unlink()
                        else:
                            exists = staged.is_file()
                        calls[remote] += 1
                        traj.set_phase(f"2_INTER_PRIMITIVE_COLLABORATION (round {rounds})")
                        outgoing = []
                        edited = edit_component(
                            local, owner, instruction, item["objective"],
                            "\n".join(inbox[remote])[-4000:],
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
                    self._sync_artifacts(session, local, changed)
                    traj.set_phase(f"3_EXECUTION (round {rounds}, check)")
                    post_records = [self._execute(session, command)
                                    for command in checks]
                    for record in post_records:
                        commands_run.append(record["command"])
                        traj.log("terminal_check", round=rounds, **record)
                    obs = self._observation(post_records)
                    obs["shell"] = (
                        pre_obs["shell"] + "\n" + obs["shell"])[-12000:]
                    obs["runtime"] = pre_obs["shell"][-4000:]
                    if checks:
                        obs["repo_test_cmd"] = "; ".join(checks)
                        obs["repo_test_rc"] = int(any(
                            record["exit_code"] != 0 for record in post_records))
                    plan = {"edits": {item["name"]: {
                        "role": "Other", "change": item["objective"]}
                        for item in active}}
                    local_diff = "\n".join(
                        f"{name}\n{(local / name).read_text(errors='replace')[:1200]}"
                        if (local / name).is_file() else f"{name} [deleted]"
                        for name in changed)
                    traj.set_phase(f"4_CRITIC (round {rounds})")
                    report = p.critique(
                        instruction, plan, obs, local_diff,
                        [item["name"] for item in active], rounds,
                        prev_feedback=feedback, prev_report=report,
                        inactive_candidates={
                            name: "created task artifact" for name in registered})
                    if not changed and not pre_records and not post_records:
                        report["status"] = "FAIL"
                    if report["status"] == "PASS":
                        break
                    feedback = p.render_critic_report(report)
                    if rounds < self.max_iterations:
                        traj.set_phase(f"1.4_REPLANNING (round {rounds})")
                        revision = p.llm_json(
                            selection_prompt(
                                instruction,
                                obs["shell"] + "\nRegistered new artifacts: "
                                + str(sorted(registered)), feedback),
                            p.role_model("planner"), 2048)
                        active = components(
                            revision.get("components"), workspace) or active
            summary = recorder.close(
                resolved=None, rounds=rounds, elapsed_s=time.time() - start)
            (out_root / "hermes-result.json").write_text(
                json.dumps({"critic_status": report["status"] if report else None,
                            "rounds": rounds, "commands": commands_run},
                           indent=2) + "\n")
            return AgentResult(total_input_tokens=summary["prompt_tokens"],
                               total_output_tokens=summary["completion_tokens"])
        except BaseException:
            recorder.close(resolved=None, rounds=rounds,
                           elapsed_s=time.time() - start)
            raise
        finally:
            traj.set_trajectory(None)
