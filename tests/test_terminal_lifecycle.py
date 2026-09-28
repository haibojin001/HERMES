"""Adapter smoke tests for creating a persistent task artifact."""

import asyncio
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hermes import pipeline as p


def model_reply(prompt, *_args, **_kwargs):
    if "Select up to six Dev-Primitives" in prompt:
        return {"components": [
            {"path": "/workspace/report.txt", "objective": "Write report"}]}
    if "Choose up to three task-visible commands" in prompt:
        return {"observe": [], "check": ["cat /workspace/report.txt"]}
    if "You are a Dev-Primitive in HERMES." in prompt:
        return {"action": "create", "content": "healthy\n", "new_files": []}
    if "You are the Critic" in prompt:
        return {"status": "PASS", "evidence": ["report exists"],
                "summary": "The report is present."}
    raise AssertionError(f"unexpected model prompt: {prompt[:100]}")


class FakeHarborEnvironment:
    def __init__(self):
        self.files = {}

    async def is_dir(self, path):
        return path == "/workspace"

    async def is_file(self, path):
        return path in self.files

    async def exec(self, command, cwd=None, timeout_sec=None):
        from types import SimpleNamespace
        if command == "pwd":
            text = "/workspace\n"
        elif command.startswith("cat "):
            text = self.files[command[4:]]
        else:
            text = ""
        return SimpleNamespace(return_code=0, stdout=text, stderr="")

    async def download_file(self, source_path, target_path):
        Path(target_path).write_text(self.files[source_path])

    async def upload_file(self, source_path, target_path):
        self.files[target_path] = Path(source_path).read_text()


class FakeContainer:
    def __init__(self):
        self.files = {}

    def exec_run(self, command):
        from types import SimpleNamespace
        if command == ["pwd"]:
            return SimpleNamespace(exit_code=0, output=b"/workspace\n")
        operation, remote = command[2], command[-1]
        if "test -d" in operation:
            return SimpleNamespace(exit_code=1, output=b"")
        if "test -f" in operation:
            return SimpleNamespace(
                exit_code=0 if remote in self.files else 1, output=b"")
        if "cat" in operation:
            return SimpleNamespace(exit_code=0, output=self.files[remote])
        if "stat" in operation:
            return SimpleNamespace(exit_code=0, output=b"644\n")
        if "rm -f" in operation:
            self.files.pop(remote, None)
            return SimpleNamespace(exit_code=0, output=b"")
        raise AssertionError(command)


class FakeTmuxSession:
    def __init__(self):
        self.container = FakeContainer()
        self.screen = ""

    def send_keys(self, keys, block=True, max_timeout_sec=None):
        wrapped = keys[0]
        marker = re.search(r"__HERMES_EXIT_[a-f0-9]+__", wrapped).group()
        if "cat /workspace/report.txt" in wrapped:
            body = self.container.files["/workspace/report.txt"].decode()
        else:
            body = "/workspace\n"
        self.screen = f"{body}{marker}:0\n"

    def capture_pane(self, capture_entire=True):
        return self.screen

    def copy_to_container(self, staged, container_dir, container_filename):
        remote = f"{container_dir.rstrip('/')}/{container_filename}"
        self.container.files[remote] = Path(staged).read_bytes()


class TestTerminalLifecycle(unittest.TestCase):
    def test_harbor_delivers_edit_message_to_peer_in_same_round(self):
        try:
            from harbor.models.agent.context import AgentContext
            from hermes.harbor_agent import HermesHarborAgent
        except ImportError:
            self.skipTest("Harbor extra is not installed")
        seen = {"peer_message": False}

        def reply(prompt, *_args, **_kwargs):
            if "Select up to six Dev-Primitives" in prompt:
                return {"components": [
                    {"path": "/workspace/config.yaml", "objective": "Inspect config"},
                    {"path": "/workspace/report.txt", "objective": "Update report"}]}
            if "Choose up to three task-visible commands" in prompt:
                return {"observe": [], "check": ["cat /workspace/report.txt"]}
            if "You own one persistent artifact: workspace/config.yaml" in prompt:
                self.assertIn("enabled: true", prompt)
                return {"action": "keep", "messages": [
                    {"to": "/workspace/report.txt",
                     "message": "Report the configured healthy state."}]}
            if "You own one persistent artifact: workspace/report.txt" in prompt:
                seen["peer_message"] = "Report the configured healthy state." in prompt
                self.assertIn("draft", prompt)
                return {"action": "modify", "replacements": [
                    {"old": "draft", "new": "healthy"}]}
            if "You are the Critic" in prompt:
                return {"status": "PASS", "evidence": ["report updated"],
                        "summary": "The report is ready."}
            raise AssertionError(prompt[:100])

        with tempfile.TemporaryDirectory() as tmp, patch.object(
                p, "llm_json", side_effect=reply):
            agent = HermesHarborAgent(
                logs_dir=Path(tmp) / "logs", model_name="test-model",
                backend="hosted", terminal_mode="true", max_iterations=1)
            environment = FakeHarborEnvironment()
            environment.files["/workspace/config.yaml"] = "enabled: true\n"
            environment.files["/workspace/report.txt"] = "draft\n"
            asyncio.run(agent.run(
                "Update /workspace/report.txt", environment, AgentContext()))
            self.assertTrue(seen["peer_message"])
            self.assertEqual(
                environment.files["/workspace/report.txt"], "healthy\n")

    def test_harbor_registers_new_file_for_next_round(self):
        try:
            from harbor.models.agent.context import AgentContext
            from hermes.harbor_agent import HermesHarborAgent
        except ImportError:
            self.skipTest("Harbor extra is not installed")
        state = {"selection": 0, "critique": 0}

        def reply(prompt, *_args, **_kwargs):
            if "Select up to six Dev-Primitives" in prompt:
                state["selection"] += 1
                path = ("/workspace/config.yaml" if state["selection"] == 1
                        else "/workspace/report.txt")
                return {"components": [{"path": path, "objective": "Finish report"}]}
            if "Choose up to three task-visible commands" in prompt:
                return {"observe": [], "check": ["cat /workspace/report.txt"]}
            if "You are a Dev-Primitive in HERMES." in prompt:
                if "workspace/config.yaml" in prompt:
                    return {"action": "keep", "new_files": [
                        {"path": "workspace/report.txt", "content": "draft\n"}]}
                return {"action": "modify", "replacements": [
                    {"old": "draft", "new": "healthy"}]}
            if "You are the Critic" in prompt:
                state["critique"] += 1
                if state["critique"] == 1:
                    self.assertIn("/workspace/report.txt", prompt)
                    return {"status": "FAIL",
                            "missing_components": ["/workspace/report.txt"],
                            "summary": "Finish the new report."}
                return {"status": "PASS", "evidence": ["report updated"],
                        "summary": "The report is ready."}
            raise AssertionError(prompt[:100])

        with tempfile.TemporaryDirectory() as tmp, patch.object(
                p, "llm_json", side_effect=reply):
            agent = HermesHarborAgent(
                logs_dir=Path(tmp) / "logs", model_name="test-model",
                backend="hosted", terminal_mode="true", max_iterations=2)
            environment = FakeHarborEnvironment()
            environment.files["/workspace/config.yaml"] = "enabled: true\n"
            asyncio.run(agent.run(
                "Write /workspace/report.txt", environment, AgentContext()))
            self.assertEqual(
                environment.files["/workspace/report.txt"], "healthy\n")
            events = [
                json.loads(line) for line in
                (Path(tmp) / "logs" / "logs" / "trajectory.jsonl").read_text().splitlines()]
            self.assertIn(
                "/workspace/report.txt",
                [event["file"] for event in events
                 if event["kind"] == "component_registered"])

    def test_harbor_creates_report_as_artifact(self):
        try:
            from harbor.models.agent.context import AgentContext
            from hermes.harbor_agent import HermesHarborAgent
        except ImportError:
            self.skipTest("Harbor extra is not installed")
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                p, "llm_json", side_effect=model_reply):
            agent = HermesHarborAgent(
                logs_dir=Path(tmp) / "logs", model_name="test-model",
                backend="hosted", terminal_mode="true", max_iterations=1)
            environment = FakeHarborEnvironment()
            context = AgentContext()
            asyncio.run(agent.run(
                "Write /workspace/report.txt", environment, context))
            self.assertEqual(
                environment.files["/workspace/report.txt"], "healthy\n")
            self.assertEqual(context.metadata["primitive_mode"],
                             "persistent_artifact")

    def test_legacy_creates_report_as_artifact(self):
        try:
            from hermes.legacy_tb_agent import HermesTerminalBenchAgent
        except ImportError:
            self.skipTest("legacy Terminal-Bench extra is not installed")
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                p, "llm_json", side_effect=model_reply):
            agent = HermesTerminalBenchAgent(
                backend="hosted", model="test-model", max_iterations=1)
            session = FakeTmuxSession()
            agent.perform_task(
                "Write /workspace/report.txt", session, Path(tmp))
            self.assertEqual(
                session.container.files["/workspace/report.txt"], b"healthy\n")


if __name__ == "__main__":
    unittest.main()
