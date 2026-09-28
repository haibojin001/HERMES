"""Checks for official verifier result parsing and validated artifact edits."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hermes.artifacts import apply_artifact_edit, edit_component
from hermes.terminal_components import artifact_path, components
from report_official import read_devops, read_harbor, summarize


class TestArtifactEdits(unittest.TestCase):
    def test_manuscript_fields_and_invalid_json_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workspace/source.py").parent.mkdir()
            (root / "workspace/source.py").write_text("value = 1\n")
            proposal = {
                "component": "workspace/source.py",
                "action": "modify",
                "summary": "Update the value",
                "changes": [{"location": "value", "description": "Use 2"}],
                "updated_artifact": "value = 2\n",
                "messages": [{"target": "/workspace/peer.py",
                              "message": "Expect value 2."}],
                "validation_notes": ["Check peer reads value 2."],
            }
            outgoing = []
            with patch("hermes.artifacts.p.llm_json",
                       side_effect=[{"_error": "parse failed"}, proposal]) as model:
                changed = edit_component(
                    root, "workspace/source.py", "Update value", "Use 2",
                    peer_paths=["/workspace/peer.py"], outgoing=outgoing)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(changed, ["workspace/source.py"])
            self.assertEqual((root / changed[0]).read_text(), "value = 2\n")
            self.assertEqual(outgoing, [
                {"to": "/workspace/peer.py", "message": "Expect value 2."}])

    def test_missing_owner_can_be_created_as_persistent_primitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proposal = {"action": "create", "content": "status: healthy\n",
                        "new_files": []}
            with patch("hermes.artifacts.p.llm_json", return_value=proposal):
                changed = edit_component(
                    root, "workspace/report.txt", "Write report",
                    "Create the requested report")
            self.assertEqual(changed, ["workspace/report.txt"])
            self.assertEqual((root / changed[0]).read_text(), "status: healthy\n")

    def test_create_new_artifact_and_reject_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.py").write_text("value = 1\n")
            changed = apply_artifact_edit(root, "old.py", {
                "action": "modify",
                "replacements": [{"old": "value = 1", "new": "value = 2"}],
                "new_files": [{"path": "src/new.py", "content": "result = 2\n"}]})
            self.assertEqual(changed, ["old.py", "src/new.py"])
            self.assertEqual((root / "src/new.py").read_text(), "result = 2\n")
            before = (root / "old.py").read_text()
            with self.assertRaisesRegex(ValueError, "outside repository"):
                apply_artifact_edit(root, "old.py", {
                    "action": "modify",
                    "replacements": [{"old": "value = 2", "new": "value = 3"}],
                    "new_files": [{"path": "../escape.py",
                                   "content": "bad = True\n"}]})
            self.assertEqual((root / "old.py").read_text(), before)

    def test_reject_duplicate_additions_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "old.py").write_text("value = 1\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                apply_artifact_edit(root, "old.py", {
                    "action": "modify",
                    "replacements": [{"old": "value = 1", "new": "value = 2"}],
                    "new_files": [{"path": "new.py", "content": "x = 1\n"},
                                  {"path": "new.py", "content": "x = 2\n"}]})
            self.assertEqual((root / "old.py").read_text(), "value = 1\n")
            self.assertFalse((root / "new.py").exists())

class TestTerminalComponents(unittest.TestCase):
    def test_persistent_script_and_service_files_are_primitives(self):
        planned = components([
            {"path": "scripts/check.sh", "objective": "check service"},
            {"path": "/etc/systemd/system/app.service",
             "objective": "fix service"},
            {"path": "Makefile", "objective": "fix build"},
            {"path": "/workspace/output.txt", "objective": "write result"}],
            "/workspace/repo")
        self.assertEqual(
            [item["name"] for item in planned],
            ["/workspace/repo/scripts/check.sh",
             "/etc/systemd/system/app.service",
             "/workspace/repo/Makefile",
             "/workspace/output.txt"])

    def test_action_labels_and_transient_paths_are_not_primitives(self):
        planned = components([
            {"path": "monitor-service", "objective": "run monitor"},
            {"path": "restart", "objective": "restart service"},
            {"path": "/proc/123/status", "objective": "inspect process"},
            {"path": "state.log", "objective": "read transient log"},
            {"path": "output.txt", "objective": "write result"},
            {"path": "output.txt", "objective": "duplicate"}], "/workspace")
        self.assertEqual([item["name"] for item in planned],
                         ["/workspace/output.txt"])
        with self.assertRaisesRegex(ValueError, "invalid artifact path"):
            artifact_path("/workspace", "../outside.py")


class TestOfficialResults(unittest.TestCase):
    def test_swe_refactor_requires_all_valid_official_grades(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark"
            job = root / "jobs" / "hermes_swe_refactor_paper_r0"
            for index in range(20):
                task_id = f"task{index:02d}"
                path = benchmark / "tasks" / task_id
                path.mkdir(parents=True)
                (path / "task.toml").write_text("")
                trial = job / task_id
                trial.mkdir(parents=True)
                (trial / "result.json").write_text(json.dumps({
                    "task_name": f"swerefactor/{task_id}",
                    "verifier_result": {"rewards": {
                        "valid": 1, "reward": index / 20}},
                    "agent_result": {"n_input_tokens": 5,
                                     "n_output_tokens": 2}}))
            rows, usage = read_harbor(root / "jobs", "swe_refactor", "paper",
                                      benchmark, 1)
            self.assertEqual(len(rows), 20)
            self.assertEqual(usage["input_tokens"], 100)
            self.assertAlmostEqual(
                summarize(rows, "swe_refactor", 1, usage)["composite_pct"], 47.5)
            invalid = job / "task00" / "result.json"
            data = json.loads(invalid.read_text())
            data["verifier_result"]["rewards"]["valid"] = 0
            invalid.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "invalid SWE Refactor"):
                read_harbor(root / "jobs", "swe_refactor", "paper", benchmark, 1)

    def test_terminal_partial_reward_is_not_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark"
            for index in range(66):
                task = benchmark / "tasks" / f"task{index:02d}"
                task.mkdir(parents=True)
                (task / "task.toml").write_text("")
            for run in range(5):
                job = root / "jobs" / f"hermes_terminal_bench_paper_r{run}"
                for index in range(66):
                    trial = job / f"task{index:02d}"
                    trial.mkdir(parents=True)
                    reward = 0.5 if index == 0 else int(run == 0)
                    (trial / "result.json").write_text(json.dumps({
                        "task_name": f"task{index:02d}",
                        "verifier_result": {"rewards": {"reward": reward}},
                        "agent_result": {"n_input_tokens": 1,
                                         "n_output_tokens": 1}}))
            rows, usage = read_harbor(root / "jobs", "terminal_bench", "paper",
                                      benchmark, 5)
            result = summarize(rows, "terminal_bench", 5, usage)
            self.assertEqual(result["graded_trials"], 330)
            self.assertEqual(result["per_run_pct"][0], 100 * 65 / 66)
            self.assertTrue(all(rate == 0 for rate in result["per_run_pct"][1:]))
            (root / "jobs" / "hermes_terminal_bench_paper_r4" /
             "task00" / "result.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing 1 official"):
                read_harbor(root / "jobs", "terminal_bench", "paper",
                            benchmark, 5)

    def test_devops_unweighted_mean_from_legacy_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category, score in (
                    ("build", True), ("monitor", False),
                    ("issue_resolving", False), ("test_generation", False)):
                task = root / "benchmark" / "tasks" / category / "one"
                task.mkdir(parents=True)
                (task / "task.yaml").write_text("")
                job = root / "jobs" / f"hermes_{category}_paper"
                job.mkdir(parents=True)
                (job / "results.json").write_text(json.dumps({
                    "results": [{"task_id": "one", "is_resolved": score,
                                 "total_input_tokens": 4,
                                 "total_output_tokens": 2}]}))
            rows, usage = read_devops(root / "jobs", "paper",
                                      root / "benchmark")
            result = summarize(rows, "devops_gym", 1, usage)
            self.assertEqual(result["unweighted_average_pct"], 25)
            self.assertEqual(result["category_task_counts"]["monitoring"], 1)


if __name__ == "__main__":
    unittest.main()
