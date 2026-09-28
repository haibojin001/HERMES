"""Checks for manifest isolation and paper-specific score aggregation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hermes.manifest_runner import _patch, execute_visible, load_manifest, materialize
from score_benchmarks import summarize


class TestManifest(unittest.TestCase):
    def test_hidden_grader_is_not_in_the_task_visible_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "answer.py").write_text("value = 1\n")
            manifest = root / "tasks.jsonl"
            manifest.write_text(json.dumps({
                "task_id": "t1", "benchmark": "swe_refactor",
                "workspace": str(source), "issue": "change value",
                "visible_commands": ["python3 -c 'print(1)'"]}) + "\n")
            row = load_manifest(manifest)[0]
            copy = materialize(row, root / "output")
            (copy / "answer.py").write_text("value = 2\n")
            self.assertEqual((source / "answer.py").read_text(), "value = 1\n")
            observation = execute_visible(copy, row["visible_commands"], 10)
            self.assertEqual(observation["repo_test_rc"], 0)

    def test_manifest_rejects_missing_visible_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "tasks.jsonl"
            path.write_text(json.dumps({
                "task_id": "t1", "benchmark": "terminal_bench",
                "workspace": tmp, "issue": "task",
                "visible_commands": []}) + "\n")
            with self.assertRaisesRegex(ValueError, "visible_commands"):
                load_manifest(path)

    def test_patch_excludes_untracked_execution_byproducts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "answer.py").write_text("value = 1\n")
            row = {"workspace": str(source)}
            repo = materialize(row, root / "run")
            (repo / "answer.py").write_text("value = 2\n")
            (repo / "new.py").write_text("created = True\n")
            (repo / "test-cache.log").write_text("benchmark runtime output\n")
            import subprocess
            subprocess.run(["git", "add", "-N", "--", "new.py"],
                           cwd=repo, check=True)
            patch = _patch(repo)
            self.assertIn("new.py", patch)
            self.assertIn("answer.py", patch)
            self.assertNotIn("test-cache.log", patch)


class TestScoreProtocol(unittest.TestCase):
    def test_terminal_mean_and_std_across_five_runs(self):
        manifest = [{"task_id": "a", "benchmark": "terminal_bench"},
                    {"task_id": "b", "benchmark": "terminal_bench"}]
        runs = [{"task_id": task, "benchmark": "terminal_bench", "run_index": i}
                for task in ("a", "b") for i in range(5)]
        scores = [{**row, "source": "official grader", "score": int(
            row["task_id"] == "a" or row["run_index"] == 0)}
            for row in runs]
        report = summarize(manifest, runs, scores, "terminal_bench", 5)
        self.assertEqual(report["per_run_pct"], [100, 50, 50, 50, 50])
        self.assertEqual(report["resolution_pct"], 60)

    def test_devops_uses_unweighted_category_average(self):
        categories = ("build_configuration", "monitoring", "issue_resolving",
                      "test_generation")
        manifest = [{"task_id": name, "benchmark": "devops_gym", "category": name}
                    for name in categories]
        runs = [{"task_id": name, "benchmark": "devops_gym", "run_index": 0}
                for name in categories]
        scores = [{**row, "source": "official grader",
                   "score": int(row["task_id"] == "monitoring")} for row in runs]
        report = summarize(manifest, runs, scores, "devops_gym", 1)
        self.assertEqual(report["unweighted_average_pct"], 25)

    def test_missing_score_is_an_error(self):
        manifest = [{"task_id": "a", "benchmark": "swe_refactor"}]
        runs = [{"task_id": "a", "benchmark": "swe_refactor", "run_index": 0}]
        with self.assertRaisesRegex(ValueError, "missing official scores"):
            summarize(manifest, runs, [], "swe_refactor", 1)


if __name__ == "__main__":
    unittest.main()
