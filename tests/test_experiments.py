#!/usr/bin/env python3
"""Tests for the experiment plumbing - the parts that turn runs into a table.

A bug here does not raise: it prints a number. The cases below are the ones that
would have printed a wrong one. No model, no container.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_results as agg
from hermes import pipeline as p


def write_run(root: Path, label: str, instances, graded=None, traj=None):
    """A run directory shaped like the one scripts/_lib.sh produces."""
    run = root / "runs" / label
    (run / "eval").mkdir(parents=True)
    with (run / "predictions.jsonl").open("w") as f:
        for iid, patch in instances.items():
            f.write(json.dumps({"instance_id": iid, "model_patch": patch}) + "\n")
    if graded:
        with (run / "eval" / "results.jsonl").open("w") as f:
            for iid, resolved in graded.items():
                f.write(json.dumps({"instance_id": iid,
                                    "resolved": resolved}) + "\n")
    traj_root = root / "trajectories"
    for iid, summary in (traj or {}).items():
        d = traj_root / f"{iid}_{label}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(summary))
    return run, traj_root


class TestSummarize(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolved_rate_and_standard_error(self):
        run, traj = write_run(
            self.root, "complete",
            {f"i{n}": "diff" for n in range(4)},
            graded={"i0": True, "i1": True, "i2": False, "i3": False})
        s = agg.summarize(agg.collect(run, traj))
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["graded"], 4)
        self.assertEqual(s["resolved"], 2)
        self.assertAlmostEqual(s["resolved_pct"], 50.0)
        # sqrt(.5*.5/4) = .25
        self.assertAlmostEqual(s["se_pct"], 25.0)

    def test_ungraded_instances_are_excluded_from_the_rate(self):
        """The grader skips an instance with no environment image instead of
        failing it. Counting those as unresolved understates every row, and by
        a different amount per row depending on which images happened to be
        present."""
        run, traj = write_run(self.root, "complete",
                             {"i0": "diff", "i1": "diff", "i2": "diff"},
                             graded={"i0": True})
        s = agg.summarize(agg.collect(run, traj))
        self.assertEqual((s["n"], s["graded"], s["resolved"]), (3, 1, 1))
        self.assertAlmostEqual(s["resolved_pct"], 100.0)
        # and the reader is told, rather than left to infer it
        self.assertIn("1/3 graded", agg.render([s], ""))

    def test_a_regraded_instance_counts_once(self):
        """results.jsonl is appended to, so re-grading leaves two records for one
        instance. Counting both would push the rate above 100%."""
        run, traj = write_run(self.root, "complete", {"i0": "diff"})
        with (run / "eval" / "results.jsonl").open("w") as f:
            f.write(json.dumps({"instance_id": "i0", "resolved": False}) + "\n")
            f.write(json.dumps({"instance_id": "i0", "resolved": True}) + "\n")
        s = agg.summarize(agg.collect(run, traj))
        self.assertEqual((s["graded"], s["resolved"]), (1, 1))

    def test_empty_patch_share_and_trajectory_means(self):
        run, traj = write_run(
            self.root, "complete", {"i0": "diff --git", "i1": ""},
            graded={"i0": True, "i1": False},
            traj={"i0": {"instance_id": "i0", "resolved": True, "rounds": 1,
                         "elapsed_s": 100.0, "llm_calls": 20,
                         "prompt_tokens": 1000, "completion_tokens": 100,
                         "usage_missing_calls": 0,
                         "model_usage": {"qwen3:8b": {}}},
                  "i1": {"instance_id": "i1", "resolved": False, "rounds": 3,
                         "elapsed_s": 300.0, "llm_calls": 40,
                         "prompt_tokens": 3000, "completion_tokens": 300,
                         "usage_missing_calls": 0,
                         "model_usage": {"qwen3:8b": {}}}})
        s = agg.summarize(agg.collect(run, traj), price=(3.0, 15.0))
        self.assertAlmostEqual(s["empty_patch_pct"], 50.0)
        self.assertAlmostEqual(s["mean_rounds"], 2.0)
        self.assertAlmostEqual(s["mean_calls"], 30.0)
        self.assertAlmostEqual(s["mean_wall_s"], 200.0)
        # 2000 prompt * $3/M + 200 completion * $15/M
        self.assertAlmostEqual(s["mean_cost_usd"], 2000/1e6*3 + 200/1e6*15)

    def test_a_truncated_last_line_does_not_lose_the_run(self):
        """A run killed mid-write leaves half a JSON object behind."""
        run, traj = write_run(self.root, "complete", {"i0": "diff"},
                             graded={"i0": True})
        with (run / "predictions.jsonl").open("a") as f:
            f.write('{"instance_id": "i1", "model_pat')
        s = agg.summarize(agg.collect(run, traj))
        self.assertEqual(s["n"], 1)

    def test_a_row_without_trajectories_prints_a_dash_not_a_zero(self):
        """Two rows, one of which kept no trajectories. Printing 0 rounds and 0
        tokens for it would read as a cheap configuration rather than a missing
        one."""
        with_traj, traj = write_run(
            self.root, "complete", {"i0": "diff"}, graded={"i0": True},
            traj={"i0": {"instance_id": "i0", "resolved": True, "rounds": 2,
                         "llm_calls": 20, "elapsed_s": 100.0,
                         "prompt_tokens": 500, "completion_tokens": 50}})
        without, _ = write_run(self.root, "no_critic", {"i0": "diff"},
                              graded={"i0": False})
        table = agg.render([agg.summarize(agg.collect(with_traj, traj)),
                            agg.summarize(agg.collect(without, traj))], "t")
        rounds_col = [line.split("|")[6].strip() for line in
                      table.splitlines() if line.startswith("| ")]
        self.assertEqual(rounds_col[0], "rounds")
        self.assertEqual(rounds_col[1], "2.00")
        self.assertEqual(rounds_col[2], "-")

    def test_a_directory_of_trajectories_is_also_a_row(self):
        """So the shipped `results/` can be summarized without a run directory,
        using the held-out evaluation the trajectory recorded."""
        traj_dir = self.root / "standalone"
        for iid, resolved in (("a", True), ("b", False)):
            d = traj_dir / iid
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps(
                {"instance_id": iid, "resolved": resolved, "rounds": 2,
                 "llm_calls": 10, "elapsed_s": 5.0}))
        s = agg.summarize(agg.collect(traj_dir, self.root / "nowhere"))
        self.assertEqual((s["n"], s["graded"], s["resolved"]), (2, 2, 1))


class TestRoleBackbones(unittest.TestCase):
    """The backbone-scale table moves one role at a time, so an unset slot has to
    follow the reasoning default rather than pin itself at import time."""

    def setUp(self):
        self.saved = (p.PLANNER_MODEL, p.PRIMITIVE_MODEL, p.CRITIC_MODEL,
                      p.THINKING_MODEL)

    def tearDown(self):
        (p.PLANNER_MODEL, p.PRIMITIVE_MODEL, p.CRITIC_MODEL,
         p.THINKING_MODEL) = self.saved

    def test_unset_slots_follow_the_reasoning_backbone(self):
        p.THINKING_MODEL = "backbone/x"
        p.PLANNER_MODEL = p.PRIMITIVE_MODEL = p.CRITIC_MODEL = None
        for role in ("planner", "primitive", "critic"):
            self.assertEqual(p.role_model(role), "backbone/x", role)

    def test_one_slot_moves_alone(self):
        p.THINKING_MODEL = "small"
        p.PLANNER_MODEL = p.PRIMITIVE_MODEL = None
        p.CRITIC_MODEL = "big"
        self.assertEqual(p.role_model("critic"), "big")
        self.assertEqual(p.role_model("planner"), "small")
        self.assertEqual(p.role_model("primitive"), "small")

    def test_localization_is_not_one_of_the_roles(self):
        # Bug localization is thousands of short calls; it has its own slot so a
        # role sweep does not move it by accident.
        self.assertNotIn(p.TRIAGE_MODEL,
                         [p.role_model(r) for r in ("planner", "critic")]
                         if p.TRIAGE_MODEL != p.THINKING_MODEL else ["sentinel"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
