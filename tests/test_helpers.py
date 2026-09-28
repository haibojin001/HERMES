#!/usr/bin/env python3
"""Tests for the pure helpers - the ones whose bugs are silent.

None of these call a model or a container, so `python tests/test_helpers.py`
runs in under a second. They cover the guards that were added because their
absence produced a wrong number rather than an error: a setup crash read as a
successful reproduction, a test command that ran nothing read as "no
regressions", a Critic-invented path silently dropped so the active set could
only shrink, and a Critic repeating its own guidance because it could not see
that the evidence had not moved.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes import pipeline as p
from hermes import trajectory as traj


class TestReproSetupGuard(unittest.TestCase):
    """A script that dies in setup exits non-zero, exactly like one that
    reproduces the bug. Conflating them pins the whole trajectory to FAIL for a
    reason that has nothing to do with the patch."""

    def test_django_setup_failures_are_not_reproductions(self):
        for output in (
                "django.core.exceptions.ImproperlyConfigured: The SECRET_KEY "
                "setting must not be empty.",
                "ModuleNotFoundError: No module named 'admin_utils'",
                "django.core.exceptions.AppRegistryNotReady: Apps aren't loaded yet."):
            self.assertTrue(p.repro_setup_failed(output), output[:40])

    def test_a_real_assertion_failure_is_a_reproduction(self):
        self.assertFalse(p.repro_setup_failed(
            "AssertionError: '\\u4e2d\\u6587' != 'zhongwen'"))

    def test_empty_output(self):
        self.assertFalse(p.repro_setup_failed(""))
        self.assertFalse(p.repro_setup_failed(None))


class TestTestCommandUsable(unittest.TestCase):
    """A command that exits 0 without running tests reads as 'no regressions'
    every round: the Critic can then never fail a patch on it, nor pass one."""

    def test_a_run_that_collected_tests_is_usable(self):
        self.assertTrue(p.test_command_ran_anything(
            "Ran 34 tests in 0.412s\n\nOK"))

    def test_collection_errors_are_not_usable(self):
        self.assertFalse(p.test_command_ran_anything(
            "Error loading MySQLdb module."))

    def test_silence_is_not_usable(self):
        self.assertFalse(p.test_command_ran_anything(""))


class TestResolveComponent(unittest.TestCase):
    """Eq. 9 can only activate a component whose path resolves. Before this,
    `missing_components` held names like `tests/settings.py` that do not exist,
    they were dropped without a record, and `activated` was 0 in every run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        for rel in ("django/db/models/fields/json.py",
                    "django/forms/fields.py",
                    "django/urls/resolvers.py",
                    "tests/forms_tests/tests.py"):
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text("")
        self.pool = {"django/db/models/fields/json.py": "encodes JSON",
                     "django/forms/fields.py": "prepares the form value"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_path(self):
        self.assertEqual(
            p.resolve_component(self.repo, "django/forms/fields.py", self.pool),
            "django/forms/fields.py")

    def test_path_suffix_from_the_pool(self):
        self.assertEqual(
            p.resolve_component(self.repo, "db/models/fields/json.py", self.pool),
            "django/db/models/fields/json.py")

    def test_bare_basename_unique_in_the_pool(self):
        self.assertEqual(p.resolve_component(self.repo, "fields.py", self.pool),
                         "django/forms/fields.py")

    def test_basename_found_in_the_repository_but_not_the_pool(self):
        # The Critic may legitimately implicate a file localization skipped.
        self.assertEqual(
            p.resolve_component(self.repo, "resolvers.py", self.pool),
            "django/urls/resolvers.py")

    def test_invented_path_is_rejected(self):
        self.assertEqual(p.resolve_component(self.repo, "tests/settings.py",
                                             self.pool), "")
        self.assertEqual(p.resolve_component(self.repo, "test_settings.py",
                                             self.pool), "")

    def test_leading_dot_slash_and_empty_names(self):
        self.assertEqual(
            p.resolve_component(self.repo, "./django/forms/fields.py", self.pool),
            "django/forms/fields.py")
        self.assertEqual(p.resolve_component(self.repo, "", self.pool), "")
        self.assertEqual(p.resolve_component(self.repo, "django/forms/", self.pool), "")


class TestEvidenceDelta(unittest.TestCase):
    """Without this, the Critic saw identical evidence each round and re-derived
    identical guidance word for word, which made re-planning decorative."""

    OBS = {"repro_rc": None, "repo_test_rc": 0, "test": "Ran 34 tests\nOK",
           "repro_output": ""}

    def test_first_round_has_no_delta(self):
        self.assertEqual(p.evidence_delta(self.OBS, None), "")

    def test_identical_evidence_says_nothing_moved(self):
        prev = {"round": 1, "repro_rc": None, "repo_test_rc": 0,
                "test_output": "Ran 34 tests\nOK", "repro_output": ""}
        out = p.evidence_delta(self.OBS, prev)
        self.assertIn("EVIDENCE COMPARED WITH ROUND 1", out)
        self.assertIn("unchanged", out)
        self.assertIn("Nothing in the execution evidence moved", out)
        self.assertNotIn("changed:   ", out)

    def test_a_changed_signal_is_named_with_both_values(self):
        prev = {"round": 1, "repro_rc": None, "repo_test_rc": 1,
                "test_output": "FAILED (failures=2)", "repro_output": ""}
        out = p.evidence_delta(self.OBS, prev)
        self.assertIn("changed:", out)
        self.assertIn("repository test exit code", out)
        self.assertNotIn("Nothing in the execution evidence moved", out)


class TestAblationWiring(unittest.TestCase):
    """One flag per row of the component-ablation table, and a label that ends up
    in the trajectory so a run states which configuration produced it."""

    def setUp(self):
        self.saved = (p.ABLATE_COMMUNICATION, p.ABLATE_ON_DEMAND,
                      p.ABLATE_EXECUTION_FEEDBACK, p.ABLATE_CRITIC)

    def tearDown(self):
        (p.ABLATE_COMMUNICATION, p.ABLATE_ON_DEMAND,
         p.ABLATE_EXECUTION_FEEDBACK, p.ABLATE_CRITIC) = self.saved

    def test_default_is_the_complete_system(self):
        self.assertEqual(p.ablation_label(), "complete")

    def test_each_flag_names_itself(self):
        for attr, name in (("ABLATE_COMMUNICATION", "communication"),
                           ("ABLATE_ON_DEMAND", "on_demand_activation"),
                           ("ABLATE_EXECUTION_FEEDBACK", "execution_feedback"),
                           ("ABLATE_CRITIC", "critic_feedback")):
            setattr(p, attr, True)
            self.assertEqual(p.ablation_label(), f"w/o {name}")
            setattr(p, attr, False)

    def test_no_critic_report_never_accepts_and_carries_no_phi(self):
        # This is what separates `w/o Critic Feedback` from B=0: the loop keeps
        # running (status is never PASS) but the Planner gets no diagnosis.
        r = p.no_critic_report({"repro_rc": None, "repo_test_rc": 1,
                                "repro_output": "boom", "test": "FAILED"},
                               "diff --git a/x b/x", round_idx=2,
                               baseline_repo_test_rc=0)
        self.assertEqual(r["status"], "FAIL")
        self.assertEqual(r["critic"], "disabled")
        for empty in ("failure_evidence", "suspected_causes",
                      "revision_guidance", "missing_components"):
            self.assertEqual(r[empty], [], empty)
        # The raw evidence still reaches the next round - the appendix says
        # execution stays enabled under this ablation.
        self.assertEqual(r["repo_test_rc"], 1)
        self.assertEqual(r["repro_output"], "boom")


class TestReleaseHygiene(unittest.TestCase):
    """The development copy carried AWS keys and absolute home paths. Neither may
    reach the release."""

    def test_no_credentials_and_no_absolute_home_paths(self):
        pkg = Path(__file__).resolve().parent.parent / "hermes"
        for f in sorted(pkg.glob("*.py")):
            text = f.read_text()
            self.assertNotIn("AK" + "IA", text, f)
            self.assertNotIn("/" + "Users/", text, f)


class TestUsageAccounting(unittest.TestCase):
    def test_localization_usage_is_counted_without_full_prompt_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = traj.Trajectory("one", out_root=tmp)
            record.log_llm_call(p.TRIAGE_PHASE, "small", "prompt", "yes", 0.5,
                                usage=p._Usage(20, 5), is_triage=True)
            record.log_llm_call("4_CRITIC", "large", "prompt", "yes", 1.0,
                                usage=p._Usage(50, 10))
            summary = record.close(None, 1, 2.0)
            self.assertEqual(summary["llm_calls"], 2)
            self.assertEqual(summary["triage_llm_calls"], 1)
            self.assertEqual(summary["prompt_tokens"], 70)
            self.assertEqual(summary["completion_tokens"], 15)
            self.assertEqual(summary["model_usage"]["small"]["calls"], 1)
            lines = (Path(tmp) / "one" / "trajectory.jsonl").read_text()
            self.assertNotIn('"phase": "1.1_BUG_LOCALIZATION_TRIAGE"', lines)

    def test_non_python_components_can_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("src/main.ts", "tests/test_main.py", "Dockerfile"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("some persistent component")
            before = p.GENERIC_COMPONENTS
            try:
                p.GENERIC_COMPONENTS = True
                self.assertEqual(set(p.dev_primitive_files(root)),
                                 {"src/main.ts", "tests/test_main.py", "Dockerfile"})
            finally:
                p.GENERIC_COMPONENTS = before


if __name__ == "__main__":
    unittest.main(verbosity=2)
