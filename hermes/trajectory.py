"""
Trajectory recorder for HERMES.

Captures every step of the 6-phase loop to disk so a single run can be
inspected / published as a trajectory:

    trajectories/<instance_id>/
        trajectory.jsonl   # one record per step, machine readable
        trajectory.md      # rendered, human readable
        final_patch.diff   # the patch the run produced
        summary.json       # headline numbers

Usage (single-threaded per instance; phase-2 triage may be multi-threaded):

    traj = Trajectory(instance_id, repo="django/django", base_commit="abc",
                      problem_statement=problem, config={...})
    set_trajectory(traj)
    set_phase("1_UNDERSTAND")
    ...
    traj.close(resolved=True, rounds=2, elapsed_s=480.0, final_patch=patch)
    set_trajectory(None)

Every llm() call is recorded automatically once the solver's llm() wrapper
calls log_llm_call(). Prompts are truncated (head+tail) because edit prompts
embed whole source files; responses are always stored in full since they are
the interesting part. Raise PROMPT_HEAD_CHARS / PROMPT_TAIL_CHARS to keep more.
"""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Prompts embed full file contents; keep head+tail only. Responses are never cut.
PROMPT_HEAD_CHARS = 6000
PROMPT_TAIL_CHARS = 2000

# Phase 2 issues one LLM call per .py file (thousands). Their prompts/responses
# are logged compactly as "triage" records instead; set True for raw calls too.
LOG_TRIAGE_CALLS = False

_TRUNC_MARK = "\n\n... [%d chars omitted] ...\n\n"


def _clip(text, head=PROMPT_HEAD_CHARS, tail=PROMPT_TAIL_CHARS):
    """Return (clipped_text, original_len, was_clipped)."""
    if text is None:
        return None, 0, False
    n = len(text)
    if n <= head + tail:
        return text, n, False
    return text[:head] + (_TRUNC_MARK % (n - head - tail)) + text[-tail:], n, True


class Trajectory:
    def __init__(self, instance_id, repo=None, base_commit=None,
                 problem_statement=None, config=None, out_root=None):
        self.instance_id = instance_id
        self.out_dir = Path(out_root or "trajectories") / instance_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "trajectory.jsonl"
        self.path.write_text("")  # fresh run

        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.time()
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_missing_calls = 0
        self.triage_llm_calls = 0
        self.model_usage = {}

        self.log("run_start",
                 instance_id=instance_id,
                 repo=repo,
                 base_commit=base_commit,
                 started_at=datetime.now(timezone.utc).isoformat(),
                 config=config or {},
                 problem_statement=problem_statement)

    def log(self, kind, **fields):
        with self._lock:
            rec = {"seq": self._seq, "t": round(time.time() - self._t0, 3), "kind": kind}
            self._seq += 1
            rec.update(fields)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def records_of(self, kind):
        """Records of one kind logged so far, read back from the jsonl.

        The solver uses this to recover the concrete reasons an edit failed when
        it needs to feed them into the next round.
        """
        out = []
        with self._lock:
            try:
                text = self.path.read_text()
            except OSError:
                return out
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") == kind:
                out.append(rec)
        return out

    def log_llm_call(self, phase, model, prompt, response, latency_s, usage=None,
                     is_triage=False):
        clipped, n, was_clipped = _clip(prompt)
        pt = ct = None
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += pt or 0
            self.completion_tokens += ct or 0
            self.triage_llm_calls += int(is_triage)
            self.usage_missing_calls += int(pt is None or ct is None)
            model_stats = self.model_usage.setdefault(
                model, {"calls": 0, "prompt_tokens": 0,
                        "completion_tokens": 0, "usage_missing_calls": 0})
            model_stats["calls"] += 1
            model_stats["prompt_tokens"] += pt or 0
            model_stats["completion_tokens"] += ct or 0
            model_stats["usage_missing_calls"] += int(pt is None or ct is None)
        # Keep aggregate usage for localization without writing thousands of
        # large prompts and responses to each trajectory.
        if is_triage and not LOG_TRIAGE_CALLS:
            return
        self.log("llm_call", phase=phase, model=model,
                 prompt=clipped, prompt_chars=n, prompt_truncated=was_clipped,
                 response=response, response_chars=len(response or ""),
                 latency_s=round(latency_s, 2),
                 prompt_tokens=pt, completion_tokens=ct)

    def close(self, resolved, rounds, elapsed_s, final_patch=""):
        summary = {
            "instance_id": self.instance_id,
            "resolved": resolved,
            "rounds": rounds,
            "elapsed_s": round(elapsed_s, 1),
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "triage_llm_calls": self.triage_llm_calls,
            "usage_missing_calls": self.usage_missing_calls,
            "model_usage": self.model_usage,
            "patch_chars": len(final_patch or ""),
        }
        self.log("run_end", final_patch=final_patch, **summary)
        (self.out_dir / "final_patch.diff").write_text(final_patch or "")
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        try:
            md = render_markdown(self.path)
            (self.out_dir / "trajectory.md").write_text(md)
        except Exception as e:  # never let rendering kill a run
            (self.out_dir / "trajectory.md").write_text(f"render failed: {e}\n")
        return summary


# ------------------------------------------------------------------
# Ambient current-trajectory handle (solver runs one instance at a time)
# ------------------------------------------------------------------
_CURRENT = None
_PHASE = "0_INIT"


def set_trajectory(traj):
    global _CURRENT
    _CURRENT = traj


def current():
    return _CURRENT


def set_phase(phase):
    global _PHASE
    _PHASE = phase
    if _CURRENT:
        _CURRENT.log("phase", phase=phase)


def phase():
    return _PHASE


def log(kind, **fields):
    if _CURRENT:
        return _CURRENT.log(kind, phase=_PHASE, **fields)
    return None


def records_of(kind):
    return _CURRENT.records_of(kind) if _CURRENT else []


def log_llm_call(model, prompt, response, latency_s, usage=None, is_triage=False):
    if not _CURRENT:
        return
    _CURRENT.log_llm_call(_PHASE, model, prompt, response, latency_s, usage,
                          is_triage=is_triage)


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------
def _fence(text, lang=""):
    text = (text or "").rstrip()
    if not text:
        return "_(empty)_\n"
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{lang}\n{text}\n{fence}\n"


def _tail(text, n):
    text = text or ""
    if len(text) <= n:
        return text
    return f"... [{len(text) - n} chars omitted] ...\n" + text[-n:]


def render_markdown(jsonl_path, test_output_chars=6000):
    records = []
    for line in Path(jsonl_path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))

    out = []
    w = out.append
    triage_relevant = []
    triage_seen = 0

    for r in records:
        k = r["kind"]
        t = r.get("t", 0)

        if k == "run_start":
            w(f"# Trajectory: {r['instance_id']}\n")
            w(f"- repo: `{r.get('repo')}`")
            w(f"- base_commit: `{r.get('base_commit')}`")
            w(f"- started: {r.get('started_at')}")
            cfg = r.get("config") or {}
            for key in sorted(cfg):
                w(f"- {key}: `{cfg[key]}`")
            w("\n## Problem statement\n")
            w(_fence(r.get("problem_statement")))

        elif k == "phase":
            w(f"\n---\n\n## [t={t:.0f}s] PHASE {r['phase']}\n")

        elif k == "architecture":
            w("### Architecture summary (Phase 1 output)\n")
            w(_fence(r.get("text")))

        elif k == "triage":
            triage_seen += 1
            if r.get("relevant"):
                triage_relevant.append(r)

        elif k == "locate_summary":
            w(f"Scanned **{r.get('n_files_scanned', triage_seen)}** source files, "
              f"one LLM relevance judgement each. "
              f"**{r.get('n_relevant', len(triage_relevant))}** said `relevant: true`:\n")
            for tr in triage_relevant:
                lines = tr.get("lines") or []
                w(f"- `{tr['file']}` - lines {lines} - {tr.get('reason', '')}")
            w("")

        elif k == "rank":
            w(f"\n### Ranking (over {len(r.get('before', []))} candidates "
              f"-> keep {len(r.get('after', []))})\n")
            for f_ in r.get("after", []):
                w(f"- `{f_}`")
            w("")

        elif k == "task_decomposition":
            w("### Task decomposition (planner sub-stage 1.2)\n")
            w("| file | role | assigned an edit | sub-task | why this file |")
            w("|---|---|---|---|---|")
            for t in r.get("tasks") or []:
                w(f"| `{t.get('file')}` | {t.get('role')} "
                  f"| {'yes' if t.get('changes', True) else 'no (observer)'} "
                  f"| {t.get('task','')} | {t.get('why','')} |")
            w("")
            if r.get("observers"):
                w(f"Observers - in scope for inter-file communication but given no "
                  f"edit: {r['observers']}\n")
            if r.get("parse_failed"):
                w("> _structured output failed to parse; fell back to one task "
                  "per located file._\n")

        elif k == "dependency_analysis":
            w("### Dependency analysis (planner sub-stage 1.3)\n")
            edges = r.get("edges") or []
            if edges:
                for e in edges:
                    w(f"- `{e.get('from')}` -> `{e.get('to')}` "
                      f"(**{e.get('relation')}**): {e.get('risk','')}")
            else:
                w("_No dependencies between the sub-tasks._")
            w("")
            if r.get("order"):
                w(f"Apply order: {' -> '.join('`%s`' % f for f in r['order'])}\n")
            if r.get("missing_files"):
                w(f"Files added to scope because a dependency requires them: "
                  f"{r['missing_files']}\n")

        elif k == "plan":
            label = ("Edit planning - re-plan from the critic report"
                     if r.get("from_critic") else "Edit planning (planner sub-stage 1.4)")
            rnd = f" (round {r['round']})" if r.get("round") else ""
            w(f"### {label}{rnd}\n")
            if r.get("summary"):
                w(f"{r['summary']}\n")
            edits = r.get("edits") or {}
            if edits:
                for f_, e in edits.items():
                    w(f"- **`{f_}`** [{e.get('role','')}] - target "
                      f"`{e.get('target','')}`")
                    w(f"  - change: {e.get('change','')}")
                    if e.get("must_not_break"):
                        w(f"  - must not break: {e['must_not_break']}")
                w("")
            else:
                w(_fence(r.get("text")))

        elif k == "coordination":
            # Superseded by dependency_analysis; kept so pre-refactor
            # trajectories still render.
            w("### Coordination check\n")
            w(f"- LLM proposed missing files: {r.get('missing_files')}")
            w(f"- added to scope: {r.get('added')}\n")

        elif k == "negotiate":
            stage = r.get("stage")
            if stage == "propose":
                w(f"#### `{r['file']}` announces its intent\n")
                w(_fence(r.get("intent")))
                msgs = r.get("messages") or []
                if msgs:
                    w(f"`{r['file']}` sends {len(msgs)} message(s) to other file agents:\n")
                    for m in msgs:
                        w(f"- **`{r['file']}` -> `{m['to']}`**: {m['message']}")
                    w("")
                else:
                    w(f"_`{r['file']}` sends no messages - its edit is self-contained._\n")
            elif stage == "deliver":
                w(f"#### Message routing - {r.get('n_messages')} message(s) "
                  f"delivered to {r.get('recipients')}\n")
                for m in r.get("messages") or []:
                    w(f"- `{m['from']}` -> `{m['to']}`: {m['message']}")
                w("")
            elif stage == "revise":
                verdict = "**ADJUSTED** its plan" if r.get("adjusted") else "kept its plan"
                w(f"#### `{r['file']}` read its inbox (from {r.get('received')}) "
                  f"and {verdict}\n")
                w(f"Its reasoning:\n")
                w(_fence(r.get("reasoning")))
                if r.get("adjusted"):
                    w("Intent before:\n")
                    w(_fence(r.get("intent_before")))
                    w("Intent after negotiation:\n")
                    w(_fence(r.get("intent_after")))
            elif stage == "settled":
                w("#### Negotiated assignments going into EDIT\n")
                for fn, it in (r.get("intents") or {}).items():
                    w(f"- **`{fn}`**: {it}")
                w("")
            elif stage == "skipped":
                w(f"_Negotiation skipped: {r.get('reason')}._\n")
            else:
                w(f"_Negotiation {stage}: {r.get('error', '')}_\n")

        elif k == "round":
            if r.get("status") == "begin":
                w(f"\n---\n\n## [t={t:.0f}s] ROUND {r['round']} - files in scope: "
                  f"{r.get('files_in_scope')}\n")
            else:
                w(f"\n**Round {r['round']} end** - resolved={r.get('resolved')}, "
                  f"files edited={r.get('edited')}\n")

        elif k == "self_modify_scope":
            w(f"### Who self-modifies in round {r.get('round')}\n")
            w(f"- writers: {r.get('writers')}")
            w(f"- observers (no edit): {r.get('observers')}")
            if r.get("promoted_by_negotiation"):
                w(f"- **promoted to writer by inter-file communication**: "
                  f"{r['promoted_by_negotiation']}")
            w("")

        elif k == "edit":
            status = "applied" if r.get("applied") else f"FAILED ({r.get('error')})"
            role = f" [{r['role']}]" if r.get("role") else ""
            w(f"### File agent `{r['file']}`{role} - round {r.get('round')}, "
              f"attempt {r.get('attempt')}, strategy `{r.get('strategy')}` -> {status}\n")
            w("Self-modification code the file agent emitted:\n")
            w(_fence(r.get("code"), "python"))
            if r.get("diff"):
                w("Resulting diff for this file:\n")
                w(_fence(r["diff"], "diff"))

        elif k == "reproduction":
            w("### Reproduction script written from the issue text\n")
            w(_fence(r.get("script"), "python"))

        elif k == "repo_test_cmd":
            w("### Repository test command chosen as the regression signal\n")
            w(f"- command: `{r.get('command')}` "
              f"({'accepted' if r.get('accepted') else 'rejected'})")
            w(f"- rationale: {r.get('why') or r.get('reason') or '-'}\n")

        elif k == "baseline":
            ok = r.get("reproduces")
            w(f"### Baseline on the unmodified repository (attempt "
              f"{r.get('attempt')})\n")
            w(f"- reproduction exit code: `{r.get('repro_rc')}` - "
              f"{'reproduces the bug PASS' if ok else 'does NOT reproduce FAIL'}")
            w(f"- repository tests `{r.get('repo_test_cmd')}` exit code: "
              f"`{r.get('repo_test_rc')}` - "
              f"{'ran tests PASS' if r.get('repo_test_ran') else 'ran no tests FAIL'}\n")
            if r.get("repro_output"):
                w("Reproduction output before any edit:\n")
                w(_fence(_tail(r.get("repro_output"), 2000)))
            if r.get("test_output"):
                w("Repository test output before any edit:\n")
                w(_fence(_tail(r.get("test_output"), 1500)))

        elif k == "verify":
            w(f"### Execution observations - round {r.get('round')}\n")
            w(f"- reproduction script exit code: `{r.get('repro_rc')}`")
            w(f"- repository tests `{r.get('test_cmd')}` exit code: "
              f"`{r.get('repo_test_rc')}` (baseline "
              f"`{r.get('baseline_repo_test_rc')}`)\n")
            if r.get("repro_output"):
                w("Reproduction output:\n")
                w(_fence(_tail(r.get("repro_output"), 2500)))
            if r.get("output"):
                w(f"Repository test output (tail, full length "
                  f"{r.get('output_chars')} chars):\n")
                w(_fence(_tail(r.get("output"), test_output_chars)))

        elif k == "debug_script":
            w(f"### Diagnostic script - round {r.get('round')}\n")
            w(_fence(r.get("script"), "python"))
            w("Diagnostic output:\n")
            w(_fence(_tail(r.get("output"), 3000)))

        elif k == "runtime":
            w(f"### Execution environment - runtime (round {r.get('round')})\n")
            w(f"- container image: `{r.get('env_image')}`")
            if r.get("output_chars") is not None:   # pre-refactor records
                w(f"- captured {r.get('output_chars')} chars of stdout/stderr")
            if r.get("runtime"):
                w("\n`o_runtime` - runtime errors and program behaviour:\n")
                w(_fence(_tail(r.get("runtime"), 2000)))
            if r.get("trace"):
                w("`o_trace` - execution traces:\n")
                w(_fence(_tail(r.get("trace"), 2000)))
            if r.get("shell"):
                w("`o_shell` - command output and execution status:\n")
                w(_fence(_tail(r.get("shell"), 1500)))
            w("")

        elif k == "patch":
            w(f"### Refined patch after round {r.get('round')} "
              f"({r.get('chars')} chars, {r.get('files_changed')} file(s) edited, "
              f"critic {'PASS' if r.get('passed') else 'FAIL'})\n")
            w(_fence(r.get("diff"), "diff"))

        elif k == "critic":
            w(f"### Critic - round {r.get('round')}: **{r.get('status', 'FAIL')}**\n")
            w(f"- reproduction exit code: `{r.get('repro_rc')}`")
            w(f"- repository tests exit code: `{r.get('repo_test_rc')}` "
              f"(baseline `{r.get('baseline_repo_test_rc')}`, "
              f"regression: {bool(r.get('regressed'))})\n")
            if r.get("overridden"):
                w(f"> **Overridden:** {r['overridden']}\n")
            if r.get("status") == "PASS":
                w("**Evidence of success:**\n")
                for e in r.get("evidence") or []:
                    w(f"- {e}")
                w("")
            else:
                w("**`e` - observed failure evidence:**\n")
                for e in r.get("failure_evidence") or []:
                    w(f"- `[{e.get('source','?')}]` {e.get('evidence','')}")
                if not r.get("failure_evidence"):
                    w("- _(none quoted)_")
                w("\n**`c` - suspected cause and components involved:**\n")
                for c in r.get("suspected_causes") or []:
                    w(f"- `{c.get('component','?')}` - {c.get('reason','')}")
                if not r.get("suspected_causes"):
                    w("- _(none named)_")
                if r.get("missing_components"):
                    w("\nComponents implicated but **not yet activated**: "
                      + ", ".join(f"`{m}`" for m in r["missing_components"]))
                w("\n**`u` - revision guidance:**\n")
                for g in r.get("revision_guidance") or []:
                    w(f"- `{g.get('component','?')}` -> {g.get('action','')}")
                if not r.get("revision_guidance"):
                    w("- _(none given)_")
                w("")
            w(f"Summary: {r.get('summary','')}\n")
            if r.get("parse_failed"):
                w("> _critic's structured output failed to parse._\n")

        elif k == "active_set":
            w(f"### Active set revision - after round {r.get('round')}\n")
            if r.get("activated"):
                w("- **activated** by critic feedback: "
                  + ", ".join(f"`{f}`" for f in r["activated"]))
            if r.get("removed"):
                w("- **deactivated** as no longer relevant: "
                  + ", ".join(f"`{f}`" for f in r["removed"]))
            if not r.get("activated") and not r.get("removed"):
                w("- unchanged")
            w(f"- active set is now: "
              + ", ".join(f"`{f}`" for f in r.get("active") or []) + "\n")

        elif k == "held_out_evaluation":
            w("### Held-out evaluation - run only after the trajectory "
              "terminated\n")
            w(f"**resolved: {r.get('passed')}**"
              + (f" (`{r['error']}`)" if r.get("error") else "") + "\n")
            for t, v in (r.get("test_status") or {}).items():
                w(f"- {'PASS' if v == 'PASSED' else 'FAIL'} `{t}` -> {v}")
            w(f"\n> {r.get('note','')}\n")
            w(_fence(_tail(r.get("output"), test_output_chars)))

        elif k == "feedback":
            # Superseded by the critic record; kept for pre-refactor trajectories.
            w(f"### Feedback assembled for round {r.get('round') + 1 if r.get('round') else '?'}\n")
            w("Own diff shown back to the file agents:\n")
            w(_fence(r.get("our_diff"), "diff"))
            w("Extracted test errors:\n")
            w(_fence(_tail(r.get("test_errors"), 4000)))
            if r.get("test_expectations"):
                w("Test expectations:\n")
                w(_fence(r["test_expectations"]))

        elif k == "replan":
            w(f"### Re-plan after round {r.get('round')} failure"
              f"{' (failure class `%s`)' % r['attribution'] if r.get('attribution') else ''}\n")
            if r.get("active"):
                w("Active Dev-Primitives for the next round: "
                  + ", ".join(f"`{f}`" for f in r["active"]) + "\n")
            w(_fence(r.get("text")))

        elif k == "note":
            w(f"> {r.get('text')}\n")

        elif k == "run_end":
            w(f"\n---\n\n## [t={t:.0f}s] RUN END\n")
            w(f"- resolved: **{r.get('resolved')}**")
            w(f"- rounds: {r.get('rounds')}")
            w(f"- elapsed: {r.get('elapsed_s')}s")
            w(f"- llm calls: {r.get('llm_calls')} "
              f"({r.get('prompt_tokens')} prompt / {r.get('completion_tokens')} completion tokens)")
            w("\n### Final patch\n")
            w(_fence(r.get("final_patch"), "diff"))

    return "\n".join(out) + "\n"


# ------------------------------------------------------------------
# Appendix rendering: the control flow only, short enough to paste
# into a paper. Drops prompts, raw LLM calls and full test logs.
# ------------------------------------------------------------------
def _short(text, n):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rstrip() + " [...]"


def render_appendix(jsonl_path, problem_chars=700, plan_chars=1100,
                    msg_chars=320, reason_chars=300, replan_chars=900,
                    errors_chars=700):
    records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
    out = []
    w = out.append

    for r in records:
        k, t = r["kind"], r.get("t", 0)

        if k == "run_start":
            w(f"# Appendix: end-to-end trajectory - `{r['instance_id']}`\n")
            w(f"Repository `{r.get('repo')}` at base commit `{r.get('base_commit')}`.\n")
            w("**Issue (abridged):**\n")
            w(f"> {_short(r.get('problem_statement'), problem_chars)}\n")

        elif k == "architecture":
            w("\n## 1. PLANNING - Issue Analysis & Planning\n")
            w("### 1.0 Issue analysis\n")
            w(f"> {_short(r.get('text'), 600)}\n")

        elif k == "locate_summary":
            w("### 1.1 Bug localization\n")
            w(f"Every one of the **{r.get('n_files_scanned')}** files is asked, "
              f"as its own agent, whether it is implicated in this issue. "
              f"**{r.get('n_relevant')}** answer yes.\n")

        elif k == "rank":
            w(f"After ranking, **{len(r.get('after', []))} files are activated** as "
              f"Dev-Primitives:\n")
            for f_ in r.get("after", []):
                w(f"- `{f_}`")
            w("")

        elif k == "task_decomposition":
            w("### 1.2 Task decomposition\n")
            w("| Dev-Primitive | role | sub-task it owns |")
            w("|---|---|---|")
            for t in r.get("tasks") or []:
                task = (_short(t.get('task'), 200) if t.get("changes", True)
                        else "_observer - no edit assigned_")
                w(f"| `{t.get('file')}` | {t.get('role')} | {task} |")
            w("")

        elif k == "dependency_analysis":
            w("### 1.3 Dependency analysis\n")
            for e in r.get("edges") or []:
                w(f"- `{e.get('from')}` -> `{e.get('to')}` (**{e.get('relation')}**): "
                  f"{_short(e.get('risk'), 200)}")
            if not r.get("edges"):
                w("_The planner finds no coupling between the sub-tasks._")
            w("")
            if r.get("missing_files"):
                w(f"_Added to scope as a required dependency: {r['missing_files']}._\n")

        elif k == "plan":
            if r.get("from_critic"):
                w(f"\n### 1. RE-PLAN - edit planning from the critic report\n")
            else:
                w("### 1.4 Edit planning\n")
            if r.get("summary"):
                w(f"> {_short(r['summary'], plan_chars)}\n")
            for f_, e in (r.get("edits") or {}).items():
                w(f"- **`{f_}`** [{e.get('role','')}] - {_short(e.get('change'), 260)}")
            if not r.get("edits"):
                w(f"> {_short(r.get('text'), plan_chars)}")
            w("")

        elif k == "round":
            if r.get("status") == "begin":
                w(f"\n---\n\n# ROUND {r['round']}  _(t={t:.0f}s)_\n")
            elif r.get("critic"):
                w(f"\n_Round {r['round']}: {r.get('edited')} files edited, "
                  f"critic={r['critic']}._\n")
            else:
                # Pre-refactor runs recorded the benchmark verdict per round.
                w(f"\n_Round {r['round']}: {r.get('edited')} files edited, "
                  f"resolved={r.get('resolved')}._\n")

        elif k == "negotiate":
            stage = r.get("stage")
            if stage == "propose":
                msgs = r.get("messages") or []
                role = f" [{r['role']}]" if r.get("role") else ""
                w(f"**`{r['file']}`**{role} - intent: {_short(r.get('intent'), 260)}")
                if msgs:
                    for m in msgs:
                        w(f"  - mail **-> `{m['to']}`**: {_short(m['message'], msg_chars)}")
                else:
                    w("  - _(sends no messages; judges its edit self-contained)_")
                w("")
            elif stage == "deliver":
                n = r.get("n_messages", 0)
                w(f"\n### 2. COLLABORATION - inter-file communication: "
                  f"{n} message(s) exchanged\n")
                if n == 0:
                    w("_No agent believed its edit affected any other file._\n")
            elif stage == "revise":
                tag = "**adjusts its plan**" if r.get("adjusted") else "keeps its plan"
                w(f"**`{r['file']}`** reads its inbox (from {r.get('received')}) and {tag}: "
                  f"{_short(r.get('reasoning'), reason_chars)}")
                if r.get("adjusted"):
                    w(f"  - before: {_short(r.get('intent_before'), 200)}")
                    w(f"  - after:  {_short(r.get('intent_after'), 200)}")
                w("")
            elif stage == "skipped":
                w(f"\n_Negotiation skipped: {r.get('reason')}._\n")

        elif k == "self_modify_scope":
            if r.get("promoted_by_negotiation"):
                w(f"_Promoted to writer by inter-file communication: "
                  f"{r['promoted_by_negotiation']}._\n")

        elif k == "edit":
            if not r.get("applied"):
                w(f"- `{r['file']}` attempt {r.get('attempt')} failed: {r.get('error')}")
                continue
            w(f"\n#### `{r['file']}` self-modifies\n")
            w(_fence(r.get("diff"), "diff"))

        elif k == "reproduction":
            w("\n### 1.5 Reproduction script constructed from the task\n")
            w(_fence(r.get("script"), "python"))

        elif k == "repo_test_cmd":
            if r.get("accepted"):
                w(f"_Repository test command chosen as the regression signal:_ "
                  f"`{r.get('command')}` - {_short(r.get('why'), 200)}\n")
            else:
                w(f"_No repository test command accepted ({r.get('reason')})._\n")

        elif k == "baseline":
            w(f"_Baseline check on the unmodified repository (attempt "
              f"{r.get('attempt')}): reproduction exits `{r.get('repro_rc')}` "
              f"({'reproduces the bug' if r.get('reproduces') else 'does not reproduce'}), "
              f"repository tests exit `{r.get('repo_test_rc')}` "
              f"({'ran tests' if r.get('repo_test_ran') else 'ran no tests'})._\n")

        elif k == "verify":
            w(f"\n### 3. EXECUTION - run & test (round {r.get('round')})\n")
            w(f"- reproduction script: exit `{r.get('repro_rc')}`")
            w(f"- repository tests `{_short(r.get('test_cmd'), 120)}`: exit "
              f"`{r.get('repo_test_rc')}` (baseline "
              f"`{r.get('baseline_repo_test_rc')}`)\n")
            if r.get("repro_output"):
                w("`o_test` - reproduction output:\n")
                w(_fence(_short(r.get("repro_output"), 700)))

        elif k == "debug_script":
            w("\n### 3. EXECUTION - shell/terminal diagnostic\n")
            w(f"Output:\n")
            w(_fence(_short(r.get("output"), 600)))

        elif k == "critic":
            w(f"\n### 4. CRITIC - analyze & feedback (round {r.get('round')}): "
              f"**{r.get('status', 'FAIL')}**\n")
            if r.get("overridden"):
                w(f"> {r['overridden']}\n")
            if r.get("status") == "PASS":
                for e in (r.get("evidence") or [])[:4]:
                    w(f"- {_short(e, 220)}")
                w("")
            else:
                w("**`e` observed failure evidence:**")
                for e in (r.get("failure_evidence") or [])[:4]:
                    w(f"- `[{e.get('source','?')}]` {_short(e.get('evidence'), 220)}")
                if not r.get("failure_evidence"):
                    w("- _(none quoted)_")
                w("\n**`c` suspected cause and components:**")
                for c in (r.get("suspected_causes") or [])[:4]:
                    w(f"- `{c.get('component','?')}` - {_short(c.get('reason'), 220)}")
                if not r.get("suspected_causes"):
                    w("- _(none named)_")
                if r.get("missing_components"):
                    w(f"- not yet activated: "
                      + ", ".join(f"`{m}`" for m in r["missing_components"]))
                w("\n**`u` revision guidance:**")
                for g in (r.get("revision_guidance") or [])[:5]:
                    w(f"- `{g.get('component','?')}` -> {_short(g.get('action'), 220)}")
                if not r.get("revision_guidance"):
                    w("- _(none given)_")
                w("")
            w(f"Summary: {_short(r.get('summary'), 300)}\n")

        elif k == "active_set":
            parts = []
            if r.get("activated"):
                parts.append("activated " + ", ".join(f"`{f}`" for f in r["activated"]))
            if r.get("removed"):
                parts.append("deactivated " + ", ".join(f"`{f}`" for f in r["removed"]))
            w(f"\n### 1. RE-PLAN - active set A changes: "
              f"{'; '.join(parts) if parts else 'unchanged'}\n")
            w("Active Dev-Primitives: "
              + ", ".join(f"`{f}`" for f in r.get("active") or []) + "\n")

        elif k == "held_out_evaluation":
            w("\n### HELD-OUT EVALUATION - after termination, not fed back\n")
            bad = [f"{n} -> {s}" for n, s in (r.get("test_status") or {}).items()
                   if s != "PASSED"]
            w(f"- **resolved: {r.get('passed')}**"
              + (f" (`{r['error']}`)" if r.get("error") else ""))
            if bad:
                w("- non-passing held-out tests:")
                for b in bad[:8]:
                    w(f"  - `{b}`")
            w("")

        elif k == "patch":
            if not r.get("passed"):
                continue
            w(f"\n### 5. REFINED PATCH (round {r.get('round')})\n")
            w(_fence(r.get("diff"), "diff"))

        elif k == "feedback":
            # Pre-refactor trajectories only; superseded by the critic record.
            w(f"\n### FEEDBACK returned to the file agents\n")
            w(_fence(_short(r.get("test_errors"), errors_chars)))

        elif k == "replan":
            if r.get("attribution") is None:
                # Pre-refactor run: the replan text lives only in this record.
                w(f"\n### REPLAN\n")
                w(f"> {_short(r.get('text'), replan_chars)}\n")
            else:
                # The plan record just above already rendered the new edit spec.
                w(f"\n_Re-plan issued after round {r.get('round')}._\n")

        elif k == "run_end":
            w(f"\n---\n\n# OUTCOME  _(t={t:.0f}s)_\n")
            w(f"- **resolved: {r.get('resolved')}** after {r.get('rounds')} round(s)")
            w(f"- {r.get('llm_calls')} LLM calls, "
              f"{r.get('prompt_tokens')} prompt / {r.get('completion_tokens')} completion tokens")
            w(f"- wall clock {r.get('elapsed_s')}s\n")
            w("### Final patch\n")
            w(_fence(r.get("final_patch"), "diff"))

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    appendix = "--appendix" in sys.argv[1:]
    if len(args) != 1:
        print("usage: python3 trajectory.py [--appendix] <trajectory.jsonl>")
        raise SystemExit(1)
    src = Path(args[0])
    if appendix:
        dst = src.parent / "appendix.md"
        dst.write_text(render_appendix(src))
    else:
        dst = src.parent / "trajectory.md"
        dst.write_text(render_markdown(src))
    print(f"wrote {dst}")
