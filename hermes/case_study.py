#!/usr/bin/env python3
"""Render a paper-ready case study from a HERMES trajectory.

Usage:
    python -m hermes.case_study <trajectory-dir> [-o CASE_STUDY.md]

The output is organised by the stages of Fig. 1 in the paper rather than by
timestamp, so a reviewer can check each equation of Sec. 3 against a concrete
artifact: PLANNER (Eq. 2-3), Dev-Primitive collaboration (Eq. 4-5), EXECUTE
(Eq. 6), CRITIC (Eq. 7-8), re-planning (Eq. 9), and the held-out evaluation
that happens only after HERMES has terminated.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load(traj_dir: Path):
    recs = [json.loads(l) for l in (traj_dir / "trajectory.jsonl").open()
            if l.strip()]
    return recs


def by(recs, kind, round_idx=None):
    out = [r for r in recs if r.get("kind") == kind]
    if round_idx is not None:
        out = [r for r in out if r.get("round") == round_idx]
    return out


def one(recs, kind, round_idx=None):
    hits = by(recs, kind, round_idx)
    return hits[0] if hits else None


def fence(text, lang="", limit=None):
    text = (text or "").rstrip()
    if limit and len(text) > limit:
        text = text[:limit].rstrip() + "\n... (truncated)"
    return f"```{lang}\n{text}\n```"


def quote(text, limit=1500):
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " ..."
    return "\n".join("> " + l if l.strip() else ">" for l in text.split("\n"))


def emit_planner(out, recs):
    out.append("## 1. PLANNER - `Pi = PLANNER(q, R)`\n")

    arch = one(recs, "architecture")
    if arch:
        out.append("### 1.0 Issue analysis\n")
        out.append(quote(arch["text"], 2000) + "\n")

    loc = one(recs, "locate_summary")
    rank = one(recs, "rank")
    if loc:
        out.append("### 1.1 Bug localization\n")
        out.append(f"- components scanned: **{loc['n_files_scanned']}**")
        out.append(f"- flagged relevant: **{loc['n_relevant']}**")
        if rank and rank.get("after"):
            kept = rank["after"]
            out.append(f"- kept after ranking: **{len(kept)}**\n")
            out.append(fence("\n".join(kept)))
        out.append("")

    dec = one(recs, "task_decomposition")
    if dec:
        out.append("### 1.2 Task decomposition - the local objectives $x_i$\n")
        out.append("| component | role | objective $x_i$ | edits? |")
        out.append("|---|---|---|---|")
        for t in dec["tasks"]:
            task = t.get("task", "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| `{t['file']}` | {t.get('role','')} | {task} | "
                       f"{'yes' if t.get('changes') else 'observer'} |")
        out.append("")

    dep = one(recs, "dependency_analysis")
    if dep and dep.get("edges"):
        out.append("### 1.3 Dependency analysis\n")
        out.append("| from | relation | to | risk if changed in isolation |")
        out.append("|---|---|---|---|")
        for e in dep["edges"]:
            risk = e.get("risk", "").replace("|", "\\|").replace("\n", " ")
            out.append(f"| `{e['from']}` | {e.get('relation','')} | "
                       f"`{e['to']}` | {risk} |")
        out.append("")

    plan = one(recs, "plan")
    if plan:
        out.append("### 1.4 Edit plan\n")
        out.append(fence(plan["text"], limit=2500))
        out.append("")


def emit_reproduction(out, recs):
    rep = one(recs, "reproduction")
    cmd = one(recs, "repo_test_cmd")
    base = [r for r in by(recs, "baseline")][-1] if by(recs, "baseline") else None
    if not (rep or cmd or base):
        return
    out.append("## 1.5 Task-visible signals, validated before any edit\n")
    out.append("No FAIL\\_TO\\_PASS or PASS\\_TO\\_PASS test name is available to any "
               "component. The two signals below are constructed from the issue text "
               "and the repository itself, and each is only kept if it demonstrably "
               "works on the *unmodified* repository.\n")
    if rep:
        out.append("**Reproduction script** (written by the Planner from $q$ alone):\n")
        out.append(fence(rep["script"], "python", limit=2600))
    if cmd:
        out.append(f"\n**Repository test command**: `{cmd['command']}`\n")
        out.append(f"Rationale: {cmd.get('why','')}\n")
    if base:
        out.append("**Baseline on the unmodified repository**\n")
        out.append(f"- reproduction exit code: `{base.get('repro_rc')}` "
                   f"-> reproduces the issue: **{base.get('reproduces')}**")
        out.append(f"- repository test exit code: `{base.get('repo_test_rc')}` "
                   f"-> command actually ran tests: **{base.get('repo_test_ran')}**\n")
        if base.get("repro_output"):
            out.append("Reproduction output before the fix:\n")
            out.append(fence(base["repro_output"], limit=1200))
        out.append("")


def emit_round(out, recs, r):
    out.append(f"\n---\n\n# Round {r}\n")

    rnd = [x for x in by(recs, "round", r) if x.get("status") == "begin"]
    if rnd:
        out.append("Active set $\\mathcal{A}$ for this round:\n")
        for f in rnd[0]["files_in_scope"]:
            out.append(f"- `{f}` - {rnd[0].get('roles',{}).get(f,'')}")
        out.append("")

    negs = by(recs, "negotiate", r)
    if negs:
        out.append("## 2. Dev-Primitive collaboration - "
                   "`(a_i', m_i) = P_i(a_i, x_i, C_i)`\n")

        props = [n for n in negs if n.get("stage") == "propose"]
        if props:
            out.append("### Outgoing messages $m_i$\n")
            for n in props:
                out.append(f"**`{n['file']}`** ({n.get('role','')}) - "
                           f"its own objective: {n.get('intent','')}\n")
                for m in n.get("messages", []) or []:
                    msg = (m.get("message") or "").replace("\n", " ")
                    out.append(f"- $\\rightarrow$ `{m.get('to')}`: {msg}")
                out.append("")

        dl = [n for n in negs if n.get("stage") == "deliver"]
        if dl:
            out.append(f"### Delivery\n")
            out.append(f"{dl[0].get('n_messages')} messages routed to "
                       f"{len(dl[0].get('recipients') or [])} Dev-Primitives; "
                       "each primitive receives only the messages addressed to it, "
                       "so no component ever holds a global reasoning context.\n")

        rev = [n for n in negs if n.get("stage") == "revise"]
        if rev:
            out.append("### Effect of $\\mathcal{C}_i$ on each primitive's objective\n")
            for n in rev:
                out.append(f"**`{n['file']}`** - received from: "
                           + ", ".join("`%s`" % f for f in n.get("received") or [])
                           + f"; adjusted: **{n.get('adjusted')}**\n")
                out.append(f"- reasoning: {n.get('reasoning','')}")
                if n.get("intent_before") != n.get("intent_after"):
                    out.append(f"- before: {n.get('intent_before')}")
                    out.append(f"- **after**: {n.get('intent_after')}")
                out.append("")

    scope = one(recs, "self_modify_scope", r)
    if scope:
        out.append("### Writer / observer split after communication\n")
        out.append(f"- writers: {', '.join('`%s`' % f for f in scope['writers']) or '-'}")
        out.append(f"- observers: {', '.join('`%s`' % f for f in scope['observers']) or '-'}")
        if scope.get("promoted_by_negotiation"):
            out.append("- promoted from observer to writer **by a peer message**: "
                       + ", ".join("`%s`" % f for f in scope["promoted_by_negotiation"]))
        out.append("")

    edits = [e for e in by(recs, "edit", r) if e.get("applied")]
    if edits:
        out.append("### Local modifications $a_i \\rightarrow a_i'$\n")
        for e in edits:
            out.append(f"**`{e['file']}`** (attempt {e.get('attempt')}, "
                       f"{e.get('strategy')})\n")
            if e.get("diff"):
                out.append(fence(e["diff"], "diff", limit=1800))
            out.append("")

    rt = one(recs, "runtime", r)
    vf = one(recs, "verify", r)
    if rt or vf:
        out.append("## 3. EXECUTE - `o = EXECUTE(R') = (o_shell, o_test, o_runtime, o_trace)`\n")
        if rt:
            out.append(f"Environment image: `{rt.get('env_image')}`\n")
            if rt.get("shell"):
                out.append("`o_shell`:\n" + fence(rt["shell"], limit=900))
            if rt.get("runtime"):
                out.append("\n`o_runtime`:\n" + fence(rt["runtime"], limit=900))
            if rt.get("trace"):
                out.append("\n`o_trace`:\n" + fence(rt["trace"], limit=900))
        if vf:
            out.append(f"\n`o_test` - `{vf.get('test_cmd')}`, exit code "
                       f"`{vf.get('repo_test_rc')}` "
                       f"(baseline `{vf.get('baseline_repo_test_rc')}`); "
                       f"reproduction exit code `{vf.get('repro_rc')}`\n")
            if vf.get("output"):
                out.append(fence(vf["output"], limit=1400))
        out.append("")

    cr = one(recs, "critic", r)
    if cr:
        out.append(f"## 4. CRITIC - `v = CRITIC(q, Pi, R', o)` = **{cr['status']}**\n")
        if cr.get("overridden"):
            out.append("> The Critic returned PASS, but a non-zero reproduction "
                       "exit code forced FAIL: the Critic cannot override "
                       "deterministic execution failure.\n")
        if cr.get("evidence"):
            out.append("Evidence of success:\n")
            for e in cr["evidence"]:
                out.append(f"- {e}")
            out.append("")
        if cr.get("failure_evidence"):
            out.append("$e$ - observed failure evidence:\n")
            for e in cr["failure_evidence"]:
                ev = (e.get("evidence") or "").replace("\n", " ")
                out.append(f"- [{e.get('source')}] {ev}")
            out.append("")
        if cr.get("suspected_causes"):
            out.append("$c$ - suspected cause and components involved:\n")
            for c in cr["suspected_causes"]:
                out.append(f"- `{c.get('component')}`: {c.get('reason')}")
            out.append("")
        if cr.get("missing_components"):
            out.append("$c$ - components not yet activated:\n")
            for m in cr["missing_components"]:
                out.append(f"- {m}")
            out.append("")
        if cr.get("revision_guidance"):
            out.append("$u$ - revision guidance:\n")
            for g in cr["revision_guidance"]:
                out.append(f"- `{g.get('component')}`: {g.get('action')}")
            out.append("")
        if cr.get("summary"):
            out.append(f"Summary: {cr['summary']}\n")

    aset = one(recs, "active_set", r)
    if aset and (aset.get("activated") or aset.get("removed")):
        out.append("## 1. RE-PLAN - the active set $\\mathcal{A}$ itself changes\n")
        if aset.get("activated"):
            out.append("Newly activated Dev-Primitives (from the Critic's "
                       "`missing_components`):\n")
            for f in aset["activated"]:
                out.append(f"- `{f}`")
            out.append("")
        if aset.get("removed"):
            out.append("De-activated Dev-Primitives (no longer relevant):\n")
            for f in aset["removed"]:
                out.append(f"- `{f}`")
            out.append("")

    rp = one(recs, "replan", r)
    if rp:
        out.append("### Revised plan $\\Pi' = PLANNER(q, R', \\Pi, \\phi)$\n")
        out.append(fence(rp["text"], limit=2500))
        out.append("")


def emit_tail(out, recs):
    ho = one(recs, "held_out_evaluation")
    if ho:
        out.append("\n---\n\n## Held-out evaluation - run **after** HERMES terminated\n")
        out.append("This is the first and only time a benchmark test name is "
                   "executed. Its result is recorded, never fed back.\n")
        out.append(f"- resolved: **{ho.get('passed')}**")
        if ho.get("error"):
            out.append(f"- {ho['error']}")
        ts = ho.get("test_status") or {}
        if ts:
            out.append("\n| held-out test | status |")
            out.append("|---|---|")
            for k, v in list(ts.items())[:40]:
                out.append(f"| `{k}` | {v} |")
        out.append("")

    end = one(recs, "run_end")
    calls = by(recs, "llm_call")
    if end or calls:
        out.append("## Cost\n")
        if calls:
            pt = sum(c.get("prompt_tokens") or 0 for c in calls)
            ct = sum(c.get("completion_tokens") or 0 for c in calls)
            out.append(f"- LLM calls: **{len(calls)}**")
            out.append(f"- prompt tokens: **{pt}**, completion tokens: **{ct}**")
            roles = Counter(c.get("role") or c.get("agent") or "?" for c in calls)
            out.append("- calls by role: "
                       + ", ".join(f"{k} ({v})" for k, v in roles.most_common()))
        if end:
            out.append(f"- wall clock: **{end.get('t')}s**")
        out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    recs = load(traj_dir)
    start = one(recs, "run_start") or {}
    cfg = start.get("config") or {}

    out = []
    out.append(f"# Case study: `{start.get('instance_id')}`\n")
    out.append(f"- repository: `{start.get('repo')}` @ `{start.get('base_commit')}`")
    out.append(f"- model: `{cfg.get('model') or cfg.get('vllm')}`")
    out.append(f"- re-planning budget $B$: {cfg.get('max_iterations')}")
    out.append("")
    if start.get("problem_statement"):
        out.append("## Issue $q$\n")
        out.append(quote(start["problem_statement"], 2500) + "\n")

    emit_planner(out, recs)
    emit_reproduction(out, recs)

    rounds = sorted({r["round"] for r in by(recs, "round") if r.get("round")})
    for r in rounds:
        emit_round(out, recs, r)

    emit_tail(out, recs)

    text = "\n".join(out) + "\n"
    dest = Path(args.out) if args.out else traj_dir / "case_study.md"
    dest.write_text(text)
    print(f"wrote {dest} ({len(text)} chars)")


if __name__ == "__main__":
    main()
