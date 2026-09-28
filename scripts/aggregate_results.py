#!/usr/bin/env python3
"""Turn run directories into the rows of a paper table.

    python scripts/aggregate_results.py hermes_data/runs/*            # every run
    python scripts/aggregate_results.py --title "Component ablations" \
           hermes_data/runs/complete hermes_data/runs/no_critic
    python scripts/aggregate_results.py --per-instance --csv rows.csv results/trajectories
    python scripts/aggregate_results.py --price 3,15 hermes_data/runs/complete

Two kinds of argument are accepted:

  a run directory   - written by `scripts/_lib.sh`: `predictions.jsonl` plus
                      `eval/{results.jsonl,report.json}`. The resolved rate comes
                      from the grader, which is the number a paper reports.
  a trajectory dir  - any directory holding `<run>/summary.json`, e.g. the
                      `results/` in this repository. Then `resolved` is the
                      in-run held-out evaluation recorded by the trajectory,
                      which is the same check run by the same eval script.

Every number printed is read from a file some run wrote; nothing is carried over
from the paper, and a column with no data prints `-` instead of a zero. The
resolved rate carries the binomial standard error sqrt(p(1-p)/n) - with 500
instances a one-point difference is inside the noise of a single run, which is
worth knowing before reading a ranking into two adjacent rows.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes import settings


# ----------------------------------------------------------------- reading ----

def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a run killed mid-write leaves one truncated line
    return out


def graded(run_dir: Path) -> dict:
    """instance_id -> resolved, from the grader. Last line wins: the file is
    appended to, so a re-graded instance appears more than once."""
    out = {}
    for r in read_jsonl(run_dir / "eval" / "results.jsonl"):
        if "instance_id" in r:
            out[r["instance_id"]] = bool(r.get("resolved"))
    return out


def trajectory_stats(traj_dir: Path) -> dict:
    """The per-run facts, from summary.json plus the records that summary.json
    does not carry (active-set size, message count, configuration label)."""
    summary_path = traj_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        s = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {}

    stats = {
        "instance_id": s.get("instance_id", traj_dir.name),
        "resolved": bool(s.get("resolved")),
        "rounds": s.get("rounds"),
        "elapsed_s": s.get("elapsed_s"),
        "llm_calls": s.get("llm_calls"),
        "prompt_tokens": s.get("prompt_tokens"),
        "completion_tokens": s.get("completion_tokens"),
        "usage_missing_calls": s.get("usage_missing_calls"),
        "models_used": list((s.get("model_usage") or {}).keys()),
        "patch_chars": s.get("patch_chars", 0),
    }

    records = read_jsonl(traj_dir / "trajectory.jsonl")
    messages, active = 0, None
    for r in records:
        kind = r.get("kind")
        if kind == "run_start":
            cfg = r.get("config") or {}
            stats["configuration"] = cfg.get("configuration")
            stats["backbone"] = cfg.get("thinking_model")
        elif kind == "negotiate":
            # Eq. 5: one requirement addressed to one other component.
            messages += len(r.get("messages") or [])
        elif kind in ("plan", "active_set", "replan"):
            got = r.get("active") or r.get("files")
            if got:
                active = len(got)
    stats["messages"] = messages
    stats["active_set"] = active
    return stats


def collect(target: Path, traj_root: Path) -> dict:
    """One row's worth of raw per-instance records."""
    label = target.name
    preds = {p["instance_id"]: p.get("model_patch", "")
             for p in read_jsonl(target / "predictions.jsonl")
             if "instance_id" in p}

    if preds:                                   # a run directory
        scores = graded(target)
        rows = []
        for iid, patch in preds.items():
            st = trajectory_stats(traj_root / f"{iid}_{label}")
            if not st:                          # suffix unknown or not kept
                st = trajectory_stats(traj_root / iid)
            st["instance_id"] = iid
            st["patch_chars"] = len(patch)
            if iid in scores:
                st["resolved"] = scores[iid]
                st["graded"] = True
            else:
                st["graded"] = False
            rows.append(st)
        return {"label": label, "source": "run directory", "rows": rows}

    # a directory of trajectories
    rows = [trajectory_stats(d) for d in sorted(target.iterdir()) if d.is_dir()]
    rows = [r for r in rows if r]
    for r in rows:
        r["graded"] = True                      # in-run held-out evaluation
    if not rows:
        return {}
    return {"label": label, "source": "trajectories", "rows": rows}


# ---------------------------------------------------------------- reducing ----

def mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None


def summarize(row: dict, price=None) -> dict:
    rows = row["rows"]
    n = len(rows)
    scored = [r for r in rows if r.get("graded")]
    resolved = sum(1 for r in scored if r.get("resolved"))
    k = len(scored)
    p = resolved / k if k else None
    out = {
        "configuration": next((r["configuration"] for r in rows
                               if r.get("configuration")), row["label"]),
        "label": row["label"],
        "n": n,
        "graded": k,
        "resolved": resolved if k else None,
        "resolved_pct": p * 100 if p is not None else None,
        # Binomial standard error of a single run's rate. Not a confidence
        # interval over reruns: sampling noise only, temperature and container
        # flakiness are on top of this.
        "se_pct": math.sqrt(p * (1 - p) / k) * 100 if p is not None and k else None,
        "empty_patch_pct": (sum(1 for r in rows if not r.get("patch_chars"))
                            / n * 100) if n else None,
        "mean_rounds": mean(r.get("rounds") for r in rows),
        "mean_calls": mean(r.get("llm_calls") for r in rows),
        "mean_prompt_tokens": mean(r.get("prompt_tokens") for r in rows),
        "mean_completion_tokens": mean(r.get("completion_tokens") for r in rows),
        "mean_wall_s": mean(r.get("elapsed_s") for r in rows),
        "mean_active_set": mean(r.get("active_set") for r in rows),
        "mean_messages": mean(r.get("messages") for r in rows),
        "usage_missing_calls": sum(r.get("usage_missing_calls") or 0
                                   for r in rows),
        "backbone": next((r["backbone"] for r in rows if r.get("backbone")), None),
    }
    models = {model for r in rows for model in r.get("models_used", [])}
    if (price and out["mean_prompt_tokens"] is not None
            and out["usage_missing_calls"] == 0 and len(models) == 1):
        pin, pout = price
        out["mean_cost_usd"] = (out["mean_prompt_tokens"] / 1e6 * pin
                                + (out["mean_completion_tokens"] or 0) / 1e6 * pout)
    return out


# ---------------------------------------------------------------- printing ----

COLUMNS = [
    ("configuration", "configuration", "{}"),
    ("n", "n", "{:d}"),
    ("resolved_pct", "resolved %", "{:.1f}"),
    ("se_pct", "+/- SE", "{:.1f}"),
    ("empty_patch_pct", "no patch %", "{:.1f}"),
    ("mean_rounds", "rounds", "{:.2f}"),
    ("mean_active_set", "|A|", "{:.2f}"),
    ("mean_messages", "msgs", "{:.1f}"),
    ("mean_calls", "LLM calls", "{:.1f}"),
    ("mean_prompt_tokens", "prompt tok", "{:,.0f}"),
    ("mean_completion_tokens", "compl tok", "{:,.0f}"),
    ("mean_wall_s", "wall s", "{:.0f}"),
    ("mean_cost_usd", "$/inst", "{:.3f}"),
]


def render(summaries: list, title: str) -> str:
    cols = [c for c in COLUMNS
            if any(s.get(c[0]) is not None for s in summaries)]
    head = [c[1] for c in cols]
    body = [[c[2].format(s[c[0]]) if s.get(c[0]) is not None else "-"
             for c in cols] for s in summaries]
    width = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(head)]

    lines = [f"### {title}", ""] if title else []
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(head, width)) + " |")
    lines.append("|" + "|".join("-" * (w + 2) for w in width) + "|")
    for r in body:
        lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, width)) + " |")

    ungraded = [s for s in summaries if s["graded"] < s["n"]]
    if ungraded:
        lines += ["", "Not every prediction was graded - the resolved rate below "
                      "covers only the graded ones:"]
        for s in ungraded:
            lines.append(f"- {s['label']}: {s['graded']}/{s['n']} graded. "
                         f"Run `python -m hermes.evaluate` for the rest; "
                         f"instances whose environment image is missing are "
                         f"skipped by the grader, not failed.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="+", type=Path,
                    help="run directories, or directories of trajectories")
    ap.add_argument("--title", default="")
    ap.add_argument("--trajectories", type=Path, default=settings.TRAJ_ROOT,
                    help=f"trajectory root (default {settings.TRAJ_ROOT})")
    ap.add_argument("--price", default=None, metavar="IN,OUT",
                    help="USD per million prompt,completion tokens; adds a cost "
                         "column. Read docs/reproducibility.md first: the "
                         "recorder only counts calls that report usage.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the rows as CSV")
    ap.add_argument("--per-instance", action="store_true",
                    help="list every instance under each row")
    args = ap.parse_args()

    price = None
    if args.price:
        try:
            pin, pout = (float(x) for x in args.price.split(","))
            price = (pin, pout)
        except ValueError:
            return ap.error("--price wants two numbers, e.g. --price 3,15")

    collected, summaries = [], []
    for t in args.targets:
        if not t.is_dir():
            print(f"skipping {t}: not a directory", file=sys.stderr)
            continue
        row = collect(t, args.trajectories)
        if not row:
            print(f"skipping {t}: no predictions.jsonl and no summary.json "
                  f"underneath", file=sys.stderr)
            continue
        collected.append(row)
        summaries.append(summarize(row, price))

    if not summaries:
        return 1

    print(render(summaries, args.title))

    if args.per_instance:
        for row, s in zip(collected, summaries):
            print(f"\n#### {s['label']}")
            for r in sorted(row["rows"], key=lambda r: r["instance_id"]):
                mark = "resolved" if r.get("resolved") else (
                    "unresolved" if r.get("graded") else "ungraded")
                print(f"  {r['instance_id']:44s} {mark:10s} "
                      f"rounds={r.get('rounds', '-')} "
                      f"calls={r.get('llm_calls', '-')} "
                      f"wall={r.get('elapsed_s', '-')}s")

    if args.csv:
        keys = list(summaries[0].keys())
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summaries:
                w.writerow(s)
        print(f"\ncsv: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
