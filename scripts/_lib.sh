#!/usr/bin/env bash
# Shared plumbing for the experiment scripts. Source it, do not execute it.
#
# The one thing every sweep needs is that two configurations never write into
# each other's files. `--traj-suffix` separates the trajectories, but predictions
# and evaluation results are single files in `hermes/settings.py`, so a second
# configuration would append to the first one's predictions and the grader would
# score a mixture. Every configuration therefore gets its own run directory:
#
#     $HERMES_RUNS/<label>/predictions.jsonl
#     $HERMES_RUNS/<label>/eval/{results.jsonl,report.json}
#     $HERMES_RUNS/<label>/solve.log, eval.log
#     $HERMES_TRAJECTORIES/<instance>_<label>/
#
# That layout is also what `scripts/aggregate_results.py` reads, so the tables
# come out of the same files the runs wrote, with no bookkeeping in between.
#
# Knobs (all optional):
#   PYTHON          interpreter                      (default python3)
#   BACKEND_ARGS    e.g. "--vllm Qwen/Qwen3-8B --concurrency 32"
#   MAX_ITER        1 + B; the paper's default B=3 is 4
#   INSTANCES       space-separated ids; empty means all 500
#   HERMES_RUNS     where run directories go         (default $HERMES_HOME/runs)
#   NO_RESUME=1     re-solve instances already in this configuration's predictions
#   SKIP_EVAL=1     solve only, grade later

set -euo pipefail

HERMES_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$HERMES_REPO${PYTHONPATH:+:$PYTHONPATH}"

PY=${PYTHON:-python3}
BACKEND_ARGS=${BACKEND_ARGS:-}
MAX_ITER=${MAX_ITER:-4}
INSTANCES=${INSTANCES:-}
HERMES_RUNS=${HERMES_RUNS:-${HERMES_HOME:-$HERMES_REPO/hermes_data}/runs}

# hermes_solve <label> [extra pipeline flags ...]
hermes_solve() {
  local label=$1; shift
  local dir="$HERMES_RUNS/$label"
  mkdir -p "$dir"

  local resume=()
  [ "${NO_RESUME:-0}" = "1" ] && resume=(--no-resume)
  local only=()
  # shellcheck disable=SC2206
  [ -n "$INSTANCES" ] && only=(--instance $INSTANCES)

  echo "=== $(date +%F_%H:%M:%S) solve [$label]  max-iterations=$MAX_ITER  $* $BACKEND_ARGS"
  # shellcheck disable=SC2086
  HERMES_PREDICTIONS="$dir/predictions.jsonl" \
  HERMES_EVAL_DIR="$dir/eval" \
    "$PY" -u -m hermes.pipeline \
      --traj-suffix "_$label" --max-iterations "$MAX_ITER" \
      "${only[@]}" "${resume[@]}" $BACKEND_ARGS "$@" \
      2>&1 | tee -a "$dir/solve.log"
}

# hermes_eval <label> - Stage 6, after the loop has terminated for every instance
hermes_eval() {
  local label=$1
  local dir="$HERMES_RUNS/$label"
  [ "${SKIP_EVAL:-0}" = "1" ] && { echo "  (eval skipped)"; return 0; }
  [ -s "$dir/predictions.jsonl" ] || { echo "  no predictions for $label"; return 0; }

  echo "=== $(date +%F_%H:%M:%S) grade [$label]"
  HERMES_PREDICTIONS="$dir/predictions.jsonl" \
  HERMES_EVAL_DIR="$dir/eval" \
    "$PY" -u -m hermes.evaluate --predictions "$dir/predictions.jsonl" \
      2>&1 | tee -a "$dir/eval.log" | tail -25
}

hermes_report() {
  "$PY" "$HERMES_REPO/scripts/aggregate_results.py" "$@"
}
