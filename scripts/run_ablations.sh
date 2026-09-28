#!/usr/bin/env bash
# The five rows of the component-ablation table (paper Table
# `component-ablation`): the complete system, then one mechanism removed at a
# time. Same instances, same backbone, same budget - one flag differs per row.
#
#   scripts/run_ablations.sh                                  # all 500, all rows
#   INSTANCES="django__django-13512 django__django-11333" scripts/run_ablations.sh
#   ROWS="complete no_critic" scripts/run_ablations.sh        # just two rows
#   BACKEND_ARGS="--vllm Qwen/Qwen3-8B --concurrency 32" scripts/run_ablations.sh
#
# Each row gets its own run directory and trajectory suffix, so nothing
# overwrites anything and any row can be re-run or rendered afterwards with
# `python -m hermes.case_study`.
#
# Note what is *not* here: `w/o Critic Feedback` is not `B=0`. Under the
# ablation the loop still runs its rounds and still executes the repository -
# the Critic simply returns no verdict and no diagnosis, so re-planning proceeds
# on raw evidence. `B=0` removes the rounds themselves and is swept separately by
# `scripts/run_replanning_budget.sh`.
source "$(dirname "$0")/_lib.sh"

ALL_ROWS="complete no_communication no_on_demand no_execution_feedback no_critic"
ROWS=${ROWS:-$ALL_ROWS}

flags_for() {
  case "$1" in
    complete)              echo "" ;;
    no_communication)      echo "--ablate-communication" ;;
    no_on_demand)          echo "--ablate-on-demand" ;;
    no_execution_feedback) echo "--ablate-execution-feedback" ;;
    no_critic)             echo "--ablate-critic" ;;
    *) echo "unknown row: $1" >&2; exit 2 ;;
  esac
}

dirs=()
for row in $ROWS; do
  # shellcheck disable=SC2046
  hermes_solve "$row" $(flags_for "$row")
  hermes_eval "$row"
  dirs+=("$HERMES_RUNS/$row")
done

hermes_report --title "Component ablations" "${dirs[@]}"
