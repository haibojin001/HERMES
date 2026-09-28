#!/usr/bin/env bash
# The re-planning budget sweep (paper Table `replanning-budget`): B = 0, 1, 2, 3, 5.
#
#   scripts/run_replanning_budget.sh
#   BUDGETS="0 3" INSTANCES="django__django-11333" scripts/run_replanning_budget.sh
#
# B is the number of re-planning rounds allowed *after* the first attempt, so the
# loop runs at most B+1 rounds and the flag is `--max-iterations $((B+1))`. B=0 is
# the single-shot system: activate, edit, execute, grade - no Critic feedback
# reaching a second round, because there is no second round.
#
# The interesting column is not only the resolved rate but the cost: each extra
# round is charged whether or not it changes the patch, and the aggregate table
# prints mean rounds actually used alongside mean tokens, which is what shows
# whether a larger budget was spent or merely offered.
source "$(dirname "$0")/_lib.sh"

BUDGETS=${BUDGETS:-"0 1 2 3 5"}

dirs=()
for b in $BUDGETS; do
  MAX_ITER=$((b + 1))
  hermes_solve "budget_B$b"
  hermes_eval "budget_B$b"
  dirs+=("$HERMES_RUNS/budget_B$b")
done

hermes_report --title "Re-planning budget" "${dirs[@]}"
