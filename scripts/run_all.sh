#!/usr/bin/env bash
# The whole experimental pipeline, in the order it has to happen.
#
#   scripts/run_all.sh --smoke     # 3 instances, every stage, a few hours
#   scripts/run_all.sh             # the full protocol; days, and it is meant to
#                                  # be run under tmux/nohup
#   scripts/run_all.sh --stages main,ablations
#
# Stages:
#   dataset     download SWE-bench Verified, report which env images are visible
#   main        the complete system (Table `swebench-verified`)
#   ablations   the five component rows (Table `component-ablation`)
#   budget      B = 0..3 (Table `replanning-budget`)
#   backbones   one model per row (Tables `backbone-scale`, `scaling-cost`)
#   report      one table over every run directory found
#
# Environment images are not built here: building all of them is hours of CPU and
# ~200 GB, and it needs a Docker API socket. Run `scripts/build_env_images.py`
# once, before this. Without images the solver does a single edit round and
# reports resolved=false, which is not a HERMES number (docs/reproducibility.md).
source "$(dirname "$0")/_lib.sh"

STAGES=${STAGES:-dataset,main,ablations,budget,report}
SMOKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)  SMOKE=1; shift ;;
    --stages) STAGES=$2; shift 2 ;;
    *) echo "usage: $0 [--smoke] [--stages a,b,c]" >&2; exit 2 ;;
  esac
done

if [ "$SMOKE" = "1" ]; then
  # Three instances, one re-planning round: enough to exercise localization,
  # activation, communication, execution, the Critic and the grader, and to fail
  # loudly if the container runtime or the backend is misconfigured.
  INSTANCES=${INSTANCES:-"django__django-13512 django__django-11333 astropy__astropy-14365"}
  MAX_ITER=${MAX_ITER:-2}
  STAGES=${STAGES_SMOKE:-dataset,main,ablations,report}
  echo "smoke run: $INSTANCES (max-iterations=$MAX_ITER)"
fi

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
export INSTANCES MAX_ITER BACKEND_ARGS HERMES_RUNS

if has_stage dataset; then
  "$PY" "$HERMES_REPO/scripts/prepare_dataset.py"
fi

if has_stage main; then
  "$HERMES_REPO/scripts/run_swebench.sh"
fi

if has_stage ablations; then
  "$HERMES_REPO/scripts/run_ablations.sh"
fi

if has_stage budget; then
  "$HERMES_REPO/scripts/run_replanning_budget.sh"
fi

if has_stage backbones; then
  "$HERMES_REPO/scripts/run_backbone_sweep.sh"
fi

if has_stage report; then
  echo
  # Every run directory, whichever stages produced them.
  # Not `mapfile`: macOS still ships bash 3.2, which does not have it.
  dirs=()
  while IFS= read -r d; do dirs+=("$d"); done < <(
    find "$HERMES_RUNS" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  if [ ${#dirs[@]} -eq 0 ]; then
    echo "no run directories under $HERMES_RUNS"
  else
    hermes_report --title "All runs" --csv "$HERMES_RUNS/summary.csv" "${dirs[@]}"
  fi
fi
