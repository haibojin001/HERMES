#!/usr/bin/env bash
# Backbone sweep (paper Tables `backbone-scale`, `scaling-cost`, `all-models`).
# Same pipeline, same budget, same instances; one backbone configuration per row.
#
#   scripts/run_backbone_sweep.sh                 # the homogeneous rows
#   scripts/run_backbone_sweep.sh --roles         # the per-role allocation rows
#   MODELS="qwen3_8b:--vllm Qwen/Qwen3-8B --concurrency 32" \
#     scripts/run_backbone_sweep.sh
#
# Two kinds of row, because the paper reports two things:
#
#   homogeneous   one model in every reasoning slot. MODELS is a list of
#                 `label:flags`, newline- or semicolon-separated.
#   per role      the Planner, the Dev-Primitives and the Critic on different
#                 models, which is what shows that strengthening the Critic buys
#                 more than strengthening the Planner. ROLES is a list of
#                 `label:flags`, using --planner-model / --primitive-model /
#                 --critic-model. An unset slot follows --model, so
#                 `--critic-model X` alone is the "scale the Critic only" row.
#
# Any backend the pipeline understands can appear in the flags:
#
#   hosted (litellm id):   --model bedrock/us.anthropic.claude-sonnet-4-...
#   served locally:        --vllm Qwen/Qwen3-8B --concurrency 32
#   ollama:                --ollama qwen3-8b-32k
#
# The lists below are shapes, not claims: put in the ids you have access to.
# Keep the budget (MAX_ITER) and the instance list (INSTANCES) fixed across rows,
# or the sweep is not a backbone comparison. Bug localization is a separate slot
# (`--triage-model`) and stays put unless you move it deliberately - it is one
# short call per candidate file and would dominate a cost column.
#
# On the cost column: `aggregate_results.py --price IN,OUT` prices the tokens the
# trajectory recorded, and the recorder only counts calls that report usage. Read
# docs/reproducibility.md before putting a dollar figure in a table.
source "$(dirname "$0")/_lib.sh"

DEFAULT_MODELS='qwen3_8b:--vllm Qwen/Qwen3-8B --concurrency 32
qwen3_14b:--vllm Qwen/Qwen3-14B --concurrency 16
qwen3_32b:--vllm Qwen/Qwen3-32B --concurrency 8'

# `SMALL` is the lightweight model the Dev-Primitives keep in the mixed rows;
# `BIG` is the frontier model moved into one orchestration role at a time.
SMALL=${SMALL:-Qwen/Qwen3-8B}
BIG=${BIG:-bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0}
DEFAULT_ROLES="homogeneous_small:--vllm $SMALL --concurrency 32
planner_big:--vllm $SMALL --concurrency 32 --planner-model $BIG
critic_big:--vllm $SMALL --concurrency 32 --critic-model $BIG
planner_critic_big:--vllm $SMALL --concurrency 32 --planner-model $BIG --critic-model $BIG
homogeneous_big:--model $BIG"

LIST=${MODELS:-$DEFAULT_MODELS}
TITLE="Backbone sweep (homogeneous)"
if [ "${1:-}" = "--roles" ]; then
  LIST=${ROLES:-$DEFAULT_ROLES}
  TITLE="Backbone sweep (per role)"
  shift
fi
[ $# -eq 0 ] || { echo "usage: $0 [--roles]" >&2; exit 2; }

dirs=()
while IFS= read -r spec; do
  [ -z "${spec// }" ] && continue
  label=${spec%%:*}
  flags=${spec#*:}
  # shellcheck disable=SC2086
  hermes_solve "$label" $flags
  hermes_eval "$label"
  dirs+=("$HERMES_RUNS/$label")
done <<< "${LIST//;/$'\n'}"

hermes_report --title "$TITLE" "${dirs[@]}"
