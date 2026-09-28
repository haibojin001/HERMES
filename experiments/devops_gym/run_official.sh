#!/usr/bin/env bash
# DevOps-Gym uses the older `tb` harness, not Harbor v4.
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "usage: run_official.sh {build|monitor|issue_resolving|test_generation} [OUTPUT_DIR]" >&2
  exit 2
fi
CATEGORY=$1
case "$CATEGORY" in
  build|monitor|issue_resolving|test_generation) ;;
  *) echo "unknown DevOps-Gym category: $CATEGORY" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${HERMES_BENCHMARKS:-"$HERE/hermes_data/benchmarks"}
BENCH_ROOT="$DATA_ROOT/devops_gym"
if [ ! -d "$BENCH_ROOT/.git" ]; then
  echo "fetch DevOps-Gym first: python benchmarks/fetch.py devops_gym terminal_bench_legacy" >&2
  exit 2
fi
EXPECTED=$(python3 - "$HERE/benchmarks/lock.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["devops_gym"]["commit"])
PY
)
ACTUAL=$(git -C "$BENCH_ROOT" rev-parse HEAD)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "DevOps-Gym commit mismatch: expected $EXPECTED, found $ACTUAL" >&2
  exit 2
fi
if ! command -v tb >/dev/null 2>&1; then
  echo "legacy Terminal-Bench `tb` CLI missing" >&2
  exit 2
fi
if [ "${HERMES_CONCURRENT:-1}" -ne 1 ]; then
  echo "HERMES_CONCURRENT must be 1: the in-process model and trajectory state is per trial" >&2
  exit 2
fi
if [ "${HERMES_BACKEND:-ollama}" = hosted ] && [ -z "${HERMES_MODEL:-}" ]; then
  echo "set HERMES_MODEL to the hosted provider/model ID" >&2
  exit 2
fi
OUTPUT=${2:-"$HERE/hermes_data/official_runs/devops_gym"}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
TAG=${HERMES_RUN_TAG:-paper}
case "$TAG" in
  *[!A-Za-z0-9_-]*)
    echo "HERMES_RUN_TAG must contain letters, digits, underscore or dash" >&2
    exit 2 ;;
esac
ARGS=(--dataset-path "$BENCH_ROOT/tasks/$CATEGORY"
      --agent-import-path hermes.legacy_tb_agent:HermesTerminalBenchAgent
      --model "${HERMES_MODEL:-qwen3:8b}"
      --agent-kwarg "model=${HERMES_MODEL:-qwen3:8b}"
      --agent-kwarg "backend=${HERMES_BACKEND:-ollama}"
      --agent-kwarg "max_iterations=${HERMES_MAX_ITERATIONS:-4}"
      --n-concurrent 1
      --output-path "$OUTPUT"
      --run-id "hermes_${CATEGORY}_${TAG}")
for slot in PLANNER PRIMITIVE CRITIC TRIAGE; do
  key="HERMES_${slot}_MODEL"
  if [ -n "${!key:-}" ]; then
    case "$slot" in
      PLANNER) name=planner_model ;;
      PRIMITIVE) name=primitive_model ;;
      CRITIC) name=critic_model ;;
      TRIAGE) name=triage_model ;;
    esac
    ARGS+=(--agent-kwarg "$name=${!key}")
  fi
done
if [ -n "${HERMES_REASONING_EFFORT:-}" ]; then
  ARGS+=(--agent-kwarg "reasoning_effort=$HERMES_REASONING_EFFORT")
fi
if [ -n "${HERMES_OLLAMA_URL:-}" ]; then
  ARGS+=(--agent-kwarg "ollama_url=$HERMES_OLLAMA_URL")
fi
if [ -n "${HERMES_VLLM_URL:-}" ]; then
  ARGS+=(--agent-kwarg "vllm_url=$HERMES_VLLM_URL")
fi
tb run "${ARGS[@]}" "${@:3}"
