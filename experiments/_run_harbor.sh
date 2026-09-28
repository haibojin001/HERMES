#!/usr/bin/env bash
# Run a pinned official Harbor task set and its post-agent verifier.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: _run_harbor.sh {swe_refactor|terminal_bench} {all|TASK_ID} [JOBS_DIR]" >&2
  exit 2
fi
BENCHMARK=$1
TASK=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
DATA_ROOT=${HERMES_BENCHMARKS:-"$HERE/hermes_data/benchmarks"}
BENCH_ROOT="$DATA_ROOT/$BENCHMARK"
if [ ! -d "$BENCH_ROOT/.git" ]; then
  echo "fetch $BENCHMARK first: python benchmarks/fetch.py $BENCHMARK" >&2
  exit 2
fi
EXPECTED=$(python3 - "$HERE/benchmarks/lock.json" "$BENCHMARK" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]]["commit"])
PY
)
ACTUAL=$(git -C "$BENCH_ROOT" rev-parse HEAD)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "$BENCHMARK commit mismatch: expected $EXPECTED, found $ACTUAL" >&2
  exit 2
fi
if [ "$TASK" = all ]; then
  TASK_PATH="$BENCH_ROOT/tasks"
else
  TASK_PATH="$BENCH_ROOT/tasks/$TASK"
  if [ ! -f "$TASK_PATH/task.toml" ]; then
    echo "unknown task: $TASK" >&2
    exit 2
  fi
fi
if ! command -v harbor >/dev/null 2>&1; then
  echo "Harbor CLI missing: install Harbor in this environment" >&2
  exit 2
fi
case "$BENCHMARK" in
  swe_refactor) ATTEMPTS=${HERMES_ATTEMPTS:-1} ;;
  terminal_bench) ATTEMPTS=${HERMES_ATTEMPTS:-5} ;;
  *) echo "unknown Harbor benchmark: $BENCHMARK" >&2; exit 2 ;;
esac
if [ "$BENCHMARK" = terminal_bench ] && [ "$ATTEMPTS" -ne 5 ]; then
  echo "Terminal-Bench paper protocol requires five attempts" >&2
  exit 2
fi
MODEL=${HERMES_MODEL:-qwen3:8b}
BACKEND=${HERMES_BACKEND:-ollama}
if [ "$BACKEND" = hosted ] && [ -z "${HERMES_MODEL:-}" ]; then
  echo "set HERMES_MODEL to the hosted provider/model ID" >&2
  exit 2
fi
if [ "${HERMES_CONCURRENT:-1}" -ne 1 ]; then
  echo "HERMES_CONCURRENT must be 1: the in-process model and trajectory state is per trial" >&2
  exit 2
fi
OUTPUT=${3:-"$HERE/hermes_data/official_runs/$BENCHMARK"}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
if [ "$BENCHMARK" = swe_refactor ]; then
  export PYTHONPATH="$BENCH_ROOT/infra:$PYTHONPATH"
fi
cd "$BENCH_ROOT"
ARGS=(-p "$TASK_PATH" -a hermes.harbor_agent:HermesHarborAgent
      -m "$MODEL" -k 1 --n-concurrent 1
      --jobs-dir "$OUTPUT" --ak "backend=$BACKEND"
      --ak "max_iterations=${HERMES_MAX_ITERATIONS:-4}")
for slot in PLANNER PRIMITIVE CRITIC TRIAGE; do
  key="HERMES_${slot}_MODEL"
  if [ -n "${!key:-}" ]; then
    case "$slot" in
      PLANNER) name=planner_model ;;
      PRIMITIVE) name=primitive_model ;;
      CRITIC) name=critic_model ;;
      TRIAGE) name=triage_model ;;
    esac
    ARGS+=(--ak "$name=${!key}")
  fi
done
if [ -n "${HERMES_REASONING_EFFORT:-}" ]; then
  ARGS+=(--ak "reasoning_effort=$HERMES_REASONING_EFFORT")
fi
if [ -n "${HERMES_OLLAMA_URL:-}" ]; then
  ARGS+=(--ak "ollama_url=$HERMES_OLLAMA_URL")
fi
if [ -n "${HERMES_VLLM_URL:-}" ]; then
  ARGS+=(--ak "vllm_url=$HERMES_VLLM_URL")
fi
if [ "$BENCHMARK" = swe_refactor ]; then
  ARGS+=(--ak workspace=/workspace/repo)
else
  ARGS+=(--ak terminal_mode=auto)
fi
if [ -n "${HERMES_RUN_TAG:-}" ]; then
  case "$HERMES_RUN_TAG" in
    *[!A-Za-z0-9_-]*)
      echo "HERMES_RUN_TAG must contain letters, digits, underscore or dash" >&2
      exit 2 ;;
  esac
fi
for ((RUN_INDEX=0; RUN_INDEX<ATTEMPTS; RUN_INDEX++)); do
  TAG=${HERMES_RUN_TAG:-paper}
  harbor run "${ARGS[@]}" \
    --job-name "hermes_${BENCHMARK}_${TAG}_r${RUN_INDEX}"
done
