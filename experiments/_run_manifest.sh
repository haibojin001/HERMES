#!/usr/bin/env bash
# Shared launcher for benchmark-provided task snapshots.
set -euo pipefail

BENCHMARK=$1
REPEATS=$2
shift 2
if [ "$#" -lt 2 ]; then
  echo "usage: run.sh MANIFEST.jsonl OUTPUT_DIR [extra runner flags]" >&2
  exit 2
fi
MANIFEST=$1
OUTPUT=$2
shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON:-python3}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "${HERMES_BACKEND:-ollama}" in
  ollama) MODEL_ARGS=(--ollama "${HERMES_MODEL:-qwen3:8b}") ;;
  vllm) MODEL_ARGS=(--vllm "${HERMES_MODEL:-Qwen/Qwen3-8B}") ;;
  hosted)
    if [ -z "${HERMES_MODEL:-}" ]; then
      echo "HERMES_MODEL is required for the hosted backend" >&2
      exit 2
    fi
    MODEL_ARGS=(--model "$HERMES_MODEL")
    ;;
  *) echo "HERMES_BACKEND must be ollama, vllm or hosted" >&2; exit 2 ;;
esac

PROTOCOL_ARGS=(--paper-protocol)
if [ "${HERMES_SMOKE:-0}" = 1 ]; then
  PROTOCOL_ARGS=()
fi

"$PY" -m hermes.manifest_runner \
  --benchmark "$BENCHMARK" --manifest "$MANIFEST" --output "$OUTPUT" \
  --repeat "$REPEATS" --max-iterations "${HERMES_MAX_ITERATIONS:-4}" \
  "${PROTOCOL_ARGS[@]}" "${MODEL_ARGS[@]}" "$@"
