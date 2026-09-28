#!/usr/bin/env bash
# The main result: the complete system on SWE-bench Verified (paper Table
# `swebench-verified`). Solve, then grade, then print the row.
#
#   scripts/run_swebench.sh                      # all 500, Bedrock backbone
#   scripts/run_swebench.sh --limit 50           # the first 50 ids in the dataset
#   BACKEND_ARGS="--vllm Qwen/Qwen3-8B --concurrency 32" scripts/run_swebench.sh
#
# Resumable: a run interrupted after 300 instances continues where it stopped,
# because the pipeline skips instance ids already present in this configuration's
# predictions file. Pass NO_RESUME=1 to start over.
#
# Expect this to take days on 500 instances: every round runs the repository in a
# container, and a container start plus a test suite dominates the wall clock, not
# the model. Run it under `nohup`/`tmux`, or split the id list across machines and
# merge the run directories afterwards - the aggregate script accepts several.
source "$(dirname "$0")/_lib.sh"

LABEL=${LABEL:-complete}
LIMIT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --limit) LIMIT=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    *) echo "usage: $0 [--limit N] [--label NAME]" >&2; exit 2 ;;
  esac
done

if [ -n "$LIMIT" ]; then
  # Take ids from the prepared dataset rather than hardcoding a subset, so the
  # selection is reproducible from the file the solver itself reads.
  INSTANCES=$("$PY" - "$LIMIT" <<'EOF'
import json, sys
from hermes import settings
rows = json.loads(settings.DATASET_PATH.read_text())
print(" ".join(r["instance_id"] for r in rows[:int(sys.argv[1])]))
EOF
)
  echo "restricted to $LIMIT instances"
fi

hermes_solve "$LABEL"
hermes_eval "$LABEL"
hermes_report "$HERMES_RUNS/$LABEL"
