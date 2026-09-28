#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../_run_manifest.sh" terminal_bench 5 "$@"
