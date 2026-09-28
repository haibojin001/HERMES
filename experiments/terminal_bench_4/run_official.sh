#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../_run_harbor.sh" terminal_bench "${1:-all}" "${2:-}"
