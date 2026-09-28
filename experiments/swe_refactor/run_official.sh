#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../_run_harbor.sh" swe_refactor "${1:-all}" "${2:-}"
