#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../_run_manifest.sh" swe_refactor 1 "$@"
