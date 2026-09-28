#!/usr/bin/env python3
"""Download SWE-bench Verified and check the local environment.

    python scripts/prepare_dataset.py

Writes the 500 instances to $HERMES_DATASET as a JSON list, creates the working
directories, and reports which environment images are present. It does not build
or pull images: that is the SWE-bench harness's job and takes hours of disk.
"""

import json
import subprocess
import sys
from pathlib import Path

# Run as `python scripts/<name>.py` from anywhere: the repository root has
# to be importable, and the script directory is what python puts on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes import settings


def main() -> int:
    settings.ensure_dirs()

    if settings.DATASET_PATH.exists():
        n = len(json.loads(settings.DATASET_PATH.read_text()))
        print(f"dataset: {settings.DATASET_PATH} ({n} instances, already present)")
    else:
        from datasets import load_dataset
        print("downloading princeton-nlp/SWE-bench_Verified ...")
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        rows = [dict(r) for r in ds]
        settings.DATASET_PATH.write_text(json.dumps(rows))
        print(f"dataset: {settings.DATASET_PATH} ({len(rows)} instances)")

    print(f"container CLI: {settings.CONTAINER_CLI}")
    try:
        out = subprocess.run(
            [settings.CONTAINER_CLI, "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=60).stdout
    except FileNotFoundError:
        print(f"  {settings.CONTAINER_CLI} not found - set HERMES_CONTAINER_CLI")
        return 1
    env_images = [l for l in out.splitlines() if "sweb.eval" in l or "swebench" in l]
    print(f"  {len(env_images)} SWE-bench images visible locally")
    if not env_images:
        print("  Without environment images the solver does one edit round and\n"
              "  stops: no tests, no Critic, no re-planning, no grading. Build\n"
              "  them with the SWE-bench harness before reporting any number.")

    print(f"work dir:     {settings.WORK_DIR}")
    print(f"trajectories: {settings.TRAJ_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
