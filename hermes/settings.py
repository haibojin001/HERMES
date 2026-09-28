"""Filesystem locations and container runtime, all overridable by environment.

Nothing here is a model or a method parameter - those live in `pipeline.py` and
are set from the command line. This module exists so that a clone of the
repository runs without editing source.

    HERMES_HOME           root for everything this repository writes
                          (default: ./hermes_data next to the repository)
    HERMES_WORK_DIR       per-instance repository checkouts
    HERMES_DATASET        SWE-bench Verified as a JSON list of instance dicts
    HERMES_PREDICTIONS    where `model_patch` records are appended
    HERMES_TRAJECTORIES   one directory per run, see docs/trajectory-format.md
    HERMES_CONTAINER_CLI  `docker`, `finch` or `podman` (default: autodetect)
"""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HOME = Path(os.environ.get("HERMES_HOME") or REPO_ROOT / "hermes_data")

WORK_DIR = Path(os.environ.get("HERMES_WORK_DIR") or HOME / "swebench_workdir")
DATASET_PATH = Path(os.environ.get("HERMES_DATASET")
                    or HOME / "swebench_verified_all.json")
PREDICTIONS_PATH = Path(os.environ.get("HERMES_PREDICTIONS")
                        or HOME / "predictions.jsonl")
TRAJ_ROOT = Path(os.environ.get("HERMES_TRAJECTORIES") or HOME / "trajectories")
EVAL_DIR = Path(os.environ.get("HERMES_EVAL_DIR") or HOME / "eval_results")


def _autodetect_container_cli() -> str:
    """The CLI used to build and run the per-instance evaluation images.

    Docker first because that is what the SWE-bench harness assumes; finch and
    podman are accepted because they are CLI-compatible for the three commands
    used here (`images`, `build`, `run`). Our own runs used finch on macOS.
    """
    for cli in ("docker", "finch", "podman"):
        if shutil.which(cli):
            return cli
    return "docker"


CONTAINER_CLI = os.environ.get("HERMES_CONTAINER_CLI") or _autodetect_container_cli()

# Image tag suffix. SWE-bench's `test_spec.instance_image_key` ends in `:latest`;
# our finch images were built under a `-finch` repository suffix so they could
# coexist with docker-built ones. Keep it empty unless you have both.
IMAGE_SUFFIX = os.environ.get("HERMES_IMAGE_SUFFIX", "")


def ensure_dirs() -> None:
    for d in (HOME, WORK_DIR, TRAJ_ROOT, EVAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
