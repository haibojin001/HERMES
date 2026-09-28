#!/usr/bin/env python3
"""Build the SWE-bench environment images the solver needs.

    python scripts/build_env_images.py --limit 30        # first 30 instances
    python scripts/build_env_images.py --instance django__django-13512
    python scripts/build_env_images.py                   # all 500, hours + ~200 GB

Why this matters more than it looks: with no environment image the solver does a
single edit round and stops - no tests, no Critic, no re-planning, no grading -
and reports `resolved=false`. Numbers produced that way describe a degraded
single-shot system, not HERMES. `scripts/prepare_dataset.py` reports how many
images are visible; this script builds the missing ones.

The solver needs the *environment* image (one per repo/version, shared by many
instances). The per-instance image is built by the solver itself on first use, so
building env images is enough and is far cheaper than building 500 instance
images.

This wraps the official SWE-bench builder, which talks to the Docker API through
the `docker` Python SDK rather than to a CLI. So:

  * docker  - works directly.
  * podman  - start the compatibility socket and point DOCKER_HOST at it
              (`podman system service --time 0 unix:///tmp/podman.sock`,
              `export DOCKER_HOST=unix:///tmp/podman.sock`).
  * finch   - no Docker-API socket. Either build with docker on the same machine,
              or build inside the finch VM with the SWE-bench harness installed
              there. Our own runs built with finch directly on macOS.
"""

import argparse
import json
import sys
from pathlib import Path

# Run as `python scripts/<name>.py` from anywhere: the repository root has
# to be importable, and the script directory is what python puts on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes import settings


def load_rows(args) -> list:
    if settings.DATASET_PATH.exists():
        rows = json.loads(settings.DATASET_PATH.read_text())
    else:
        from datasets import load_dataset
        rows = [dict(r) for r in
                load_dataset("princeton-nlp/SWE-bench_Verified", split="test")]
    if args.instance:
        wanted = set(args.instance)
        rows = [r for r in rows if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in rows}
        if missing:
            sys.exit(f"not in SWE-bench Verified: {', '.join(sorted(missing))}")
    if args.limit:
        rows = rows[:args.limit]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instance", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=4,
                    help="parallel builds; each one compiles a Python "
                         "environment, so more is not always faster")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the env images that would be built and stop")
    args = ap.parse_args()

    rows = load_rows(args)
    if not rows:
        sys.exit("no instances selected")

    from swebench.harness.test_spec.test_spec import make_test_spec
    specs = [make_test_spec(r) for r in rows]
    keys = sorted({s.env_image_key for s in specs})
    print(f"{len(rows)} instances -> {len(keys)} environment images")

    if args.dry_run:
        for k in keys:
            print(" ", k)
        return 0

    try:
        import docker
        client = docker.from_env()
        client.ping()
    except Exception as e:
        print(f"cannot reach a Docker API socket: {e}\n"
              "See the module docstring for podman and finch.", file=sys.stderr)
        return 1

    from swebench.harness.docker_build import build_env_images
    # Builds the base image first when it is missing, then one image per
    # repo/version pair. Already-present images are skipped unless forced.
    build_env_images(client, rows, force_rebuild=args.force_rebuild,
                     max_workers=args.max_workers)
    print("done - re-run scripts/prepare_dataset.py to confirm what is visible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
