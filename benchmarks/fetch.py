#!/usr/bin/env python3
"""Fetch and verify exact upstream benchmark snapshots outside the public tree."""

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOCK = json.loads((Path(__file__).parent / "lock.json").read_text())


def run(argv: list[str], cwd: Path | None = None,
        env: dict | None = None) -> str:
    result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                            capture_output=True, check=True)
    return result.stdout.strip()


def fetch(name: str, destination: Path, assets: bool = False) -> Path:
    spec = LOCK[name]
    target = destination / name
    destination.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.mkdir()
        run(["git", "init", "-q", str(target)])
        run(["git", "remote", "add", "origin", spec["url"]], cwd=target)
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        if name == "terminal_bench_legacy":
            run(["git", "sparse-checkout", "set", "terminal_bench"], cwd=target)
        run(["git", "fetch", "--depth", "1", "--filter=blob:none",
             "origin", spec["ref"]], cwd=target, env=env)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target, env=env)
    actual = run(["git", "rev-parse", "HEAD"], cwd=target)
    if actual != spec["commit"]:
        raise RuntimeError(f"{name}: expected {spec['commit']}, found {actual}")
    if assets and name == "devops_gym":
        run(["git", "lfs", "pull"], cwd=target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", nargs="+", choices=sorted(LOCK))
    parser.add_argument("--destination", type=Path,
                        default=ROOT / "hermes_data" / "benchmarks")
    parser.add_argument("--with-assets", action="store_true",
                        help="materialize DevOps-Gym Git LFS task assets")
    args = parser.parse_args()
    for name in args.benchmark:
        path = fetch(name, args.destination, args.with_assets)
        print(f"{name}: {path} ({LOCK[name]['commit']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
