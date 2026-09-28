"""List SWE-bench Verified tasks that can actually produce a paper case study.

A usable candidate needs all of: an env image already pulled (otherwise the
held-out evaluation is skipped entirely and `resolved` is False no matter how
good the patch is), a gold patch touching two non-test files so the case can
show inter-primitive communication at all, and both of those files nameable
from the issue text so bug localization has a chance.
"""
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from hermes import settings

from datasets import load_dataset
from swebench.harness.test_spec.test_spec import make_test_spec

WORK_DIR = settings.WORK_DIR


def local_images():
    fmt = ["--format", "{{.Repository}}:{{.Tag}}"]
    for cmd in ([settings.CONTAINER_CLI, "images"] + fmt, ["docker", "images"] + fmt):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
    return ""


def touched_files(patch):
    return re.findall(r"^diff --git a/(\S+)", patch, re.M)


def main():
    images = local_images()
    data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = []
    for inst in data:
        files = touched_files(inst["patch"])
        src = [f for f in files if "test" not in f.lower()]
        if not 2 <= len(src) <= 3 or len(src) != len(files):
            continue
        issue = inst["problem_statement"].lower()
        stems = [Path(f).stem.lower() for f in src]
        # How many gold files the issue text actually names. Not a filter any
        # more: dropping everything the issue does not name left 8 tasks, of
        # which 5 had already failed, so the pool has to be wider than that.
        named = sum(s in issue or f.lower() in issue for s, f in zip(stems, src))
        spec = make_test_spec(inst)
        if spec.env_image_key not in images:
            continue
        rows.append({
            "id": inst["instance_id"],
            "diff": len(inst["patch"]),
            "files": src,
            "checkout": (WORK_DIR / inst["instance_id"]).is_dir(),
            "f2p": len(json.loads(inst["FAIL_TO_PASS"])),
            "named": named,
            "difficulty": inst.get("difficulty", "?"),
        })

    # Most promising first: the issue naming both files, then a small gold diff
    # and few held-out tests to satisfy, then a checkout that already exists.
    rows.sort(key=lambda r: (-r["named"], r["f2p"], r["diff"], not r["checkout"]))
    by_repo = defaultdict(int)
    print(f"{len(rows)} candidates with a local env image\n")
    for r in rows:
        by_repo[r["id"].split("__")[0]] += 1
        print(f"{r['id']:38s} named={r['named']}/{len(r['files'])} "
              f"f2p={r['f2p']:2d} diff={r['diff']:5d} "
              f"checkout={'y' if r['checkout'] else 'n'} [{r['difficulty']}]")
        for f in r["files"]:
            print(f"    {f}")
    print("\nby repo:", dict(by_repo))


if __name__ == "__main__":
    main()
