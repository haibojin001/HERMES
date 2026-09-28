"""
Held-out SWE-bench grading. Stage 6: runs only after HERMES has
terminated and the patch has been written, so no component ever sees it.

Metrics reported per instance:
  - fail_to_pass:  Did FAIL_TO_PASS tests flip from FAIL -> PASS?
  - pass_to_pass:  Did PASS_TO_PASS tests stay PASS (no regression)?
  - resolved:      Both above are True
  - gold_overlap:  File-level overlap with gold patch (precision, recall, F1)

Usage:
    python -m hermes.evaluate
    python -m hermes.evaluate --instance django__django-10880 astropy__astropy-13579
    python -m hermes.evaluate --resume
"""

import os
import sys
import json
import subprocess
import time
import base64
import re
from pathlib import Path

from hermes import settings
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================
PREDICTIONS_PATH = settings.PREDICTIONS_PATH
DATASET_PATH = settings.DATASET_PATH
RESULTS_PATH = settings.EVAL_DIR / "results.jsonl"
REPORT_PATH = settings.EVAL_DIR / "report.json"
TIMEOUT = 300


# ============================================================
# Data Loading
# ============================================================
def load_hf_data() -> dict:
    """Load test patches and test lists from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    data = {}
    for item in ds:
        ftp = item.get("FAIL_TO_PASS", "[]")
        ptp = item.get("PASS_TO_PASS", "[]")
        data[item["instance_id"]] = {
            "test_patch": item.get("test_patch", ""),
            "FAIL_TO_PASS": json.loads(ftp) if isinstance(ftp, str) else ftp,
            "PASS_TO_PASS": json.loads(ptp) if isinstance(ptp, str) else ptp,
            "base_commit": item.get("base_commit"),
            "repo": item.get("repo", ""),
        }
    return data


def load_gold_patches() -> dict:
    """Load gold patch file lists from local dataset."""
    gold = {}
    with open(DATASET_PATH) as f:
        for inst in json.load(f):
            iid = inst["instance_id"]
            files = set(re.findall(r'diff --git a/(.*?) b/', inst.get("full_patch", "")))
            gold[iid] = files
    return gold


# ============================================================
# Test Command Builders (separate FAIL_TO_PASS and PASS_TO_PASS)
# ============================================================
def build_test_cmd_ftp(instance_id: str, tests: list) -> str:
    """Build test command for FAIL_TO_PASS tests only."""
    if not tests:
        return "echo 'NO_FTP_TESTS'"
    return _build_cmd(instance_id, tests)


def build_test_cmd_ptp(instance_id: str, tests: list) -> str:
    """Build test command for PASS_TO_PASS tests only."""
    if not tests:
        return "echo 'NO_PTP_TESTS'"
    return _build_cmd(instance_id, tests)


def _build_cmd(instance_id: str, tests: list) -> str:
    if "django" in instance_id:
        labels = set()
        for t in tests:
            if "(" in t:
                mod = t.split("(")[1].rstrip(")").strip()
                parts = mod.split(".")
                labels.add(parts[0])
            else:
                labels.add(t.split(".")[0])
        return f"cd /testbed/tests && python runtests.py --settings=test_sqlite --parallel 1 {' '.join(labels)} 2>&1"
    elif "sympy" in instance_id:
        return f"cd /testbed && python -m pytest {' '.join(tests)} -x 2>&1"
    elif "astropy" in instance_id:
        return f"cd /testbed && python -m pytest {' '.join(tests)} -xvs 2>&1"
    else:
        return f"cd /testbed && python -m pytest {' '.join(tests)} -x 2>&1"


# ============================================================
# Test Result Parsing
# ============================================================
def parse_test_passed(output: str, instance_id: str) -> bool:
    """Parse test output to determine if all tests passed."""
    if not output.strip():
        return False
    if "PATCH_APPLY_FAILED" in output:
        return False

    # Django uses its own test runner
    if "django" in instance_id:
        # "OK" at end means all passed, "FAILED" means some failed
        last_lines = output.strip().split('\n')[-10:]
        last_block = '\n'.join(last_lines)
        if "OK" in last_block and "FAIL" not in last_block:
            return True
        if "FAILED" in last_block or "ERROR" in last_block:
            return False
        # Check for "Ran X tests" with no failures
        if re.search(r'Ran \d+ tests? in', last_block) and "OK" in last_block:
            return True
        return False

    # pytest-based repos
    last_lines = output.strip().split('\n')[-5:]
    last_block = '\n'.join(last_lines)
    if "passed" in last_block and "failed" not in last_block and "error" not in last_block:
        return True
    if re.search(r'\d+ passed', last_block) and not re.search(r'\d+ (failed|error)', last_block):
        return True
    if "FAILED" in last_block or "ERRORS" in last_block:
        return False
    # Sympy-specific
    if "OK" in last_block and "FAIL" not in last_block:
        return True

    return False


# ============================================================
# Evaluate Single Instance
# ============================================================
def evaluate_one(instance_id: str, model_patch: str, image: str,
                 test_patch: str, fail_to_pass: list,
                 pass_to_pass: list, timeout: int = 300) -> dict:
    """
    Evaluate one prediction. Runs FAIL_TO_PASS and PASS_TO_PASS separately.
    Returns per-instance metrics.
    """
    result = {
        "instance_id": instance_id,
        "fail_to_pass": False,
        "pass_to_pass": False,
        "resolved": False,
        "patch_applies": False,
        "reason": "",
        "ftp_output": "",
        "ptp_output": "",
    }

    if not model_patch.strip():
        result["reason"] = "empty_patch"
        return result

    model_b64 = base64.b64encode(model_patch.encode()).decode()
    test_b64 = base64.b64encode(test_patch.encode()).decode() if test_patch else ""

    ftp_cmd = build_test_cmd_ftp(instance_id, fail_to_pass)
    ptp_cmd = build_test_cmd_ptp(instance_id, pass_to_pass)

    # Run both test sets in one container invocation for efficiency
    # Determine repo_dir for this instance (for volume mounting)
    repo_dir = settings.WORK_DIR / instance_id
    work_dir = settings.WORK_DIR

    # Write patches to temp files for mounting
    model_patch_file = work_dir / f"_eval_model_{instance_id}.diff"
    model_patch_file.write_text(model_patch)

    test_patch_file = work_dir / f"_eval_test_{instance_id}.diff"
    test_patch_file.write_text(test_patch if test_patch else "")

    script = f"""
source /opt/miniconda3/bin/activate
conda activate testbed

# Setup testbed from mounted repo
mkdir -p /testbed && cp -a /mnt/repo/. /testbed/ 2>/dev/null || {{ echo "COPY_FAILED"; exit 1; }}
cd /testbed
git config --global --add safe.directory /testbed
pip install -e . -q 2>/dev/null || pip install -e ".[test]" -q 2>/dev/null || export PYTHONPATH=/testbed:$PYTHONPATH

# Apply test patch (new tests from the PR)
{"git apply /mnt/test.diff 2>&1 || git apply --3way /mnt/test.diff 2>&1" if test_patch else "true"}

# Apply model patch
git apply /mnt/model.diff 2>&1
APPLY_RC=$?
if [ $APPLY_RC -ne 0 ]; then
    echo "PATCH_APPLY_FAILED_INITIAL"
    git apply --3way /mnt/model.diff 2>&1
    APPLY_RC=$?
    if [ $APPLY_RC -ne 0 ]; then
        echo "PATCH_APPLY_FAILED_FINAL"
        exit 1
    fi
fi
echo "PATCH_APPLIED_OK"

# Run FAIL_TO_PASS tests
echo "=== FAIL_TO_PASS ==="
{ftp_cmd} 2>&1 | tail -80
echo "=== END_FTP ==="

# Run PASS_TO_PASS tests
echo "=== PASS_TO_PASS ==="
{ptp_cmd} 2>&1 | tail -80
echo "=== END_PTP ==="
"""

    run_cmd = [settings.CONTAINER_CLI, "run", "--rm"]
    if repo_dir.exists():
        run_cmd += ["-v", f"{repo_dir}:/mnt/repo:ro"]
    run_cmd += ["-v", f"{model_patch_file}:/mnt/model.diff:ro"]
    if test_patch:
        run_cmd += ["-v", f"{test_patch_file}:/mnt/test.diff:ro"]
    run_cmd += [image, "bash", "-c", script]

    try:
        proc = subprocess.run(
            run_cmd,
            capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        result["reason"] = "timeout"
        return result

    output = proc.stdout + proc.stderr

    # Check patch application
    if "PATCH_APPLY_FAILED_FINAL" in output:
        result["reason"] = "patch_apply_failed"
        return result
    if "PATCH_APPLIED_OK" in output:
        result["patch_applies"] = True

    # Parse FAIL_TO_PASS section
    ftp_section = ""
    if "=== FAIL_TO_PASS ===" in output and "=== END_FTP ===" in output:
        ftp_section = output.split("=== FAIL_TO_PASS ===")[1].split("=== END_FTP ===")[0]
    elif "=== FAIL_TO_PASS ===" in output:
        ftp_section = output.split("=== FAIL_TO_PASS ===")[1][:3000]
    result["ftp_output"] = ftp_section[-1500:]
    result["fail_to_pass"] = parse_test_passed(ftp_section, instance_id)

    # Parse PASS_TO_PASS section
    ptp_section = ""
    if "=== PASS_TO_PASS ===" in output and "=== END_PTP ===" in output:
        ptp_section = output.split("=== PASS_TO_PASS ===")[1].split("=== END_PTP ===")[0]
    elif "=== PASS_TO_PASS ===" in output:
        ptp_section = output.split("=== PASS_TO_PASS ===")[1][:3000]

    # If no PASS_TO_PASS tests, consider it passing
    if not pass_to_pass or "NO_PTP_TESTS" in ptp_section:
        result["pass_to_pass"] = True
    else:
        result["ptp_output"] = ptp_section[-1500:]
        result["pass_to_pass"] = parse_test_passed(ptp_section, instance_id)

    # Resolved = both pass
    result["resolved"] = result["fail_to_pass"] and result["pass_to_pass"]

    return result


# ============================================================
# Gold Patch Overlap
# ============================================================
def compute_gold_overlap(model_patch: str, gold_files: set) -> dict:
    """Compute file-level overlap between model patch and gold patch."""
    if not model_patch.strip() or not gold_files:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "model_files": [], "gold_files": list(gold_files),
                "matched": [], "missed": list(gold_files), "extra": []}

    model_files = set(re.findall(r'diff --git a/(.*?) b/', model_patch))

    matched = model_files & gold_files
    missed = gold_files - model_files
    extra = model_files - gold_files

    precision = len(matched) / len(model_files) if model_files else 0.0
    recall = len(matched) / len(gold_files) if gold_files else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "model_files": sorted(model_files),
        "gold_files": sorted(gold_files),
        "matched": sorted(matched),
        "missed": sorted(missed),
        "extra": sorted(extra),
    }


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="HERMES held-out SWE-bench evaluator")
    parser.add_argument("--instance", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH))
    parser.add_argument("--timeout", type=int, default=TIMEOUT)
    args = parser.parse_args()

    timeout = args.timeout

    # Load predictions
    predictions = {}
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                predictions[d["instance_id"]] = d["model_patch"]

    if args.instance:
        predictions = {k: v for k, v in predictions.items() if k in set(args.instance)}

    print(f"Loaded {len(predictions)} predictions")

    # Load data
    print("Loading HuggingFace dataset...")
    hf_data = load_hf_data()
    print("Loading gold patches...")
    gold_patches = load_gold_patches()

    # Check available images
    img_result = subprocess.run(
        [settings.CONTAINER_CLI, "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True
    )
    available_images = set(img_result.stdout.strip().split('\n'))

    # Map instance_id -> env image using swebench test specs
    print("Mapping instances to env images...")
    from swebench.harness.docker_build import make_test_spec
    from datasets import load_dataset as _load_ds
    _ds = _load_ds("princeton-nlp/SWE-bench_Verified", split="test")
    instance_to_image = {}
    for item in _ds:
        spec = make_test_spec(item)
        if spec.env_image_key in available_images:
            instance_to_image[item["instance_id"]] = spec.env_image_key
    print(f"  {len(instance_to_image)} instances have docker available")

    # Skip already evaluated
    done = set()
    if args.resume and RESULTS_PATH.exists():
        for line in RESULTS_PATH.read_text().strip().split('\n'):
            if line.strip():
                try:
                    d = json.loads(line)
                    done.add(d["instance_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} already evaluated")

    # Match predictions to available images
    to_eval = []
    skipped_no_image = []
    gold_only = []  # instances where we can only compute gold overlap (no image)

    for iid, patch in predictions.items():
        if iid in done:
            continue
        image = instance_to_image.get(iid)
        if image:
            to_eval.append((iid, patch, image))
        else:
            skipped_no_image.append(iid)
            gold_only.append((iid, patch))

    print(f"\nTo evaluate (with docker): {len(to_eval)}")
    print(f"Gold overlap only (no image): {len(gold_only)}")

    # ============================================================
    # Phase 1: Gold overlap for ALL predictions (no docker needed)
    # ============================================================
    print(f"\n{'='*70}")
    print("GOLD PATCH OVERLAP (all predictions)")
    print(f"{'='*70}")

    all_overlaps = {}
    for iid, patch in predictions.items():
        gold_files = gold_patches.get(iid, set())
        overlap = compute_gold_overlap(patch, gold_files)
        all_overlaps[iid] = overlap

    # Aggregate stats
    recalls = [v["recall"] for v in all_overlaps.values()]
    precisions = [v["precision"] for v in all_overlaps.values()]
    f1s = [v["f1"] for v in all_overlaps.values()]

    print(f"  Instances: {len(all_overlaps)}")
    print(f"  File Recall:    mean={sum(recalls)/len(recalls):.3f}  "
          f"median={sorted(recalls)[len(recalls)//2]:.3f}")
    print(f"  File Precision: mean={sum(precisions)/len(precisions):.3f}  "
          f"median={sorted(precisions)[len(precisions)//2]:.3f}")
    print(f"  File F1:        mean={sum(f1s)/len(f1s):.3f}  "
          f"median={sorted(f1s)[len(f1s)//2]:.3f}")
    print(f"  100% recall: {sum(1 for r in recalls if r >= 1.0)}/{len(recalls)}")
    print(f"  >0% recall:  {sum(1 for r in recalls if r > 0)}/{len(recalls)}")

    # ============================================================
    # Phase 2: Docker evaluation (FAIL_TO_PASS + PASS_TO_PASS)
    # ============================================================
    if to_eval:
        print(f"\n{'='*70}")
        print("DOCKER EVALUATION (FAIL_TO_PASS + PASS_TO_PASS)")
        print(f"{'='*70}")

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results = []

        for i, (iid, patch, image) in enumerate(to_eval):
            hf = hf_data.get(iid, {})
            test_patch = hf.get("test_patch", "")
            fail_to_pass = hf.get("FAIL_TO_PASS", [])
            pass_to_pass = hf.get("PASS_TO_PASS", [])

            print(f"  [{i+1}/{len(to_eval)}] {iid}...", end=" ", flush=True)

            result = evaluate_one(
                iid, patch, image, test_patch, fail_to_pass, pass_to_pass,
                timeout=timeout
            )
            # Add gold overlap
            result["gold_overlap"] = all_overlaps.get(iid, {})
            results.append(result)

            status = "RESOLVED" if result["resolved"] else \
                     f"FTP={'PASS' if result['fail_to_pass'] else 'FAIL'} " \
                     f"PTP={'PASS' if result['pass_to_pass'] else 'FAIL'}"
            print(status)

            # Save incrementally
            with open(RESULTS_PATH, "a") as f:
                # Don't save full output to results file (too large)
                save_result = {k: v for k, v in result.items()
                               if k not in ("ftp_output", "ptp_output")}
                f.write(json.dumps(save_result) + "\n")

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")

        n = len(results)
        ftp_pass = sum(1 for r in results if r["fail_to_pass"])
        ptp_pass = sum(1 for r in results if r["pass_to_pass"])
        resolved = sum(1 for r in results if r["resolved"])
        patch_ok = sum(1 for r in results if r["patch_applies"])

        print(f"  Total evaluated:  {n}")
        print(f"  Patch applies:    {patch_ok}/{n} ({patch_ok/n*100:.1f}%)")
        print(f"  FAIL_TO_PASS:     {ftp_pass}/{n} ({ftp_pass/n*100:.1f}%)")
        print(f"  PASS_TO_PASS:     {ptp_pass}/{n} ({ptp_pass/n*100:.1f}%)")
        print(f"  RESOLVED:         {resolved}/{n} ({resolved/n*100:.1f}%)")
        print()

        # Per-instance breakdown
        print("  Per-instance:")
        for r in results:
            mark = "yes" if r["resolved"] else "no"
            ftp = "FTPyes" if r["fail_to_pass"] else "FTPno"
            ptp = "PTPyes" if r["pass_to_pass"] else "PTPno"
            go = r.get("gold_overlap", {})
            gold_str = f"F1={go.get('f1', 0):.2f}" if go else ""
            print(f"    {mark} {r['instance_id']:40s} {ftp} {ptp} {gold_str}")

    # ============================================================
    # Save full report
    # ============================================================
    report = {
        "total_predictions": len(predictions),
        "evaluated_with_docker": len(to_eval),
        "skipped_no_image": len(skipped_no_image),
        "gold_overlap": {
            "mean_recall": round(sum(recalls)/len(recalls), 3) if recalls else 0,
            "mean_precision": round(sum(precisions)/len(precisions), 3) if precisions else 0,
            "mean_f1": round(sum(f1s)/len(f1s), 3) if f1s else 0,
            "perfect_recall": sum(1 for r in recalls if r >= 1.0),
        },
        "docker_results": {},
    }

    if to_eval:
        n = len(to_eval)
        report["docker_results"] = {
            "total": n,
            "patch_applies": patch_ok,
            "fail_to_pass": ftp_pass,
            "pass_to_pass": ptp_pass,
            "resolved": resolved,
            "resolved_pct": round(resolved / n * 100, 1),
            "resolved_instances": [r["instance_id"] for r in results if r["resolved"]],
            "ftp_only_instances": [r["instance_id"] for r in results
                                   if r["fail_to_pass"] and not r["pass_to_pass"]],
        }

    report["per_instance"] = {}
    for iid in predictions:
        entry = {"gold_overlap": all_overlaps.get(iid, {})}
        report["per_instance"][iid] = entry

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()