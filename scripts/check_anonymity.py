#!/usr/bin/env python3
"""Fail if the public code tree contains local identity or runtime artifacts."""

import argparse
import os
import re
from pathlib import Path


TEXT_SUFFIXES = {".py", ".sh", ".md", ".toml", ".txt", ".cff", ".json",
                 ".jsonl", ".sbatch", ".yml", ".yaml", ".cfg", ".ini"}
SKIP_PARTS = {".git", "__pycache__", ".venv", "venv"}
FORBIDDEN_PARTS = {"hermes_data", ".idea", ".DS_Store"}
PATTERNS = [
    re.compile(r"/" + "Users" + r"/[A-Za-z0-9_.-]+/"),
    re.compile(r"/" + "home" + r"/[A-Za-z0-9_.-]+/"),
    re.compile("AK" + "IA" + r"[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]
EMAIL_ALLOW = ("example.com", "example.org", "example.net", "invalid.example")


def violations(root: Path) -> list[str]:
    bad = []
    identity_terms = [term.strip() for term in
                      os.environ.get("HERMES_ANON_BLOCKLIST", "").split(",")
                      if term.strip()]
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if path.is_symlink():
            bad.append(f"{rel}: symbolic link")
            continue
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            bad.append(f"{rel}: runtime or metadata path")
            continue
        if (not path.is_file() or
                (path.suffix.lower() not in TEXT_SUFFIXES
                 and path.name not in ("LICENSE", "Dockerfile", ".gitignore"))):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            bad.append(f"{rel}: non-text content")
            continue
        for name in identity_terms:
            if name.lower() in text.lower():
                bad.append(f"{rel}: local identity token")
                break
        for regex in PATTERNS:
            matches = regex.findall(text)
            if regex.pattern.startswith("[A-Za-z0-9_.+-]+@"):
                matches = [value for value in matches
                           if value.rsplit("@", 1)[-1] not in EMAIL_ALLOW]
            if matches:
                bad.append(f"{rel}: {regex.pattern}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    bad = violations(args.root)
    if bad:
        print("\n".join(bad))
        return 1
    print(f"anonymous tree clean: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
