"""Map terminal-task Dev-Primitives to persistent filesystem artifacts."""

from pathlib import PurePosixPath


_BARE_ARTIFACTS = {"Dockerfile", "Makefile", "Justfile", "Gemfile",
                   "Rakefile", "Procfile", "CMakeLists.txt", "README",
                   "LICENSE", "NOTICE"}
_TRANSIENT_ROOTS = {"dev", "proc", "sys"}
_TRANSIENT_NAMES = {".cache", "__pycache__"}
_TRANSIENT_SUFFIXES = {".log", ".tmp", ".cache", ".pyc"}


def artifact_path(workspace: str, candidate: str) -> tuple[str, str]:
    """Return (container absolute path, host mirror path relative to its root)."""
    if not isinstance(candidate, str) or not candidate or candidate != candidate.strip():
        raise ValueError("empty or malformed artifact path")
    if any(char in candidate for char in "\n\r\0`$|;&<>*?[]{}"):
        raise ValueError(f"artifact path contains shell syntax: {candidate}")
    raw = PurePosixPath(candidate)
    if (candidate in (".", "./") or ".." in raw.parts or ".git" in raw.parts
            or any(part in _TRANSIENT_NAMES for part in raw.parts)
            or candidate.endswith("/")):
        raise ValueError(f"invalid artifact path: {candidate}")
    if not raw.is_absolute() and "/" not in candidate and "." not in candidate:
        if candidate not in _BARE_ARTIFACTS:
            raise ValueError(f"expected a file path, received: {candidate}")
    workspace_path = PurePosixPath(workspace)
    if not workspace_path.is_absolute():
        raise ValueError(f"workspace is not absolute: {workspace}")
    absolute = raw if raw.is_absolute() else workspace_path / raw
    if (not absolute.name or absolute.parts[1] in _TRANSIENT_ROOTS
            or absolute.suffix in _TRANSIENT_SUFFIXES):
        raise ValueError(f"transient or invalid artifact path: {candidate}")
    return str(absolute), str(absolute).lstrip("/")


def components(value: object, workspace: str) -> list[dict]:
    """Accept artifact paths only; command names and process state are not owners."""
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        path, objective = item.get("path"), item.get("objective")
        if not isinstance(objective, str) or not objective.strip():
            continue
        try:
            absolute, owner = artifact_path(workspace, path)
        except ValueError:
            continue
        if absolute in seen:
            continue
        result.append({"name": absolute, "owner": owner,
                       "objective": objective.strip()})
        seen.add(absolute)
        if len(result) == 6:
            break
    return result


def selection_prompt(instruction: str, observation: str, feedback: str = "") -> str:
    return (
        f"Task: {instruction}\nTask-visible environment:\n{observation[-4000:]}\n"
        f"Critic feedback: {feedback[-3000:] or '(initial selection)'}\n"
        "Select up to six Dev-Primitives. Each primitive owns one persistent "
        "filesystem artifact that can be independently inspected or edited: "
        "a source file, shell script, configuration file, service definition, "
        "build file, or required output file. Select a required output path "
        "even if that file does not yet exist. Use absolute paths for files "
        "outside the current working directory. Do not select a shell command, "
        "monitoring action, runtime process, terminal output, or transient "
        "system state as a primitive. Return JSON "
        '{"components":[{"path":"/absolute/or/relative/file",'
        '"objective":"local goal"}]}.'
    )
