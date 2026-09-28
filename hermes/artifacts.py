"""Apply model-proposed artifact edits as data, without executing model code."""

import json
from pathlib import Path

from hermes import pipeline as p
from hermes import trajectory as traj


def _path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("empty artifact path")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or ".git" in part.parts:
        raise ValueError(f"artifact path outside repository: {relative}")
    path = (root / part).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"artifact path outside repository: {relative}")
    return path


def apply_artifact_edit(root: Path, owner: str, edit: dict) -> list[str]:
    """Apply one primitive's own change and at most eight new artifacts."""
    if not isinstance(edit, dict) or "action" not in edit:
        raise ValueError("proposal must be a JSON object with an action")
    file = _path(root, owner)
    if edit.get("component", owner) != owner:
        raise ValueError(f"proposal names another component: {edit.get('component')}")
    action = edit.get("action", "keep")
    if action == "no_change":
        action = "keep"
    if action not in ("keep", "modify", "delete", "create"):
        raise ValueError(f"invalid action: {action}")
    exists = file.is_file()
    if action == "create" and file.exists():
        raise ValueError(f"component already exists: {owner}")
    if action != "create" and not exists:
        raise ValueError(f"component does not exist: {owner}")
    original = file.read_text(errors="replace") if exists else ""
    modified = original
    if action == "modify":
        replacements = edit.get("replacements")
        if replacements:
            if not isinstance(replacements, list):
                raise ValueError("replacements must be an array")
            for item in replacements:
                if not isinstance(item, dict):
                    raise ValueError("replacement must be an object")
                old, new = item.get("old"), item.get("new")
                if not isinstance(old, str) or not isinstance(new, str) or not old:
                    raise ValueError("invalid replacement")
                if modified.count(old) != 1:
                    raise ValueError(f"replacement anchor matched {modified.count(old)} times")
                modified = modified.replace(old, new, 1)
        else:
            modified = edit.get("updated_artifact")
            if not isinstance(modified, str):
                raise ValueError("modify requires replacements or updated_artifact")
        if modified == original:
            raise ValueError("no change to component")
        if file.suffix == ".py":
            compile(modified, owner, "exec")
    elif action == "create":
        modified = edit.get("content") or edit.get("updated_artifact")
        if not isinstance(modified, str) or not modified:
            raise ValueError("create requires nonempty content")
        if file.suffix == ".py":
            compile(modified, owner, "exec")
    additions = edit.get("new_files") or []
    if not isinstance(additions, list) or len(additions) > 8:
        raise ValueError("new_files must contain at most eight artifacts")
    created = []
    seen = set()
    for item in additions:
        if not isinstance(item, dict):
            raise ValueError("new file must be an object")
        rel, content = item.get("path"), item.get("content")
        path = _path(root, rel)
        if path.exists() or rel == owner or path in seen:
            raise ValueError(f"artifact already exists: {rel}")
        seen.add(path)
        if not isinstance(content, str) or not content:
            raise ValueError(f"new artifact is empty: {rel}")
        if path.suffix == ".py":
            compile(content, rel, "exec")
        created.append((rel, path, content))
    # Validate the full proposal before mutating the workspace.
    if action in ("modify", "create"):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(modified)
    elif action == "delete":
        file.unlink()
    for rel, path, content in created:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return ([owner] if action != "keep" else []) + [rel for rel, _, _ in created]


def edit_component(root: Path, owner: str, issue: str, objective: str,
                   messages: str = "", feedback: str = "",
                   round_index: int = 1, architecture: str = "",
                   plan: str = "", role: str = "Other",
                   must_not_break: str = "",
                   new_file_path_rule: str = "",
                   peer_paths: list[str] | None = None,
                   outgoing: list[dict] | None = None) -> list[str]:
    """Ask the owner primitive for a validated local edit proposal."""
    file = _path(root, owner)
    exists = file.is_file()
    snippet = p._clip_file(file.read_text(errors="replace")) if exists else "(file does not exist yet)"
    available_actions = "modify, delete, or no_change" if exists else "create"
    example_action = "modify" if exists else "create"
    example = {
        "component": owner,
        "action": example_action,
        "summary": "concise local reasoning",
        "changes": [{"location": "function, class, or code region",
                     "description": "what changed and why"}],
        "replacements": ([{"old": "exact unique substring",
                           "new": "replacement"}] if exists else []),
        "updated_artifact": (None if exists else "complete new artifact content"),
        "new_files": [],
        "messages": [],
        "validation_notes": [],
    }
    additions_rule = new_file_path_rule or (
        "New file paths are repository-relative.")
    prompt = f"""You are a Dev-Primitive in HERMES. You own one persistent artifact: {owner}.
Issue context: {issue}
Local objective: {objective}
Role: {role}
Dependency context and repository architecture: {architecture[:2000]}
Cross-component plan: {plan[:3000]}
Compatibility constraints: {must_not_break}
Incoming messages: {messages or '(none)'}
Other activated artifact paths: {peer_paths or []}
Execution and diagnosis feedback: {feedback or '(initial round)'}
Current artifact:
{snippet}

Reason about whether this artifact needs a change, what local code or
configuration is affected, and whether your edit changes an interface or
assumption in another activated artifact. Preserve unrelated behavior and
existing conventions. Do not guess about artifacts whose contents you cannot
inspect. Choose exactly one action: {available_actions}.

Return one JSON object with the manuscript's Dev-Primitive fields.
This is the JSON shape; replace example strings with task-specific content:
{json.dumps(example, indent=2)}

Use action no_change if no local edit is needed. For modify, provide either
exact, unique replacements or the complete updated_artifact; the runtime
applies the replacements as a patch. For create, put the complete new content
in updated_artifact. new_files may list at most eight necessary new artifacts
as {{"path":"new file path","content":"complete contents"}}.
Only edit this artifact; do not edit a peer directly. Send a message when your
local reasoning or change imposes a requirement on an activated peer. Use its
exact path as target, in the shape
{{"target":"exact activated peer path","message":"task-relevant requirement"}}.
{additions_rule} Return empty arrays when no changes,
messages, or validation notes are needed. Do not return shell commands or
Python edit scripts."""
    last_error = ""
    for attempt in range(3):
        proposal = p.llm_json(
            prompt + (f"\nPrevious proposal failed: {last_error}" if last_error else ""),
            p.role_model("primitive"), max_tokens=8192)
        try:
            changed = apply_artifact_edit(root, owner, proposal)
            delivered = []
            for item in proposal.get("messages") or []:
                target = item.get("target", item.get("to")) if isinstance(item, dict) else None
                if (isinstance(item, dict)
                        and target in (peer_paths or [])
                        and isinstance(item.get("message"), str)
                        and item["message"].strip()):
                    delivered.append({"to": target,
                                      "message": item["message"]})
            if outgoing is not None:
                outgoing.extend(delivered)
            traj.log("artifact_edit", round=round_index, file=owner,
                     changed=changed, action=proposal.get("action"),
                     summary=proposal.get("summary", ""),
                     changes=proposal.get("changes", []),
                     messages=delivered,
                     validation_notes=proposal.get("validation_notes", []))
            return changed
        except (ValueError, SyntaxError, OSError) as exc:
            last_error = str(exc)
            traj.log("artifact_edit", round=round_index, file=owner,
                     error=last_error, attempt=attempt + 1,
                     proposal=json.dumps(proposal)[:2000])
    return []
