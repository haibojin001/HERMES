"""HERMES: harness engineering for software engineering via modular executable
Dev-Primitives. Reference implementation for SWE-bench Verified.

One module, because the stages share the LLM wrapper, the trajectory recorder and
the mutable active set, and splitting them would mean threading that state
through signatures without making any of it easier to read. The stages appear in
execution order and each is labelled with the equation of the paper it realises:

    Stage 1  PLANNER                 Eq. 2   Pi = PLANNER(q, R)
             on-demand activation    Eq. 3   P_A = {P_i | i in A}
    Stage 2  Dev-Primitive           Eq. 1,5 (a_i', m_i) = P_i(a_i, x_i, C_i)
             collaboration                   messages are addressed, not broadcast
    Stage 3  EXECUTE                 Eq. 6   o = EXECUTE(R')
    Stage 4  CRITIC                  Eq. 7   v = CRITIC(q, Pi, R', o)
             structured feedback     Eq. 8   phi = (e, c, u)
    Stage 5  re-planning             Eq. 9   Pi' = PLANNER(q, R', Pi, phi)
    Stage 6  held-out evaluation             runs only after termination

`docs/paper-to-code.md` maps every equation, table and appendix claim to the
function that implements it.

Usage:
    python -m hermes.pipeline --instance django__django-13512
    python -m hermes.pipeline --vllm Qwen/Qwen3-8B --max-iterations 3
    python -m hermes.pipeline --ablate-communication    # one ablation row

Prerequisites: the SWE-bench environment images for the instances you run, and a
model backend (Bedrock by default, or --vllm / --ollama). See README.md.
"""

import os
import sys
import json
import subprocess
import time
import base64
import re
import traceback
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from hermes import settings
from hermes import trajectory as traj

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ============================================================
# Configuration
# ============================================================
WORK_DIR = settings.WORK_DIR
DATASET_PATH = settings.DATASET_PATH
OUTPUT_PATH = settings.PREDICTIONS_PATH
TRAJ_ROOT = settings.TRAJ_ROOT

# Bedrock credentials come from the standard AWS environment or profile
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION_NAME, or
# AWS_PROFILE). litellm reads them directly; nothing is set here.

TRIAGE_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
THINKING_MODEL = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"

# Per-role backbone slots, for the backbone-scaling table: the Planner, the
# Dev-Primitives and the Critic can each run a different model. None means "use
# THINKING_MODEL", so a homogeneous run needs no flags at all and a
# heterogeneous one names only the slots it moves.
#
# Two slots stay outside this: bug localization keeps TRIAGE_MODEL, because it is
# one short call per candidate file and thousands per run, and the execution
# scaffolding (reproduction script, repository test command, test-error
# extraction) keeps THINKING_MODEL, because it is not one of the three roles and
# moving it with a role would confound the comparison.
PLANNER_MODEL = None
PRIMITIVE_MODEL = None
CRITIC_MODEL = None


def role_model(role: str) -> str:
    """The backbone for one role, resolved at call time so the CLI can set it."""
    return {"planner": PLANNER_MODEL,
            "primitive": PRIMITIVE_MODEL,
            "critic": CRITIC_MODEL}.get(role) or THINKING_MODEL

MAX_RETRIES = 3
MAX_ITERATIONS = 5
MAX_RELEVANT_FILES = 6
# How many triage survivors the ranking call is allowed to see at once. Triage on
# a Django-sized repository can flag 500 files, and their paths plus reasons do
# not fit in a 24k context.
RANK_CANDIDATES = 60
# File content sent to each agent during inter-file negotiation (Phase 3.6).
NEGOTIATE_FILE_CHARS = 60000
TRIAGE_CONCURRENCY = 8
INSTANCE_CONCURRENCY = 1  # Sequential for docker (resource-heavy)

# Non-Python Dev-Primitives. Kept shallow: a deeply nested .cfg/.ini is almost
# always package metadata or test data, not behaviour the fix has to change.
CONFIG_SUFFIXES = (".yaml", ".yml", ".toml", ".ini", ".cfg")
SOURCE_SUFFIXES = (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
                   ".java", ".sh", ".rb", ".c", ".cc", ".cpp", ".h",
                   ".hpp", ".zig", ".cs", ".kt", ".scala", ".bash",
                   ".sql", ".php", ".swift")
GENERIC_COMPONENTS = False
ALLOW_TEST_EDITS = False
# Localization call payloads are omitted from the trajectory to keep it small;
# their counts and token usage are retained in summary.json.
TRIAGE_PHASE = "1.1_BUG_LOCALIZATION_TRIAGE"
CONFIG_MAX_DEPTH = 3
# The role label the method figure prints on each Dev-Primitive box. Assigned by
# the planner's task-decomposition stage.
PRIMITIVE_ROLES = ("API Logic", "Core Class", "Shared Utilities", "Config",
                   "Tests", "Other")

MODEL_NAME = "hermes"
# Appended to the trajectory dir name so runs with different models coexist.
TRAJ_SUFFIX = ""

# ------------------------------------------------------------
# Ablation switches.
#
# Each one disables exactly one mechanism and leaves the backbone, reasoning
# effort, re-planning budget B and execution environment untouched, so a row of
# the component ablation differs from the complete configuration in one place
# only. All default to False, i.e. complete HERMES.
#
# The re-planning budget B is not here: it is `--max-iterations`, because B is a
# parameter of the method rather than a mechanism being removed.
# ------------------------------------------------------------
ABLATE_COMMUNICATION = False        # w/o Inter-Primitive Communication
ABLATE_ON_DEMAND = False            # w/o On-Demand Activation
ABLATE_EXECUTION_FEEDBACK = False   # w/o Execution Feedback
ABLATE_CRITIC = False               # w/o Critic Feedback
# `w/o On-Demand Activation` activates everything the initial task analysis
# identified. On a Django-sized repository bug localization flags several hundred
# components, and activating all of them costs one edit call each per round, which
# does not finish. We therefore activate the candidate pool the ranking stage sees
# (RANK_CANDIDATES) rather than every triage survivor, and report the cap. Set to
# 0 for no cap.
ABLATE_ON_DEMAND_CAP = 60


def ablation_label() -> str:
    """Which configuration is running, for the trajectory header."""
    off = [n for n, v in (("communication", ABLATE_COMMUNICATION),
                          ("on_demand_activation", ABLATE_ON_DEMAND),
                          ("execution_feedback", ABLATE_EXECUTION_FEEDBACK),
                          ("critic_feedback", ABLATE_CRITIC)) if v]
    return "complete" if not off else "w/o " + ", ".join(off)

# ------------------------------------------------------------
# Model backend. "bedrock" is the frontier-model configuration the main
# results use; "ollama" runs the identical pipeline against a local open
# model so the method can be reported independently of model scale.
# ------------------------------------------------------------
BACKEND = "bedrock"
HOSTED_REASONING_EFFORT = None
# Chars of file content a single prompt may carry. None = unlimited (frontier
# models have the context for a whole file); a small local model does not.
FILE_CONTEXT_CHARS = None
# Local reasoning models emit a thinking block that must not reach the JSON /
# code-block parsers.
STRIP_THINK_TAGS = False
# Extra kwargs handed to litellm.completion (e.g. Ollama's context length).
LLM_EXTRA_KWARGS = {}
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_NUM_CTX = 32768
OLLAMA_TIMEOUT = 900
OLLAMA_MODEL = None
OLLAMA_TEMPERATURE = 0.6
OLLAMA_TOP_P = 0.95
OLLAMA_TOP_K = 20
# Reasoning on the planning / negotiation / edit calls, off for bulk triage.
OLLAMA_THINK = True


def use_ollama(model="qwen3:8b", num_ctx=32768, concurrency=4, think=True,
               url=None):
    """Point the whole pipeline at a locally served open model."""
    global TRIAGE_MODEL, THINKING_MODEL, BACKEND, FILE_CONTEXT_CHARS
    global STRIP_THINK_TAGS, NEGOTIATE_FILE_CHARS, OLLAMA_NUM_CTX, OLLAMA_THINK
    global TRIAGE_CONCURRENCY, MODEL_NAME, OLLAMA_MODEL, OLLAMA_URL
    BACKEND = "ollama"
    TRIAGE_MODEL = THINKING_MODEL = model
    # Leave room for the surrounding prompt and the response inside num_ctx;
    # ~16k chars is roughly 5k tokens of file content.
    FILE_CONTEXT_CHARS = 16000
    NEGOTIATE_FILE_CHARS = 16000
    STRIP_THINK_TAGS = True
    OLLAMA_NUM_CTX = num_ctx
    OLLAMA_MODEL = model
    if url:
        OLLAMA_URL = url
    OLLAMA_THINK = think
    TRIAGE_CONCURRENCY = concurrency
    MODEL_NAME = f"hermes-{model}"


# ------------------------------------------------------------
# OpenAI-compatible backend: a vLLM server (e.g. Qwen3-8B on a cluster A100,
# reached through an SSH tunnel). Continuous batching makes the thousands of
# per-file triage calls tractable, which they are not on a laptop.
# ------------------------------------------------------------
VLLM_URL = "http://localhost:8000/v1/chat/completions"
VLLM_TIMEOUT = 900
VLLM_MODEL = None


def use_hosted(model: str, reasoning_effort: str | None = None) -> None:
    """Configure a hosted model without carrying local-backend state."""
    global BACKEND, THINKING_MODEL, TRIAGE_MODEL, MODEL_NAME
    global HOSTED_REASONING_EFFORT, STRIP_THINK_TAGS, FILE_CONTEXT_CHARS
    if not model:
        raise ValueError("hosted model id is required")
    if reasoning_effort not in (None, "none", "minimal", "low", "medium",
                                "high", "xhigh", "max"):
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
    BACKEND = "hosted"
    THINKING_MODEL = TRIAGE_MODEL = model
    HOSTED_REASONING_EFFORT = reasoning_effort
    STRIP_THINK_TAGS = False
    FILE_CONTEXT_CHARS = None
    MODEL_NAME = f"hermes-{model.split('/')[-1]}"


def _openai_response(model: str, prompt: str, max_tokens: int,
                     no_think: bool):
    from openai import OpenAI

    effort = "low" if no_think and HOSTED_REASONING_EFFORT else HOSTED_REASONING_EFFORT
    options = {
        "model": model.removeprefix("openai/"),
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": max(
            max_tokens, 8192 if effort in ("medium", "high", "xhigh", "max")
            else 4096 if effort == "low" else max_tokens),
        "store": False,
    }
    if effort:
        options["reasoning"] = {"effort": effort}
    client = OpenAI()
    response = client.responses.create(**options)
    usage_items = [response.usage] if response.usage else []
    if (response.status == "incomplete"
            and getattr(response.incomplete_details, "reason", None)
            == "max_output_tokens"):
        options["max_output_tokens"] = max(
            25000, 2 * options["max_output_tokens"])
        response = client.responses.create(**options)
        if response.usage:
            usage_items.append(response.usage)
    if response.status != "completed":
        raise RuntimeError(
            f"OpenAI response did not complete (status={response.status}, "
            f"details={response.incomplete_details})")
    text = (response.output_text or "").strip()
    if not text:
        raise RuntimeError(
            f"OpenAI returned no text (status={response.status}, "
            f"details={response.incomplete_details})")
    return text, (_Usage(
        sum(item.input_tokens for item in usage_items),
        sum(item.output_tokens for item in usage_items))
        if usage_items else None)


def _anthropic_response(model: str, prompt: str, max_tokens: int,
                        no_think: bool):
    from anthropic import Anthropic

    effort = "low" if no_think and HOSTED_REASONING_EFFORT else HOSTED_REASONING_EFFORT
    if effort in ("none", "minimal"):
        raise ValueError("Claude effort must be low, medium, high, xhigh or max")
    options = {
        "model": model.removeprefix("anthropic/"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(
            max_tokens, 8192 if effort in ("medium", "high", "xhigh", "max")
            else max_tokens),
    }
    if effort:
        options["output_config"] = {"effort": effort}
    client = Anthropic()
    response = client.messages.create(**options)
    usage_items = [response.usage]
    if response.stop_reason == "max_tokens":
        options["max_tokens"] = max(16000, 2 * options["max_tokens"])
        response = client.messages.create(**options)
        usage_items.append(response.usage)
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude output reached max_tokens twice")
    text = "\n".join(
        block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise RuntimeError(f"Claude returned no text (stop_reason={response.stop_reason})")
    return text, _Usage(
        sum(item.input_tokens for item in usage_items),
        sum(item.output_tokens for item in usage_items))


def use_vllm(model="Qwen/Qwen3-8B", url=None, concurrency=32,
             file_chars=16000, think=True):
    """Point the whole pipeline at an OpenAI-compatible vLLM server."""
    global TRIAGE_MODEL, THINKING_MODEL, BACKEND, FILE_CONTEXT_CHARS
    global STRIP_THINK_TAGS, NEGOTIATE_FILE_CHARS, TRIAGE_CONCURRENCY
    global MODEL_NAME, VLLM_URL, OLLAMA_THINK, VLLM_MODEL
    BACKEND = "vllm"
    TRIAGE_MODEL = THINKING_MODEL = model
    VLLM_MODEL = model
    FILE_CONTEXT_CHARS = file_chars
    NEGOTIATE_FILE_CHARS = file_chars
    # Qwen3 emits <think>...</think> inline over the OpenAI-compatible route.
    STRIP_THINK_TAGS = True
    OLLAMA_THINK = think
    TRIAGE_CONCURRENCY = concurrency
    MODEL_NAME = f"hermes-{model.split('/')[-1]}"
    if url:
        VLLM_URL = url


def _vllm_chat(model, prompt, max_tokens, think):
    import urllib.request, urllib.error
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # vLLM's Qwen3 chat template reads this to gate the thinking block.
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    req = urllib.request.Request(
        VLLM_URL, json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=VLLM_TIMEOUT) as resp:
            d = json.load(resp)
    except urllib.error.HTTPError as e:
        # A bare "HTTP Error 400: Bad Request" is undiagnosable after the fact,
        # and 400 is how vLLM reports a prompt that exceeds max_model_len - the
        # one failure mode that kills a whole run. Keep the server's own message.
        body = ""
        try:
            body = e.read().decode(errors="replace")[:600]
        except Exception:
            pass
        raise RuntimeError(f"vLLM {e.code}: {body} "
                           f"[prompt {len(prompt)} chars, "
                           f"max_tokens={max_tokens}]") from None
    text = (d["choices"][0]["message"].get("content") or "").strip()
    u = d.get("usage") or {}
    return text, _Usage(u.get("prompt_tokens") or 0,
                        u.get("completion_tokens") or 0)


def _clip_file(display: str) -> str:
    """Trim embedded file content to what the active backend can actually read."""
    if FILE_CONTEXT_CHARS is None or len(display) <= FILE_CONTEXT_CHARS:
        return display
    half = FILE_CONTEXT_CHARS // 2
    return (display[:half]
            + f"\n... [{len(display) - FILE_CONTEXT_CHARS} chars omitted] ...\n"
            + display[-half:])


_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think>, including an unclosed block at the end."""
    text = _THINK_RE.sub('', text)
    if '<think>' in text:  # truncated before </think>
        text = text.split('<think>')[0]
    return text.strip()


# ============================================================
# LLM Interface
# ============================================================
class _Usage:
    """Minimal stand-in for litellm's usage object, for the trajectory recorder."""
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def _ollama_chat(model, prompt, max_tokens, think):
    """Call Ollama's /api/chat directly.

    litellm's ollama_chat path routes a reasoning model's thinking into
    reasoning_content and leaves content empty, and Qwen3 ignores the /no_think
    soft switch, so the thinking budget silently eats the whole token allowance.
    Ollama's structured `think` flag is the only reliable control.
    """
    import urllib.request
    body = json.dumps({
        "model": model.removeprefix("ollama/"),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": bool(think),
        "options": {"num_ctx": OLLAMA_NUM_CTX,
                    "temperature": OLLAMA_TEMPERATURE,
                    "top_p": OLLAMA_TOP_P, "top_k": OLLAMA_TOP_K,
                    "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body,
                                {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
        d = json.load(resp)
    msg = d.get("message") or {}
    return (msg.get("content") or "").strip(), \
        _Usage(d.get("prompt_eval_count") or 0, d.get("eval_count") or 0), \
        (msg.get("thinking") or "")


def llm(prompt: str, model: str = None, max_tokens: int = 8192,
        no_think: bool = False) -> str:
    # Resolved at call time, not import time, so use_ollama() can retarget it.
    model = model or THINKING_MODEL
    # Triage calls are dropped from the trajectory (thousands of them). Identify
    # them by phase, not by model: under a single-model backend TRIAGE_MODEL ==
    # THINKING_MODEL, so matching on the model name would silently drop every
    # plan/negotiate/edit call too. Exact match, not a prefix - a prefix test
    # swallowed the ranking call, which runs in a sibling phase of the same box.
    is_triage = traj.phase() == TRIAGE_PHASE
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            # Reasoning is worth its cost on the few planning calls, but not on
            # the thousands of per-file relevance judgements.
            think = OLLAMA_THINK and not no_think
            backend = ("vllm" if model.startswith("vllm/")
                       or (BACKEND == "vllm" and model == VLLM_MODEL)
                       else "ollama" if model.startswith("ollama/")
                       or (BACKEND == "ollama" and model == OLLAMA_MODEL)
                       else "hosted")
            if backend == "vllm":
                served_model = model.removeprefix("vllm/")
                text, usage = _vllm_chat(served_model, prompt, max_tokens, think)
                traj.log_llm_call(model, prompt, text, time.time() - t0,
                                  usage=usage, is_triage=is_triage)
                # Truncation mid-<think> leaves text non-empty but with no answer
                # in it, so test emptiness after stripping, not before.
                if think and not _strip_think(text):
                    t0 = time.time()
                    text, usage = _vllm_chat(served_model, prompt, max_tokens, False)
                    traj.log_llm_call(model, prompt, text, time.time() - t0,
                                      usage=usage, is_triage=is_triage)
            elif backend == "ollama":
                text, usage, thinking = _ollama_chat(model, prompt, max_tokens, think)
                traj.log_llm_call(model, prompt, text, time.time() - t0,
                                  usage=usage, is_triage=is_triage)
                if think and not _strip_think(text):
                    # Budget was consumed by thinking; redo without it.
                    t0 = time.time()
                    text, usage, thinking = _ollama_chat(
                        model, prompt, max_tokens, False)
                    traj.log_llm_call(model, prompt, text, time.time() - t0,
                                      usage=usage, is_triage=is_triage)
            elif model.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
                text, usage = _openai_response(
                    model, prompt, max_tokens, no_think)
                traj.log_llm_call(model, prompt, text, time.time() - t0,
                                  usage=usage, is_triage=is_triage)
            elif model.startswith(("anthropic/", "claude-")):
                text, usage = _anthropic_response(
                    model, prompt, max_tokens, no_think)
                traj.log_llm_call(model, prompt, text, time.time() - t0,
                                  usage=usage, is_triage=is_triage)
            else:
                import litellm
                options = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
                if HOSTED_REASONING_EFFORT:
                    options["reasoning_effort"] = (
                        "low" if no_think else HOSTED_REASONING_EFFORT)
                r = litellm.completion(**options)
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError(f"{model} returned no text")
                usage = getattr(r, "usage", None)
                traj.log_llm_call(model, prompt, text, time.time() - t0,
                                  usage=usage, is_triage=is_triage)
            if backend in ("ollama", "vllm"):
                text = _strip_think(text)
            return text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"LLM failed: {e}")


def _parse_json_object(text: str):
    """Parse the first JSON object in `text`, or None. Repairs illegal escapes."""
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    candidates = [m.group()] if m else []
    # Small models routinely write regexes straight into a JSON string, e.g.
    #   "reason": "use \A[\w.@+-]+\Z instead"
    # \A \w \Z are not JSON escapes, so json.loads rejects the whole object and
    # the caller sees a parse failure. Since the files most worth flagging are
    # the ones whose explanation mentions a regex, this bias is not random: it
    # silently dropped django/contrib/auth/validators.py from LOCATE. Escape any
    # backslash that does not begin a legal JSON escape and try again.
    # Consume legal escapes (including \\) as a unit, then double whatever lone
    # backslashes are left; a plain negative lookahead would mangle \\A into \\\A.
    candidates += [re.sub(r'\\(["\\/bfnrtu])|\\',
                          lambda mo: '\\' + mo.group(1) if mo.group(1) else '\\\\', c)
                   for c in list(candidates)]
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def llm_json(prompt: str, model: str = None, max_tokens: int = 8192,
             no_think: bool = False) -> dict:
    prompt = prompt + "\n\nRespond with ONLY valid JSON."
    text = llm(prompt, model, max_tokens, no_think=no_think)
    obj = _parse_json_object(text)
    if isinstance(obj, dict):
        return obj
    # The dominant failure mode on a small reasoning model is not malformed JSON
    # but *truncated* JSON: <think> eats the completion budget and the object is
    # cut off mid-string. llm()'s empty-answer guard does not catch it because a
    # half-written object is not empty. Retry once with thinking off, which hands
    # the whole budget to the answer. Silently falling through here is what
    # emptied the ranking list and the task decomposition.
    if not no_think:
        text2 = llm(prompt, model, max_tokens, no_think=True)
        obj = _parse_json_object(text2)
        if isinstance(obj, dict):
            traj.log("json_retry", recovered=True, first_chars=len(text))
            return obj
        text = text2
    return {"_raw": text[:500], "_error": "parse failed"}


# ============================================================
# File Operations
# ============================================================
def numbered_content(filepath: Path) -> str:
    lines = filepath.read_text(errors="replace").split('\n')
    return '\n'.join(f"{i+1:>5}|{line}" for i, line in enumerate(lines))


def _extract_test_errors(output: str) -> str:
    """Extract the meaningful error parts from test output.
    Prioritize tracebacks and assertion errors over boilerplate."""
    # Narrow to the real test output first; the eval script runs with `set -x`,
    # so the surrounding trace would otherwise drown out the actual failures.
    if ">>>>> Start Test Output" in output:
        output = output.split(">>>>> Start Test Output")[-1].split(">>>>> End Test Output")[0]

    lines = output.split('\n')

    # Find lines with errors/failures
    error_lines = []
    in_error = False
    for i, line in enumerate(lines):
        if any(kw in line for kw in ['FAIL:', 'ERROR:', 'Traceback', 'AssertionError',
                                       'TypeError', 'ValueError', 'AttributeError',
                                       'ImportError', 'NameError', 'KeyError']):
            in_error = True
        if in_error:
            error_lines.append(line)
            if len(error_lines) > 150:
                break
        if in_error and line.strip() == '' and len(error_lines) > 5:
            in_error = False

    if error_lines:
        return '\n'.join(error_lines[:150])

    # Fallback: return full output (up to 4000 chars)
    return output[:4000]


# ============================================================
# Docker (Finch) Runner
# ============================================================
class ContainerRunner:
    """Run the repository in a container: the task-visible execution of Stage 3
    and, after termination, the held-out SWE-bench eval script of Stage 6."""

    def __init__(self, instance_id: str, test_spec):
        self.instance_id = instance_id
        self.test_spec = test_spec
        self.image = test_spec.instance_image_key.replace(
            ":", f"{settings.IMAGE_SUFFIX}:")
        # We'll use the env image directly + setup at runtime
        self.env_image = test_spec.env_image_key

    def ensure_image(self) -> bool:
        """Check if instance image exists, build if not."""
        result = subprocess.run(
            [settings.CONTAINER_CLI, "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True
        )
        if self.env_image in result.stdout:
            return True
        # Try building it
        return self._build_instance_image()

    def _build_instance_image(self) -> bool:
        """Build the instance image from env image using Finch."""
        # Write Dockerfile
        dockerfile = self.test_spec.instance_dockerfile
        # Replace image reference to match our naming
        build_dir = WORK_DIR / f"_docker_build_{self.instance_id}"
        build_dir.mkdir(exist_ok=True)

        (build_dir / "Dockerfile").write_text(dockerfile)

        # Write setup_repo.sh
        repo_script = '\n'.join(self.test_spec.repo_script_list)
        (build_dir / "setup_repo.sh").write_text(repo_script)

        try:
            result = subprocess.run(
                [settings.CONTAINER_CLI, "build", "-t", self.image, str(build_dir)],
                capture_output=True, text=True, timeout=600
            )
            return result.returncode == 0
        except Exception:
            return False

    def run_tests(self, repo_dir: Path, debug_script: str = "") -> dict:
        """Apply model patch + run the official SWE-bench eval script in a container.

        The eval script is what grades submissions: it restores the gold TEST
        files (a fix must satisfy the post-fix tests, not the pre-fix ones),
        installs the project, and runs FAIL_TO_PASS + PASS_TO_PASS. Rolling our
        own test command instead makes the loop optimise against the wrong target.
        """
        diff = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=repo_dir)
        if not diff.stdout:
            return {"passed": False, "error": "no diff", "output": "", "test_status": {}}

        # Write patch + official eval script to temp files and mount them
        patch_file = WORK_DIR / f"_patch_{self.instance_id}.diff"
        patch_file.write_text(diff.stdout)
        eval_file = WORK_DIR / f"_eval_{self.instance_id}.sh"
        eval_file.write_text(self.test_spec.eval_script)

        debug_section = ""
        if debug_script:
            debug_file = WORK_DIR / f"_debug_{self.instance_id}.py"
            debug_file.write_text(debug_script)
            debug_section = """
echo "=== DEBUG ==="
cd /testbed && python /mnt/debug.py 2>&1 | tail -50
"""

        script = f"""
source /opt/miniconda3/bin/activate
conda activate testbed

# Older Django/pytest write non-ASCII to stdout; without this the test runner
# dies with UnicodeEncodeError before a single test executes.
export PYTHONIOENCODING=utf-8

# Copy repo into container
mkdir -p /testbed && cp -a /mnt/repo/. /testbed/
cd /testbed
git config --global --add safe.directory /testbed

# The mounted working tree already carries our edits. Reset to the base commit
# so applying patch.diff is a real check rather than a guaranteed conflict.
git checkout . >/dev/null 2>&1

# Apply our model patch
echo "=== APPLY ==="
git apply /mnt/patch.diff 2>&1
APPLY_RC=$?
if [ $APPLY_RC -ne 0 ]; then
    echo "PATCH_APPLY_FAILED"
    git apply --3way /mnt/patch.diff 2>&1
fi

# Official SWE-bench evaluation: restores gold test files, installs, runs tests
echo "=== EVAL ==="
bash /mnt/eval.sh 2>&1
{debug_section}
"""
        run_cmd = [
            settings.CONTAINER_CLI, "run", "--rm",
            "-v", f"{repo_dir}:/mnt/repo:ro",
            "-v", f"{patch_file}:/mnt/patch.diff:ro",
            "-v", f"{eval_file}:/mnt/eval.sh:ro",
        ]
        if debug_script:
            run_cmd += ["-v", f"{WORK_DIR / f'_debug_{self.instance_id}.py'}:/mnt/debug.py:ro"]
        run_cmd += [self.env_image, "bash", "-c", script]

        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout", "output": "timeout", "test_status": {}}

        output = result.stdout + result.stderr
        apply_failed = "PATCH_APPLY_FAILED" in output
        passed, status, error = self._grade(output)
        if apply_failed and not passed:
            error = f"patch did not apply cleanly; {error or 'tests not trusted'}"
        return {"passed": passed, "output": output,
                "test_status": status, "error": error}

    def _setup_commands(self) -> str:
        """The repository's own build/install commands, and nothing after them.

        Taken from the harness's own script because that is where the correct
        per-instance install incantation lives, but truncated at the first line
        that touches the benchmark's test files: everything from that point on
        restores gold tests, applies the gold test patch and runs FAIL_TO_PASS /
        PASS_TO_PASS, none of which may be visible during solving. What survives
        the cut is `pip install -e .` and its neighbours - build commands, which
        the method explicitly allows.
        """
        keep = []
        for line in self.test_spec.eval_script_list:
            s = line.strip()
            if (s.startswith("git checkout") or s.startswith("git apply")
                    or ">>>>> Start Test Output" in s):
                break
            # `set -u` aborts our script on any unset variable and `-x` floods
            # the log; the shebang is meaningless mid-script.
            if s.startswith("#!") or s.startswith("set "):
                continue
            keep.append(line)
        return "\n".join(keep)

    def run_task_visible(self, repo_dir: Path, repro_script: str = "",
                         repo_test_cmd: str = "", timeout: int = 900) -> dict:
        """EXECUTE(R') using only task-visible signals.

        Deliberately does NOT use `test_spec.eval_script`. That script restores
        the benchmark's gold test files and runs FAIL_TO_PASS / PASS_TO_PASS,
        which are held-out evaluation tests: the method reserves them for final
        scoring and forbids returning them as feedback during solving. What runs
        here instead is (a) the reproduction script the solver wrote from the
        issue text and (b) a command from the repository's own test suite that
        the solver chose. Nothing in the returned dict names a benchmark test.

        Returns the four observations EXECUTE(R') is defined to produce:
        o_shell, o_test, o_runtime, o_trace.
        """
        mounts, sections = [], []
        if repro_script:
            repro_file = WORK_DIR / f"_repro_{self.instance_id}.py"
            repro_file.write_text(repro_script)
            mounts += ["-v", f"{repro_file}:/mnt/repro.py:ro"]
            # Copied into /testbed rather than run from /mnt: `python /mnt/repro.py`
            # puts /mnt on sys.path instead of the repository, so every import of
            # the project under test fails. And the exit code is captured before
            # the pipe - `cmd | tail` reports tail's status, which is always 0,
            # which silently turned every failing reproduction into a pass.
            sections.append("""
echo "===REPRO_BEGIN==="
cp /mnt/repro.py /testbed/_hermes_repro.py
cd /testbed && timeout 300 python _hermes_repro.py > /tmp/repro.out 2>&1
REPRO_RC=$?
tail -80 /tmp/repro.out
echo "===REPRO_RC=$REPRO_RC==="
""")
        if repo_test_cmd:
            # Single line, run verbatim; the solver is responsible for choosing a
            # command that exists in this repository.
            sections.append(f"""
echo "===REPOTEST_BEGIN==="
cd /testbed && timeout 600 {repo_test_cmd} > /tmp/repotest.out 2>&1
REPOTEST_RC=$?
tail -120 /tmp/repotest.out
echo "===REPOTEST_RC=$REPOTEST_RC==="
""")

        script = f"""
source /opt/miniconda3/bin/activate
conda activate testbed
export PYTHONIOENCODING=utf-8

mkdir -p /testbed && cp -a /mnt/repo/. /testbed/
cd /testbed
git config --global --add safe.directory /testbed

# Make the project importable. Without this the env image has the dependencies
# but not the project itself, and every reproduction script dies on its first
# `import`. These are the repository's own build commands.
echo "===SETUP_BEGIN==="
{self._setup_commands()}
echo "===SETUP_RC=$?==="

echo "===SHELL_BEGIN==="
python -c "import sys; print('python', sys.version.split()[0])"
git --no-pager diff --stat
echo "===SHELL_RC=$?==="
{''.join(sections)}
"""
        run_cmd = [settings.CONTAINER_CLI, "run", "--rm", "-v", f"{repo_dir}:/mnt/repo:ro"] \
            + mounts + [self.env_image, "bash", "-c", script]
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True,
                                    timeout=timeout)
            output = result.stdout + result.stderr
            container_rc = result.returncode
        except subprocess.TimeoutExpired:
            output = "===REPRO_BEGIN===\nexecution timed out\n===REPRO_RC=124==="
            container_rc = 124

        def _section(name):
            if f"==={name}_BEGIN===" not in output:
                return "", None
            body = output.split(f"==={name}_BEGIN===", 1)[1]
            rc = None
            m = re.search(rf"==={name}_RC=(\d+)===", body)
            if m:
                rc, body = int(m.group(1)), body[:m.start()]
            return body.strip(), rc

        shell, _ = _section("SHELL")
        setup_out, setup_rc = _section("SETUP")
        repro_out, repro_rc = _section("REPRO")
        test_out, test_rc = _section("REPOTEST")
        if setup_rc not in (None, 0):
            # A failed install poisons everything downstream, and the resulting
            # ImportError looks exactly like a bug in the patch. Say so.
            shell = (f"WARNING: project install failed (exit {setup_rc}); "
                     f"import errors below are environment, not the patch.\n"
                     + setup_out[-1500:] + "\n" + shell)
        # Runtime and trace are the exception/traceback material inside whatever
        # actually ran; they are separated out because the Critic is specified to
        # receive them as distinct observations.
        combined = f"{repro_out}\n{test_out}"
        trace = '\n'.join(re.findall(r'(?:Traceback \(most recent call last\):(?:\n.*)+?)'
                                     r'(?=\n\S|\Z)', combined)) or ""
        runtime = '\n'.join(l for l in combined.splitlines()
                            if re.match(r'\s*\w*(Error|Exception|Warning)\b', l)
                            or 'assert' in l.lower())
        return {
            "shell": shell,
            "test": test_out,
            "runtime": runtime[:4000],
            "trace": trace[:4000],
            "repro_output": repro_out,
            "repro_rc": repro_rc,
            "repo_test_rc": test_rc,
            "repo_test_cmd": repo_test_cmd,
            "setup_rc": setup_rc,
            "shell_rc": _section("SHELL")[1],
            "container_rc": container_rc,
            "output": output,
        }

    def _grade(self, output: str) -> tuple:
        """Grade with swebench's own log parser: every FAIL_TO_PASS and
        PASS_TO_PASS test must report PASSED. Returns (passed, status, error)."""
        from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
        from swebench.harness.constants import TestStatus

        if ">>>>> Start Test Output" in output:
            log = output.split(">>>>> Start Test Output")[-1]
            log = log.split(">>>>> End Test Output")[0]
        else:
            return False, {}, "no test output markers: eval script never ran tests"

        parser = MAP_REPO_TO_PARSER.get(self.test_spec.repo)
        if parser is None:
            return False, {}, f"no log parser for {self.test_spec.repo}"
        try:
            status = parser(log, self.test_spec)
        except Exception as e:
            return False, {}, f"log parse failed: {type(e).__name__}: {e}"

        def _tests(raw):
            return json.loads(raw) if isinstance(raw, str) else (raw or [])

        ftp = _tests(self.test_spec.FAIL_TO_PASS)
        ptp = _tests(self.test_spec.PASS_TO_PASS)

        missing, failing = [], []
        for t in ftp + ptp:
            st = status.get(t)
            if st is None:
                missing.append(t)
            elif st != TestStatus.PASSED.value:
                failing.append(f"{t} -> {st}")

        if not missing and not failing:
            return True, status, None
        parts = []
        if failing:
            parts.append(f"{len(failing)} not passing (e.g. {failing[0]})")
        if missing:
            parts.append(f"{len(missing)} never ran (e.g. {missing[0]})")
        return False, status, "; ".join(parts)

    @property
    def test_command(self) -> str:
        """The test command the official eval script actually runs."""
        script = self.test_spec.eval_script
        if ">>>>> Start Test Output" in script:
            body = script.split(">>>>> Start Test Output")[-1]
            body = body.split(">>>>> End Test Output")[0]
            # Markers are shell no-ops (`: '>>>>> ...'`), so the split leaves
            # dangling quote fragments around the real command.
            cmds = [l for l in body.splitlines()
                    if l.strip() and not l.strip().startswith(("'", ":"))]
            if cmds:
                return "\n".join(cmds).strip()
        return self._build_test_cmd()

    def _build_test_cmd(self) -> str:
        """Legacy hand-rolled test command. Kept as a fallback only; the real
        grading path is the official eval script (see run_tests)."""
        ftp = self.test_spec.FAIL_TO_PASS
        if isinstance(ftp, str):
            ftp = json.loads(ftp)

        if "django" in self.instance_id:
            labels = set()
            for t in ftp:
                if "(" in t:
                    mod = t.split("(")[1].rstrip(")").strip().split(".")[0]
                    labels.add(mod)
            if not labels:
                labels.add(".")
            return f"cd /testbed/tests && python runtests.py --settings=test_sqlite --parallel 1 --verbosity 2 {' '.join(labels)}"
        elif "sympy" in self.instance_id:
            return f"cd /testbed && python -m pytest {' '.join(ftp)} -x"
        elif "astropy" in self.instance_id:
            return f"cd /testbed && python -m pytest {' '.join(ftp)} -xvs"
        else:
            return f"cd /testbed && python -m pytest {' '.join(ftp)} -x"

    def _parse_passed(self, test_section: str, full_output: str) -> bool:
        if "django" in self.instance_id:
            # Django: look for "OK" anywhere in test section, no "FAIL"
            if "OK" in test_section and "FAIL" not in test_section:
                return True
            return False
        # pytest
        if "passed" in test_section and "failed" not in test_section and "error" not in test_section.lower():
            return True
        if re.search(r'\d+ passed', test_section) and not re.search(r'\d+ (failed|error)', test_section):
            return True
        return False

    def generate_debug_script(self, problem: str, test_output: str) -> str:
        """Ask LLM to write a diagnostic script."""
        test_errors = _extract_test_errors(test_output)
        prompt = f"""A test failed:
{test_errors}

Bug: {problem}

Write a SHORT Python diagnostic script that prints intermediate values.
Runs in /testbed with the project installed. Only print statements, no assertions.
Output ONLY the script, no markdown."""
        script = llm(prompt, THINKING_MODEL, 2048)
        if script.startswith("```"):
            script = re.sub(r'^```(?:python)?\s*\n?', '', script)
            script = re.sub(r'\n?```\s*$', '', script)
        return script


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```[a-zA-Z]*\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    return text


# Frameworks that cannot be imported without being configured first. Left to
# write this preamble itself, an 8B model gets it wrong in a different way every
# attempt - observed on three consecutive tasks: an empty SECRET_KEY, an
# INSTALLED_APPS entry naming the benchmark's own test app ('admin_utils'), and a
# DJANGO_SETTINGS_MODULE pointing at a settings module that does not exist. Each
# burns a retry on boilerplate instead of on the issue, and when all three are
# spent the run proceeds with no reproduction signal at all. The preamble is read
# off the repository's own public configuration API, not off the benchmark's test
# setup, so the isolation in Appendix B.1 still holds.
BOOTSTRAP_HINTS = {
    "django": """
THE REPOSITORY IS DJANGO. Start the script with exactly this preamble and change
nothing in it except by adding settings you actually need:

import sys, django
from django.conf import settings
settings.configure(
    DEBUG=True,
    SECRET_KEY="x",
    DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3",
                           "NAME": ":memory:"}},
    INSTALLED_APPS=["django.contrib.contenttypes", "django.contrib.auth"],
    USE_TZ=True,
)
django.setup()

Do not set DJANGO_SETTINGS_MODULE, do not import a settings module, and do not
add an app to INSTALLED_APPS unless it ships inside django/ itself.
""",
}


def bootstrap_hint(repo_dir) -> str:
    """A known-good configuration preamble for the framework in `repo_dir`."""
    for marker, hint in BOOTSTRAP_HINTS.items():
        if (Path(repo_dir) / marker / "__init__.py").exists():
            return hint
    return ""


def build_reproduction(repo_dir, problem, architecture, feedback="") -> str:
    """Write a reproduction script from the issue text alone.

    This is the only test signal the loop is allowed to optimise against, so it
    has to be derived from the issue and the repository - never from the
    benchmark's test set. Contract: exit 1 while the bug is present, exit 0 once
    it is fixed, so the pass/fail signal is the process exit code rather than a
    model's reading of stdout.

    `feedback` carries what the previous attempt actually printed when run
    against the unmodified repository. Without it a retry is just the same
    prompt sampled twice and reproduces the same mistake - an 8B model will
    happily emit `sys.exit(0)` in a stub it never finished writing, or import a
    module that does not exist, and never learn otherwise.
    """
    retry = f"""
YOUR PREVIOUS ATTEMPT DID NOT WORK. Run against the UNMODIFIED, STILL-BUGGY
repository it should have exited 1; here is what actually happened:
{feedback[:2500]}

Fix the cause. If it crashed on an import, import something that exists. If it
exited 0, it did not actually detect the bug - assert the specific wrong
behaviour the issue describes and exit 1 when you see it. Do not leave
placeholder comments or an unfinished function: the script must run top to
bottom and call what it defines.
""" if feedback else ""
    prompt = f"""Write a reproduction script for this issue.

ISSUE: {problem}
REPOSITORY: {architecture[:1500]}
{bootstrap_hint(repo_dir)}{retry}
The script runs from the repository root with the project importable. It must:
1. exercise the exact behaviour the issue describes;
2. print what it observed, so a reader can see the symptom;
3. `sys.exit(1)` if the buggy behaviour is still present,
   `sys.exit(0)` if the behaviour is correct;
4. use only the standard library and this repository - no test framework, no
   network, no new dependencies;
5. if the repository needs configuration to import (for example Django needs
   `django.conf.settings.configure(...)` and `django.setup()`), do that first
   inside a try/except and print the failure rather than crashing silently;
6. be complete and self-executing - every function it defines must be called,
   and it must not depend on a service that is not running (no live database,
   no server). Call the function or class directly, and use
   `unittest.mock.patch` from the standard library if you need to observe what
   the code passes to `subprocess` or similar.

Write the script so it fails NOW, before any fix, and passes after the fix.
Output ONLY the Python source, no markdown, no explanation."""
    script = _strip_code_fence(llm(prompt, THINKING_MODEL, 3072))
    traj.log("reproduction", script=script, had_feedback=bool(feedback))
    return script


TEST_RAN_PATTERNS = (r"Ran \d+ test", r"\d+ passed", r"\d+ failed",
                     r"collected \d+ item", r"\d+ error", r"OK \(skipped")

# A command that dies while *collecting* tests is as useless as one that runs
# none: the failure describes the command, not the patch, so it reads the same in
# every round and the Critic can never attribute it. Observed twice - a label
# that does not exist (`urls.test_reverse` -> "module 'urls' has no attribute")
# and a backend whose driver is absent from the testbed ("Error loading MySQLdb
# module") - each of which pinned the Critic to FAIL for the whole trajectory.
TEST_UNUSABLE_PATTERNS = (
    r"has no attribute '[\w.]+'\s*$",
    r"ModuleNotFoundError: No module named",
    r"Failed to import test module",
    r"Error loading \w+ module",
    r"ImportError: cannot import name",
    r"error: unrecognized arguments",
    r"No such file or directory",
)


# A reproduction script "fails" for two very different reasons: the bug, or the
# script never getting as far as the bug. The second kind is worse than no
# script, because a non-zero exit pins the Critic to FAIL for the whole
# trajectory on evidence that has nothing to do with the patch. Observed:
# "Failed to setup Django: No module named 'settings'".
REPRO_SETUP_FAILURE_PATTERNS = (
    r"No module named",
    r"Failed to (setup|configure)",
    r"ImproperlyConfigured",
    r"cannot import name",
    r"SyntaxError",
    r"IndentationError",
    r"NameError: name '\w+' is not defined",
    r"command not found",
    # django.setup() not called, or called after a model import. Observed three
    # times in one run, all three before the script reached the issue behaviour.
    r"AppRegistryNotReady",
    r"Apps aren't loaded yet",
)


def repro_setup_failed(output: str) -> bool:
    """Whether the reproduction script died before exercising the issue."""
    return any(re.search(p, output or "") for p in REPRO_SETUP_FAILURE_PATTERNS)


def test_command_ran_anything(output: str) -> bool:
    """Whether a test command actually executed tests.

    A command that exits 0 without running anything - wrong label, no such
    directory, everything skipped - is worse than no signal at all: it reads as
    "no regressions" every single round, so the Critic can never fail a patch on
    it and can never pass one either. The same holds for a command that reaches
    the test runner but cannot import what it was asked to run.
    """
    if any(re.search(p, output, re.M) for p in TEST_UNUSABLE_PATTERNS):
        return False
    return any(re.search(p, output) for p in TEST_RAN_PATTERNS)


def choose_repo_test_cmd(repo_dir, problem, files, feedback="") -> str:
    """Pick a command from the repository's own test suite.

    Task-visible by construction: it must already exist in the repository. Used
    as a regression signal next to the reproduction script.
    """
    hints = []
    for probe in ("tests/runtests.py", "runtests.py", "setup.cfg", "tox.ini",
                  "pytest.ini", "pyproject.toml", "Makefile"):
        if (repo_dir / probe).exists():
            hints.append(probe)
    # The repository documents its own invocation, including the settings module
    # or flags a bare `runtests.py` will not work without. Cheaper and more
    # reliable than hoping the model remembers this project's convention.
    for doc in ("tests/README.rst", "tests/README", "CONTRIBUTING.rst",
                "CONTRIBUTING.md"):
        p = repo_dir / doc
        if p.exists():
            try:
                hints.append(f"--- {doc} (first 600 chars) ---\n"
                             + p.read_text(errors="replace")[:600])
            except OSError:
                pass
            break
    # Offer the labels that actually exist. Left to guess, the model produced
    # `urls.test_reverse` - plausible, absent from the repository, and it turned
    # the regression signal into the same collection error every round.
    existing = []
    tests_root = repo_dir / "tests"
    if tests_root.is_dir():
        # Tokens of the components being changed, so labels that share a word
        # with them are offered first. Everything else still follows: a test
        # directory is often named after the feature, not after the module.
        tokens = set()
        for f in files:
            tokens.update(t for t in re.split(r"[/_.]", f.lower())
                          if len(t) > 3 and t not in ("django", "core"))
        labels = [p.name for p in sorted(tests_root.iterdir())
                  if p.is_dir() and not p.name.startswith("_")]
        related = [l for l in labels
                   if any(t in l or l in t or l[:4] == t[:4] for t in tokens)]
        rest = [l for l in labels if l not in related]
        # All of them: a few thousand characters of directory names is cheap
        # next to picking a label that does not exist and losing the signal.
        existing = related + rest
    existing_section = (f"TEST LABELS THAT EXIST UNDER tests/ - use one of these "
                        f"verbatim:\n{existing}\n" if existing else "")
    retry = f"""
YOUR PREVIOUS SUGGESTION DID NOT RUN ANY TESTS. What happened:
{feedback[:1500]}

Pick a different command. Use the exact invocation this repository documents,
including any required settings module or flag, and a test label that exists.
""" if feedback else ""
    prompt = f"""Name ONE command from this repository's existing test suite that
exercises the code being changed.

ISSUE: {problem[:1200]}
FILES BEING CHANGED: {files}
TEST ENTRY POINTS PRESENT IN THE REPOSITORY: {hints}
{existing_section}{retry}
Rules:
- The command must already exist in this repository. Do not invent a test file.
- Target the narrowest existing test module that covers those files, so it runs
  in under a few minutes.
- It must run without any external service: no live database server, no network.
- Use the invocation the repository itself documents. Django's test runner, for
  example, needs a settings module: `python tests/runtests.py --settings=test_sqlite <label>`,
  where the label is a dotted path under `tests/` such as `dbshell` or
  `backends.base`. Most other projects use `python -m pytest <path> -x -q`.
- One single line, runnable from the repository root. No `&&`, no `cd`.

Return JSON: {{"command": "...", "why": "which existing tests this runs"}}"""
    r = llm_json(prompt, THINKING_MODEL, 512)
    cmd = (r.get("command") or "").strip().splitlines()[0] if r.get("command") else ""
    # A shell-injection-shaped or multi-command answer is dropped rather than
    # run: the regression signal is optional, correctness of the sandbox is not.
    if any(tok in cmd for tok in (";", "&&", "||", "|", "`", "$(", ">", "<")):
        traj.log("repo_test_cmd", command=cmd, accepted=False,
                 reason="rejected: contains shell metacharacters")
        return ""
    traj.log("repo_test_cmd", command=cmd, accepted=bool(cmd),
             why=r.get("why", ""))
    return cmd


# ============================================================
# Phases 1-4 (same as batch_solver.py - imported logic)
# ============================================================
def discover_architecture(repo_dir: Path, problem: str) -> str:
    tree = subprocess.run(
        ["find", ".", "-type", "f", "-name", "*.py", "-not", "-path", "./.git/*",
         "-not", "-path", "./__pycache__/*"],
        capture_output=True, text=True, cwd=repo_dir
    )
    prompt = f"""Analyze codebase architecture for this bug:

BUG: {problem[:2000]}

FILES: {tree.stdout.strip()[:5000]}

What subsystem, key files, patterns, and constraints?"""
    architecture = llm(prompt, role_model("planner"), max_tokens=2048)
    traj.log("architecture", text=architecture,
             n_py_files=len(tree.stdout.strip().split('\n')))
    return architecture


def is_test_file(rel: str) -> bool:
    """True for the repo's test files.

    A test file is a Dev-Primitive - the figure shows `test_module.py` as one, and
    it has genuinely useful things to say about what the fix must satisfy - but it
    must never be a *writer*. SWE-bench restores every test file to its base
    revision before grading, so an edit there is discarded work, and a submitted
    patch that touches tests reads as test tampering.
    """
    name = Path(rel).name
    return (rel.startswith("tests/") or "/tests/" in rel or "/test_" in rel
            or name.startswith("test_") or name.endswith("_test.py")
            or name == "tests.py" or name == "conftest.py")


def dev_primitive_files(repo_dir: Path) -> list:
    """Every file that gets wrapped as a Dev-Primitive.

    Not only `*.py`: the method figure shows `config.yaml` as a primitive beside
    the source files, and declared defaults are exactly the kind of thing a fix
    has to change in two places at once. Config files are few, so including them
    costs almost nothing in triage. `.json` is deliberately excluded - in these
    repos it is overwhelmingly test fixtures and locale data, thousands of files
    with no behaviour in them.
    """
    files = []
    for f in repo_dir.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(repo_dir).as_posix()
        if any(skip in rel for skip in ("__pycache__", ".git/", "node_modules/",
                                       ".venv/", "vendor/", "target/", "dist/")):
            continue
        if is_test_file(rel) and not GENERIC_COMPONENTS:
            continue
        if f.suffix == ".py" or (GENERIC_COMPONENTS
                                and f.suffix in SOURCE_SUFFIXES):
            files.append(rel)
        elif f.suffix in CONFIG_SUFFIXES and rel.count('/') <= CONFIG_MAX_DEPTH:
            files.append(rel)
        elif GENERIC_COMPONENTS and f.name in ("Dockerfile", "Makefile",
                                                "CMakeLists.txt", "package.json",
                                                "Cargo.toml", "go.mod"):
            files.append(rel)
    return files


def locate_files(repo_dir: Path, problem: str, architecture: str) -> list:
    all_files = dev_primitive_files(repo_dir)

    def triage_one(filename):
        filepath = repo_dir / filename
        try:
            content = filepath.read_text(errors="replace")
        except Exception:
            return None
        if len(content) < 50:
            return None
        lines = content.split('\n')
        if len(lines) > 3000:
            display = '\n'.join(f"{i+1:>5}|{lines[i]}" for i in range(1000))
            display += f"\n... [{len(lines)-2000} lines omitted] ...\n"
            display += '\n'.join(f"{i+1:>5}|{lines[i]}" for i in range(len(lines)-1000, len(lines)))
        else:
            display = '\n'.join(f"{i+1:>5}|{line}" for i, line in enumerate(lines))
        display = _clip_file(display)

        # `lines` is capped on purpose. Asked for an open-ended list, an 8B model
        # enumerates every line of the file, runs out of the 400-token budget
        # mid-array, and the answer fails to parse - which used to be recorded as
        # "not relevant" and silently dropped the file from scope. `relevant`
        # comes first so that even a truncated answer carries the decision.
        prompt = f"""You are the agent for: {filename}
BUG: {problem[:1000]}
ARCHITECTURE: {architecture[:500]}
FILE:\n{display}

Does this file need changes? Return JSON:
{{"relevant": true/false, "reason": "<one sentence>", "lines": [<at most 3 line numbers>]}}"""
        for attempt in range(MAX_RETRIES):
            try:
                r = llm_json(prompt, TRIAGE_MODEL, 400, no_think=True)
                # A malformed answer is not a "no". Accepting it as one silently
                # drops the file from scope, which is how the gold file was lost
                # on the first qwen3-8b run; retry, then try to salvage the
                # decision from the raw text before giving up, and record the raw
                # text if every attempt fails so the loss is visible.
                if "_error" in r:
                    if attempt < MAX_RETRIES - 1:
                        continue
                    raw = r.get("_raw", "")
                    salvaged = re.search(r'"relevant"\s*:\s*true', raw)
                    reason = re.search(r'"reason"\s*:\s*"([^"]{0,300})', raw)
                    traj.log("triage", file=filename,
                             relevant=bool(salvaged), lines=[],
                             reason=reason.group(1) if reason else "",
                             parse_failed=True, salvaged=bool(salvaged), raw=raw)
                    if salvaged:
                        return {"file": filename, "lines": [],
                                "reason": reason.group(1) if reason else ""}
                    return None
                traj.log("triage", file=filename, relevant=bool(r.get("relevant")),
                         lines=r.get("lines", []), reason=r.get("reason", ""))
                if r.get("relevant"):
                    return {"file": filename, "lines": r.get("lines", []), "reason": r.get("reason", "")}
                return None
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
        traj.log("triage", file=filename, relevant=False, lines=[], reason="",
                 call_failed=True)
        return None

    relevant = []
    with ThreadPoolExecutor(max_workers=TRIAGE_CONCURRENCY) as ex:
        futures = {ex.submit(triage_one, f): f for f in all_files}
        for fut in as_completed(futures):
            try:
                r = fut.result(timeout=90)
                if r:
                    relevant.append(r)
            except Exception:
                pass
    traj.log("locate_summary", n_files_scanned=len(all_files),
             n_relevant=len(relevant),
             relevant=[r["file"] for r in relevant])
    return relevant


# ============================================================
# Stage 1. PLANNING - the Planner's four sub-stages.
#
# The method figure decomposes "Issue Analysis & Planning" into four boxes:
# Bug Localization, Task Decomposition, Dependency Analysis, Edit Planning.
# Each is its own call with its own structured output so the trajectory shows
# which box produced which decision. Bug Localization is locate_files() above;
# the other three are here.
# ============================================================
def _file_context(repo_dir, relevant_files, ctx_lines=3, max_flagged=5) -> str:
    """The flagged line neighbourhoods of each located file, for planner prompts."""
    out = []
    for rf in relevant_files:
        filepath = repo_dir / rf["file"]
        try:
            lines = filepath.read_text(errors="replace").split('\n')
        except Exception:
            continue
        ctx = []
        for ln in (rf.get("lines") or [])[:max_flagged]:
            if not isinstance(ln, int):
                continue
            if 0 < ln <= len(lines):
                s, e = max(0, ln - ctx_lines), min(len(lines), ln + ctx_lines)
                ctx.append(f"  Lines {s+1}-{e}:\n" +
                           '\n'.join(f"    {i+1}|{lines[i]}" for i in range(s, e)))
        out.append(f"\n{rf['file']} ({rf.get('reason','')}):\n" + '\n'.join(ctx))
    return ''.join(out)


def decompose_tasks(repo_dir, problem, architecture, relevant_files) -> list:
    """Planner sub-stage 2 - Task Decomposition.

    Splits one issue into one sub-task per file and labels each file with the
    role it plays in the fix. The role is what the figure prints on each
    Dev-Primitive box, and it is also real information for the file agent: a
    Config primitive changes a declared default, a Tests primitive must not be
    "fixed" to match broken behaviour.
    """
    prompt = f"""Decompose this issue into one concrete sub-task per file.

BUG: {problem}
ARCHITECTURE: {architecture}
CANDIDATE FILES:{_file_context(repo_dir, relevant_files)}

For each candidate file, state the role it plays and the single sub-task it owns.
Roles: {', '.join(PRIMITIVE_ROLES)}.
If a file needs no change at all, set its task to exactly "none".

Return JSON:
{{"tasks": [{{"file": "path as given above",
              "role": "one of the roles listed",
              "task": "the one change this file is responsible for",
              "why": "why this file rather than another"}}]}}"""
    r = llm_json(prompt, role_model("planner"), 4096)
    known = {rf["file"]: rf for rf in relevant_files}
    tasks = [t for t in (r.get("tasks") or [])
             if isinstance(t, dict) and t.get("file") in known]
    # A parse failure must not empty the scope. Degrade to one task per located
    # file carrying its triage reason, i.e. to what the pre-decomposition
    # pipeline did, rather than losing every Dev-Primitive at once.
    if not tasks:
        tasks = [{"file": rf["file"], "role": "Other",
                  "task": rf.get("reason", ""), "why": "from bug localization"}
                 for rf in relevant_files]
    seen, deduped = set(), []
    for t in tasks:
        if t["file"] not in seen:
            seen.add(t["file"])
            deduped.append(t)
    tasks = deduped
    # A file the planner simply did not mention is not the same as a file it
    # decided needs no change, but either way it must not fall out of scope
    # silently: bug localization already said it is implicated, so keep it as an
    # observer and let stage 2. decide whether it has something to say.
    for rf in relevant_files:
        if rf["file"] not in seen:
            tasks.append({"file": rf["file"], "role": "Other", "task": "none",
                          "why": "located but not mentioned by task decomposition",
                          "unmentioned": True})
    for t in tasks:
        if t.get("role") not in PRIMITIVE_ROLES:
            t["role"] = "Other"
        # A file the planner gives no work to stays a Dev-Primitive: it still
        # takes part in stage 2., where it can tell the file that *does* change
        # what its own code needs. Dropping it from scope instead is what would
        # make the figure's collaboration stage vacant on a single-file fix - and
        # it is also how a file the planner underestimated gets silently lost.
        t["changes"] = str(t.get("task", "")).strip().lower() not in ("none", "")
        if is_test_file(t["file"]):
            t["role"] = "Tests"
            if not ALLOW_TEST_EDITS:
                t["changes"] = False
    if not any(t["changes"] for t in tasks):
        for t in tasks:
            t["changes"] = ALLOW_TEST_EDITS or not is_test_file(t["file"])
    traj.log("task_decomposition", tasks=tasks,
             observers=[t["file"] for t in tasks if not t["changes"]],
             parse_failed="_error" in r)
    return tasks


def analyze_dependencies(repo_dir, problem, architecture, tasks) -> dict:
    """Planner sub-stage 3 - Dependency Analysis.

    Names the couplings between the sub-tasks: which file's change forces which
    other file's change, and what breaks if only one side of the pair moves.
    These edges then seed stage 2. so the agents message each other about real
    dependencies instead of guessing who cares.
    """
    task_list = '\n'.join(
        f"  {t['file']} [{t.get('role')}]: {t.get('task')}" for t in tasks)
    prompt = f"""These sub-tasks belong to one bug fix and must land together.

BUG: {problem}
ARCHITECTURE: {architecture}
SUB-TASKS:
{task_list}

Identify the dependencies between them: a call, an import, a subclass, a
registration, a default value duplicated in two places, a test that asserts the
old behaviour. For each, say what breaks if only one side changes. Then give a
safe order to apply them, and name any file that must also change but is missing
from the list above.

Return JSON:
{{"edges": [{{"from": "path", "to": "path",
              "relation": "calls|imports|subclasses|registers|shares-default|tests",
              "risk": "what breaks if only one side changes"}}],
  "order": ["path", ...],
  "missing_files": ["repo-relative path"]}}"""
    r = llm_json(prompt, role_model("planner"), 2048)
    known = {t["file"] for t in tasks}
    edges = [e for e in (r.get("edges") or [])
             if isinstance(e, dict) and e.get("from") in known and e.get("to") in known
             and e["from"] != e["to"]]
    order = [f for f in (r.get("order") or []) if f in known]
    missing = [m for m in (r.get("missing_files") or [])
               if isinstance(m, str) and m not in known and (repo_dir / m).exists()]
    traj.log("dependency_analysis", edges=edges, order=order, missing_files=missing,
             parse_failed="_error" in r)
    return {"edges": edges, "order": order, "missing_files": missing}


def plan_edits(repo_dir, problem, architecture, tasks, deps,
               critic=None, round_idx=0) -> dict:
    """Planner sub-stage 4 - Edit Planning.

    Turns the sub-tasks plus their dependencies into a concrete per-file edit
    spec: what to target, what the change is, what must keep working. On a
    replan the Critic's report is the primary input - that is the figure's
    4. -> 1. arrow - so this same function serves both the first plan and every
    revision, and the trajectory shows them as the same planner box.
    """
    task_list = '\n'.join(
        f"  {t['file']} [{t.get('role')}]: {t.get('task')}"
        + ("" if t.get("changes", True) else "   (no change expected)")
        for t in tasks)
    dep_list = '\n'.join(
        f"  {e['from']} -> {e['to']} ({e.get('relation')}): {e.get('risk','')}"
        for e in deps.get("edges", [])) or "  (none identified)"
    critic_section = ""
    if critic:
        critic_section = f"""
THE PREVIOUS ATTEMPT FAILED. THE CRITIC REPORTS:
{render_critic_report(critic)}

Your previous plan was wrong or incomplete in a way this report pins down.
Address the suspected root cause directly. Do not restate the previous plan.
"""
    prompt = f"""Write the concrete edit plan for this bug fix.

BUG: {problem}
ARCHITECTURE: {architecture}
SUB-TASKS:
{task_list}
DEPENDENCIES:
{dep_list}
APPLY ORDER: {deps.get('order') or 'unconstrained'}
FILE CONTEXT:{_file_context(repo_dir, [{'file': t['file'], 'lines': [], 'reason': t.get('task','')} for t in tasks])}
{critic_section}
CONSTRAINTS:
- ONLY these files may change: {[t['file'] for t in tasks if t.get('changes', True)]}.
  Do not touch any other file even if it contains a similar pattern.
- Make the MINIMUM change that fixes the bug. Do not refactor unrelated code.
- Cover every edge case the bug report describes, not just the headline one.
- Plan for what the TEST asserts, not for what makes the code look tidy.

Return JSON:
{{"summary": "the fix in two or three sentences",
  "edits": [{{"file": "path",
              "target": "function / class / line region to change",
              "change": "exactly what the new behaviour is, precisely enough to write the code from",
              "must_not_break": "behaviour that has to keep working"}}]}}"""
    r = llm_json(prompt, role_model("planner"), 4096)
    known = {t["file"]: t for t in tasks}
    edits = {}
    for e in (r.get("edits") or []):
        # Test files are never given an edit spec, however the planner phrases it.
        if (isinstance(e, dict) and e.get("file") in known
                and (ALLOW_TEST_EDITS or not is_test_file(e["file"]))):
            edits[e["file"]] = {
                "role": known[e["file"]].get("role", "Other"),
                "target": e.get("target", ""),
                "change": e.get("change", ""),
                "must_not_break": e.get("must_not_break", ""),
            }
    summary = r.get("summary") or ""
    # Same reasoning as in decompose_tasks: an unparseable answer must not leave
    # a file with no instructions, so fall back to its sub-task text. Only for
    # files the decomposition actually assigned work to - an observer with no
    # edit spec is a decision, not an omission.
    for f, t in known.items():
        if t.get("changes", True):
            edits.setdefault(f, {"role": t.get("role", "Other"), "target": "",
                                 "change": t.get("task", ""), "must_not_break": ""})
    if not summary:
        summary = ("(planner returned no summary; per-file sub-tasks are "
                   "the plan)" if "_error" in r else "")
    plan_obj = {"summary": summary, "edits": edits}
    traj.log("plan", round=round_idx, text=render_plan(plan_obj),
             summary=summary, edits=edits, files=list(edits),
             from_critic=bool(critic), parse_failed="_error" in r)
    return plan_obj


def render_plan(plan_obj) -> str:
    """The edit plan as prose, for the prompts that take a plan as free text."""
    parts = [plan_obj.get("summary", "")]
    for f, e in (plan_obj.get("edits") or {}).items():
        parts.append(f"\n{f} [{e.get('role','Other')}]"
                     f"\n  target: {e.get('target','')}"
                     f"\n  change: {e.get('change','')}"
                     f"\n  must not break: {e.get('must_not_break','')}")
    return '\n'.join(parts).strip()


# ============================================================
# Stage 4. CRITIC - Analyze & Feedback.
#
# A component of its own, not a branch of the planner. It reads only evidence:
# o = (o_shell, o_test, o_runtime, o_trace) from EXECUTE(R'), plus the diff that
# was actually produced. It emits v in {PASS, FAIL} and, on FAIL, the structured
# feedback phi = (e, c, u) - observed failure evidence, suspected cause and the
# components involved, revision guidance. It proposes no patch; the planner
# replans from phi.
#
# Two constraints from the method are enforced here rather than trusted to the
# prompt: no held-out evaluation test ever reaches this function, and the Critic
# cannot override a deterministic execution failure - it can withhold a PASS but
# never manufacture one.
# ============================================================
def evidence_delta(obs, prev_report) -> str:
    """What changed in o between the previous round and this one.

    The Critic already receives its previous feedback, but on these tasks the raw
    evidence is often bit-for-bit identical across rounds - the reproduction script
    was discarded, and the repository test exits 0 both before and after the edit -
    so the model re-derives the same diagnosis and `revision_guidance` comes back
    verbatim for three rounds. Stating the comparison explicitly is what makes the
    repetition visible to it: if nothing moved, the previous guidance did not work
    and repeating it cannot be the answer.
    """
    if not prev_report:
        return ""
    same = []
    moved = []
    for label, now, then in (
            ("reproduction exit code", obs.get("repro_rc"),
             prev_report.get("repro_rc")),
            ("repository test exit code", obs.get("repo_test_rc"),
             prev_report.get("repo_test_rc")),
            ("repository test output", (obs.get("test") or "").strip(),
             (prev_report.get("test_output") or "").strip()),
            ("reproduction output", (obs.get("repro_output") or "").strip(),
             (prev_report.get("repro_output") or "").strip())):
        (same if now == then else moved).append(
            label if now == then else f"{label}: {then!r} -> {now!r}"[:220])
    lines = [f"EVIDENCE COMPARED WITH ROUND {prev_report.get('round')}:"]
    lines += [f"  unchanged: {s}" for s in same]
    lines += [f"  changed:   {m}" for m in moved]
    if not moved:
        lines.append("  Nothing in the execution evidence moved. The revision "
                     "guidance you gave last round was applied and produced the "
                     "same result, so it was wrong or incomplete. Do not restate "
                     "it: either name a different component or give a different "
                     "mechanism, and say which of your previous claims the "
                     "unchanged evidence rules out.")
    return '\n'.join(lines) + '\n'


def resolve_component(repo_dir: Path, name: str, known: dict) -> str:
    """Map a component name the Critic produced onto a real repository path.

    The Critic writes from the evidence, not from a directory listing, so it names
    `db/models/fields/json.py` when the path is `django/db/models/fields/json.py`,
    or just `resolvers.py`. Resolution by path suffix and then by basename keeps
    those, while an invented file that matches nothing is reported as unresolved
    instead of vanishing.
    """
    name = (name or "").strip().lstrip("./")
    if not name or name.endswith("/"):
        return ""
    if (repo_dir / name).is_file():
        return name
    pool = list(known)
    hits = [p for p in pool if p == name or p.endswith("/" + name)]
    if len(hits) == 1:
        return hits[0]
    base = Path(name).name
    hits = [p for p in pool if Path(p).name == base]
    if len(hits) == 1:
        return hits[0]
    # Fall back to the repository itself: the pool only holds what localization
    # surfaced, and the Critic may legitimately implicate a file it skipped.
    hits = [p.relative_to(repo_dir).as_posix() for p in repo_dir.rglob(base)
            if p.is_file() and "__pycache__" not in str(p)]
    return hits[0] if len(hits) == 1 else ""


def critique(problem, plan_obj, obs, our_diff, files_in_scope, round_idx,
             prev_feedback="", baseline_repo_test_rc=None,
             prev_report=None, inactive_candidates=None) -> dict:
    """CRITIC(q, Pi, R', o) -> v in {PASS, FAIL}, plus structured feedback phi.

    Reads only task-visible execution evidence: the reproduction script's exit
    code and output, the repository's own tests, runtime errors and traces. No
    benchmark test name reaches this function.

    The Critic cannot override deterministic execution failure - a non-zero
    reproduction exit code forces FAIL regardless of what the model says. It can
    only ever *withhold* a PASS, never manufacture one.
    """
    repro_rc = obs.get("repro_rc")
    setup_rc = obs.get("setup_rc")
    container_rc = obs.get("container_rc")
    # `None` means the script did not run at all, which is an absence of evidence
    # rather than evidence of failure; only a non-zero exit is deterministic.
    repro_failed = repro_rc is not None and repro_rc != 0
    repro_label = ("did not run" if repro_rc is None else
                   "failed" if repro_failed else "passed")
    execution_failed = (setup_rc not in (None, 0)
                        or container_rc not in (None, 0))
    plan_lines = '\n'.join(f"  {f} [{e.get('role')}]: {e.get('change','')}"
                           for f, e in (plan_obj.get("edits") or {}).items())
    # Eq. 9 lets the Planner activate additional components, but only the Critic
    # can implicate them, and it cannot implicate a file it does not know exists.
    inactive = [f for f in (inactive_candidates or {}) if f not in files_in_scope]
    inactive_str = ("COMPONENTS LOCALIZED FOR THIS ISSUE BUT NOT ACTIVATED - these "
                    "paths exist; name one under `missing_components` if the "
                    "evidence implicates it:\n"
                    + '\n'.join(f"  {f}: {inactive_candidates[f]}"
                                for f in inactive[:20])) if inactive else ""

    prompt = f"""You are the Critic. Evaluate the current repository state against
the original issue using execution evidence. You do not modify any file.

ORIGINAL ISSUE: {problem}

CURRENT PLAN - activated components and their local objectives:
{plan_lines or '  (none)'}

CURRENT CHANGED ARTIFACTS (contents or deletion markers):
{our_diff[:4000] or '(empty - nothing was changed on disk)'}

SHELL OUTPUT (setup exit {setup_rc}; container exit {container_rc}):
{(obs.get('shell') or '(none)')[:1200]}

REPRODUCTION SCRIPT - exit code {repro_rc} ({repro_label}):
{(obs.get('repro_output') or '(did not run)')[:2500]}

TASK-VISIBLE CHECK RUN `{obs.get('repo_test_cmd') or '(none chosen)'}` - exit code {obs.get('repo_test_rc')}:
{(obs.get('test') or '(did not run)')[:2500]}

RUNTIME EVIDENCE:
{(obs.get('runtime') or '(none)')[:1500]}

EXECUTION TRACES:
{(obs.get('trace') or '(none)')[:1500]}
{f'PREVIOUS CRITIC FEEDBACK:{chr(10)}{prev_feedback[:1500]}' if prev_feedback else ''}
{evidence_delta(obs, prev_report)}COMPONENTS CURRENTLY ACTIVATED: {files_in_scope}
{inactive_str}

Check: was the requested behaviour implemented; do the repository's own tests
still pass; were regressions introduced; do execution failures remain; are the
cross-component modifications mutually consistent; does the change address the
issue rather than only the symptom.

If it succeeded, return exactly:
{{"status": "PASS",
  "evidence": ["<execution or test evidence supporting success>"],
  "summary": "<why the repository now satisfies the issue>"}}

If it failed, return exactly:
{{"status": "FAIL",
  "failure_evidence": [{{"source": "<exactly one word: test OR shell OR runtime OR trace>",
                         "evidence": "<the observed failure, quoted>"}}],
  "suspected_causes": [{{"component": "<repository file>",
                         "reason": "<evidence-supported diagnosis>"}}],
  "missing_components": ["<file that should be activated but was not>"],
  "revision_guidance": [{{"component": "<file>", "action": "<specific revision>"}}],
  "summary": "<concise diagnosis>"}}

Only name a file under `missing_components` if the evidence implicates it and it
is not already activated, and give a path that exists in this repository - a file
that does not exist cannot be activated. Preserve the parts of the current change
that the evidence shows are working."""
    r = llm_json(prompt, role_model("critic"), 2560)

    claimed = str(r.get("status", "")).upper()
    # Preserve the regression flag for diagnosis, but any observed failing
    # task-visible test must block PASS, including one that failed at baseline.
    test_rc = obs.get("repo_test_rc")
    test_failed = test_rc is not None and test_rc != 0
    regressed = (test_rc is not None and test_rc != 0
                 and test_rc != baseline_repo_test_rc)
    # Deterministic execution failure is not overridable by model judgement.
    status = "PASS" if claimed == "PASS" else "FAIL"
    if repro_failed or test_failed or execution_failed:
        status = "FAIL"
    report = {
        "round": round_idx,
        "status": status,
        "repro_rc": repro_rc,
        "setup_rc": setup_rc,
        "container_rc": container_rc,
        "repo_test_rc": test_rc,
        "baseline_repo_test_rc": baseline_repo_test_rc,
        "regressed": regressed,
        "repo_test_cmd": obs.get("repo_test_cmd"),
        "evidence": [e for e in (r.get("evidence") or []) if isinstance(e, str)],
        "failure_evidence": [e for e in (r.get("failure_evidence") or [])
                             if isinstance(e, dict)],
        "suspected_causes": [c for c in (r.get("suspected_causes") or [])
                             if isinstance(c, dict)],
        "missing_components": [m for m in (r.get("missing_components") or [])
                               if isinstance(m, str)],
        "revision_guidance": [g for g in (r.get("revision_guidance") or [])
                              if isinstance(g, dict)],
        "summary": r.get("summary", ""),
        "our_diff": our_diff,
        "repro_output": obs.get("repro_output", ""),
        "test_output": obs.get("test", ""),
    }
    # If the model claimed PASS but execution says otherwise, say so in the record
    # rather than letting the override disappear.
    if claimed == "PASS" and status == "FAIL":
        report["overridden"] = ("critic returned PASS but execution says otherwise "
                                f"(reproduction rc={repro_rc}, repository tests "
                                f"rc={test_rc}, setup rc={setup_rc}, "
                                f"container rc={container_rc}, "
                                f"baseline rc={baseline_repo_test_rc})")
    traj.log("critic", **report, parse_failed="_error" in r)
    return report


def no_critic_report(obs, our_diff, round_idx, baseline_repo_test_rc) -> dict:
    """The `w/o Critic Feedback` stand-in for phi.

    Carries the raw execution signals forward - the appendix says environment
    execution stays enabled and its outputs remain available for planning - but
    contains no failure evidence, no suspected causes and no revision guidance,
    because nothing analysed them. Status is always FAIL: with no Critic there is
    no verdict that can accept a repository state, so the loop runs out the budget
    B rather than stopping early. That is what distinguishes this setting from
    B=0, which stops after the first round.
    """
    return {"round": round_idx, "status": "FAIL", "critic": "disabled",
            "repro_rc": obs.get("repro_rc"),
            "repo_test_rc": obs.get("repo_test_rc"),
            "baseline_repo_test_rc": baseline_repo_test_rc,
            "regressed": None, "repo_test_cmd": obs.get("repo_test_cmd"),
            "evidence": [], "failure_evidence": [], "suspected_causes": [],
            "missing_components": [], "revision_guidance": [],
            "summary": "", "our_diff": our_diff,
            "repro_output": obs.get("repro_output", ""),
            "test_output": obs.get("test", "")}


def render_critic_report(report) -> str:
    """The Critic's structured feedback phi = (e, c, u) as text for the Planner."""
    if report["status"] == "PASS":
        return ("CRITIC: PASS\n"
                + '\n'.join(f"  evidence: {e}" for e in report["evidence"])
                + f"\n  {report.get('summary','')}")
    lines = [f"CRITIC: FAIL (reproduction exit={report['repro_rc']}, "
             f"repository tests exit={report['repo_test_rc']})"]
    lines.append("FAILURE EVIDENCE:")
    for e in report["failure_evidence"][:6]:
        lines.append(f"  [{e.get('source','?')}] {e.get('evidence','')}")
    if not report["failure_evidence"]:
        lines.append("  (none quoted by the critic)")
    lines.append("SUSPECTED CAUSES:")
    for c in report["suspected_causes"][:6]:
        lines.append(f"  {c.get('component','?')}: {c.get('reason','')}")
    if report["missing_components"]:
        lines.append(f"COMPONENTS NOT YET ACTIVATED: "
                     f"{', '.join(report['missing_components'])}")
    lines.append("REVISION GUIDANCE:")
    for g in report["revision_guidance"][:8]:
        lines.append(f"  {g.get('component','?')}: {g.get('action','')}")
    lines.append(f"SUMMARY: {report.get('summary','')}")
    return '\n'.join(lines)


def revise_active_set(repo_dir, tasks, report, round_idx,
                      localized_pool=None) -> list:
    """Let re-planning change the active set A, not just the objectives x_i.

    Pi' = PLANNER(q, R', Pi, phi) keeps the same representation as Pi, so it may
    activate components the new execution evidence implicates, and drop ones that
    are no longer relevant. Without this, A is frozen at the first plan and a
    mislocalised bug can never be recovered from.

    Activation comes from the Critic's `missing_components`; removal is limited to
    components bug localization named but task decomposition never gave work to
    and the Critic has not mentioned either - the only components for which there
    is positive evidence of irrelevance.
    """
    by_file = {t["file"]: t for t in tasks}
    mentioned = {c.get("component") for c in report.get("suspected_causes", [])}
    mentioned |= {g.get("component") for g in report.get("revision_guidance", [])}

    activated, unresolved = [], []
    for raw in report.get("missing_components", []):
        f = resolve_component(repo_dir, raw, localized_pool or by_file)
        if not f:
            unresolved.append(raw)
            continue
        if f in by_file or f in activated:
            continue
        tasks.append({
            "file": f,
            "role": "Tests" if is_test_file(f) else "Other",
            "task": ("none" if is_test_file(f) and not ALLOW_TEST_EDITS else
                     "implicated by execution evidence; see the critic's guidance"),
            "changes": ALLOW_TEST_EDITS or not is_test_file(f),
            "why": f"activated by critic feedback in round {round_idx}",
            "activated_round": round_idx,
        })
        activated.append(f)

    removed = []
    for t in list(tasks):
        if (t.get("unmentioned") and not t.get("changes")
                and t["file"] not in mentioned):
            tasks.remove(t)
            removed.append(t["file"])

    # Never deactivate everything: an empty A has nothing to modify and the round
    # would be a no-op.
    if not tasks:
        tasks = [by_file[f] for f in removed]
        removed = []
    traj.log("active_set", round=round_idx, activated=activated, removed=removed,
             unresolved=unresolved, active=[t["file"] for t in tasks])
    return tasks


def negotiate(repo_dir, problem, architecture, plan, relevant_files,
              test_feedback="", round_idx=1, deps=None, roles=None) -> dict:
    """Stage 2. COLLABORATION - natural-language inter-file communication.

    Each activated file agent states the change it intends to make and sends
    messages to the other files whose behaviour its edit would break (signature
    changes, new callers, registrations, changed defaults). Every addressee then
    revises its own intent in light of what it received. Runs again on each
    replan round, so the agents re-negotiate under critic feedback.

    The planner's dependency edges are handed to each agent as a starting point,
    so a message about a coupling the planner already found is cheap, and the
    agent's own reading of its file is what adds edges the planner missed.

    Returns ({filename: intent_text}, {filenames that changed their intent}).
    """
    files = [rf["file"] for rf in relevant_files if (repo_dir / rf["file"]).exists()]
    if len(files) < 2:
        traj.log("negotiate", round=round_idx, stage="skipped",
                 reason=f"only {len(files)} file(s) activated; nobody to talk to")
        return {}, set()

    feedback_section = ""
    if test_feedback:
        feedback_section = f"\nTHE PREVIOUS ROUND FAILED:\n{test_feedback}\n"

    roles = roles or {}
    dep_lines = [f"  {e['from']} -> {e['to']} ({e.get('relation')}): {e.get('risk','')}"
                 for e in (deps or {}).get("edges", [])]

    def propose(fname):
        numbered = _clip_file(numbered_content(repo_dir / fname)[:NEGOTIATE_FILE_CHARS])
        others = [f for f in files if f != fname]
        mine = [d for d in dep_lines if d.strip().startswith(fname)]
        dep_section = ""
        if mine:
            dep_section = ("\nDEPENDENCIES THE PLANNER ALREADY FOUND FOR YOUR FILE:\n"
                           + '\n'.join(mine) + "\n")
        prompt = f"""You are the agent that owns this file, and only this file: {fname}
YOUR ROLE IN THIS FIX: {roles.get(fname, 'Other')}

BUG: {problem}
ARCHITECTURE: {architecture}
COORDINATED PLAN: {plan}
{dep_section}{feedback_section}
THE OTHER FILES ACTIVATED FOR THIS FIX, WITH THEIR ROLES:
{chr(10).join(f"  {f} [{roles.get(f, 'Other')}]" for f in others)}

YOUR FILE (with line numbers):
{numbered}

State the change YOU will make to YOUR file. Then message any of the other
activated files whose code must change as a consequence of your edit - a changed
signature they call, a new argument they must pass, a default they duplicate, a
registration they own. Say what they must do and why.

Send a message ONLY if your edit genuinely forces a change in that file. Send
nothing if your edit is self-contained.

Return JSON:
{{"intent": "the precise change I will make to my own file",
  "messages": [{{"target_file": "<one of {others}>", "message": "what you must change and why"}}]}}"""
        return fname, llm_json(prompt, role_model("primitive"), 2048)

    # Round 1 of communication: everyone speaks at once.
    intents, outgoing = {}, []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(propose, f) for f in files]
        for fut in as_completed(futures):
            try:
                fname, res = fut.result()
            except Exception as e:
                traj.log("negotiate", round=round_idx, stage="propose_failed",
                         error=f"{type(e).__name__}: {e}")
                continue
            intents[fname] = res.get("intent", "")
            msgs = [m for m in (res.get("messages") or [])
                    if isinstance(m, dict) and m.get("target_file") in files
                    and m.get("target_file") != fname and m.get("message")]
            for m in msgs:
                outgoing.append({"from": fname, "to": m["target_file"],
                                 "message": m["message"]})
            traj.log("negotiate", round=round_idx, stage="propose", file=fname,
                     role=roles.get(fname, "Other"), intent=intents[fname],
                     messages=[{"to": m["target_file"], "message": m["message"]}
                               for m in msgs])

    inbox = {}
    for m in outgoing:
        inbox.setdefault(m["to"], []).append(m)
    traj.log("negotiate", round=round_idx, stage="deliver",
             n_messages=len(outgoing), recipients=sorted(inbox),
             messages=outgoing)

    # Round 2: every addressee reconsiders its own intent.
    def revise(fname, msgs):
        incoming = '\n'.join(f"From {m['from']}: {m['message']}" for m in msgs)
        prompt = f"""You are the agent that owns this file, and only this file: {fname}

BUG: {problem}
YOUR CURRENT INTENT: {intents.get(fname, '(none stated)')}

MESSAGES YOU RECEIVED FROM THE OTHER FILE AGENTS:
{incoming}

Do these messages change what you must do to your own file? Adjust only if a
message genuinely requires it; otherwise keep your intent as it is.

Return JSON:
{{"adjusted": true/false,
  "reasoning": "why you did or did not change your intent",
  "intent": "your final intent for this file"}}"""
        return fname, llm_json(prompt, role_model("primitive"), 2048)

    adjusted_files = set()
    if inbox:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(revise, f, m) for f, m in inbox.items()]
            for fut in as_completed(futures):
                try:
                    fname, res = fut.result()
                except Exception as e:
                    traj.log("negotiate", round=round_idx, stage="revise_failed",
                             error=f"{type(e).__name__}: {e}")
                    continue
                adjusted = bool(res.get("adjusted"))
                new_intent = res.get("intent") or intents.get(fname, "")
                traj.log("negotiate", round=round_idx, stage="revise", file=fname,
                         adjusted=adjusted, reasoning=res.get("reasoning", ""),
                         intent_before=intents.get(fname, ""), intent_after=new_intent,
                         received=[m["from"] for m in inbox[fname]])
                if adjusted:
                    intents[fname] = new_intent
                    adjusted_files.add(fname)

    traj.log("negotiate", round=round_idx, stage="settled", intents=intents,
             roles={f: roles.get(f, "Other") for f in intents},
             adjusted=sorted(adjusted_files))
    return intents, adjusted_files


def edit_file(repo_dir, filename, problem, architecture, plan, test_feedback="",
              round_idx=1, intent="", role="Other", edit_spec=None) -> bool:
    """Let one primitive apply a structured edit to its owned artifact."""
    from hermes.artifacts import edit_component

    objective = (edit_spec or {}).get("change", "")
    changed = edit_component(repo_dir, filename, problem, objective,
                             intent, test_feedback, round_idx,
                             architecture=architecture, plan=plan, role=role,
                             must_not_break=(edit_spec or {}).get("must_not_break", ""))
    # Intent-to-add makes newly created source files part of the submitted diff
    # without staging their content or scooping up build/test byproducts.
    persistent = [name for name in changed if (repo_dir / name).is_file()]
    if persistent:
        subprocess.run(["git", "add", "-N", "--", *persistent],
                       cwd=repo_dir, check=True)
    traj.log("edit", round=round_idx, file=filename, role=role,
             applied=bool(changed), changed=changed,
             error=None if changed else "no valid artifact edit applied")
    return bool(changed)


# ============================================================
# Solve Single Instance (WITH docker iteration)
# ============================================================
def solve_instance(instance_id: str, instance: dict,
                   hf_data: dict, test_specs: dict) -> Optional[dict]:
    """Solve with full docker iteration loop."""
    start_time = time.time()
    problem = instance["problem_statement"]
    repo_dir = WORK_DIR / instance_id

    # Get base_commit
    base_commit = hf_data.get(instance_id, {}).get("base_commit")

    recorder = traj.Trajectory(
        instance_id + TRAJ_SUFFIX, repo=instance["repo"], base_commit=base_commit,
        problem_statement=problem, out_root=TRAJ_ROOT,
        config={"backend": BACKEND,
                "triage_model": TRIAGE_MODEL, "thinking_model": THINKING_MODEL,
                "planner_model": role_model("planner"),
                "primitive_model": role_model("primitive"),
                "critic_model": role_model("critic"),
                "file_context_chars": FILE_CONTEXT_CHARS,
                "max_iterations": MAX_ITERATIONS,
                "max_relevant_files": MAX_RELEVANT_FILES,
                "triage_concurrency": TRIAGE_CONCURRENCY,
                "solver": MODEL_NAME,
                "configuration": ablation_label(),
                "ablations": {"communication": not ABLATE_COMMUNICATION,
                              "on_demand_activation": not ABLATE_ON_DEMAND,
                              "execution_feedback": not ABLATE_EXECUTION_FEEDBACK,
                              "critic_feedback": not ABLATE_CRITIC}})
    traj.set_trajectory(recorder)
    try:
        return _solve_instance_inner(
            instance_id, instance, hf_data, test_specs,
            start_time, problem, repo_dir, base_commit, recorder)
    except BaseException as e:
        recorder.log("note", text=f"run aborted: {type(e).__name__}: {e}")
        recorder.close(resolved=False, rounds=0,
                       elapsed_s=time.time() - start_time, final_patch="")
        raise
    finally:
        traj.set_trajectory(None)
        print(f"  [{instance_id}] Trajectory: {recorder.out_dir}")


def _solve_instance_inner(instance_id, instance, hf_data, test_specs,
                          start_time, problem, repo_dir, base_commit,
                          recorder) -> Optional[dict]:

    # Setup repo (shared clone)
    if not repo_dir.exists():
        repo_slug = instance["repo"].replace("/", "__")
        base_repo = WORK_DIR / f"_base_{repo_slug}"
        try:
            if base_repo.exists():
                subprocess.run(["git", "clone", "--shared", str(base_repo), str(repo_dir)],
                    capture_output=True, timeout=60, check=True)
            else:
                subprocess.run(["git", "clone", f"https://github.com/{instance['repo']}.git", str(base_repo)],
                    capture_output=True, timeout=600, check=True)
                subprocess.run(["git", "clone", "--shared", str(base_repo), str(repo_dir)],
                    capture_output=True, timeout=60, check=True)
        except Exception as e:
            print(f"  [{instance_id}] Clone failed: {e}")
            traj.log("note", text=f"clone failed: {e}")
            recorder.close(resolved=False, rounds=0,
                           elapsed_s=time.time() - start_time, final_patch="")
            return None

    if base_commit:
        subprocess.run(["git", "checkout", "."], capture_output=True, cwd=repo_dir)
        co = subprocess.run(["git", "checkout", base_commit],
                            capture_output=True, text=True, cwd=repo_dir)
        # A checkout that leaves no working tree used to pass silently: bug
        # localization then scanned zero components, reported zero relevant
        # files, and the run ended in 30s looking like a model failure. An
        # interrupted clone leaves exactly this state (a bare `.git` and
        # nothing else), so check the working tree instead of the exit code.
        if not any(p.name != ".git" for p in repo_dir.iterdir()):
            msg = (f"repository checkout is empty at {repo_dir} "
                   f"(git checkout said: {co.stderr.strip()[:200]}); "
                   f"remove the directory and rerun so it is cloned again")
            traj.log("note", text=f"setup failed: {msg}")
            print(f"  [{instance_id}] {msg}")
            recorder.close(resolved=False, rounds=0,
                           elapsed_s=time.time() - start_time, final_patch="")
            return None

    # Setup runner
    spec = test_specs.get(instance_id)
    runner = ContainerRunner(instance_id, spec) if spec else None

    # Check if env image is available
    has_docker = False
    if runner:
        result = subprocess.run([settings.CONTAINER_CLI, "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True)
        if runner.env_image in result.stdout:
            has_docker = True
            print(f"  [{instance_id}] Docker: yes ({runner.env_image})")
        else:
            print(f"  [{instance_id}] Docker: no (need {runner.env_image})")

    traj.log("note", text=f"container feedback available: {has_docker} "
                          f"(env image {runner.env_image if runner else 'n/a'})")
    if not has_docker:
        msg = "required SWE-bench environment image is unavailable"
        traj.log("note", text=msg)
        recorder.close(resolved=False, rounds=0,
                       elapsed_s=time.time() - start_time, final_patch="")
        print(f"  [{instance_id}] {msg}")
        return None

    # ------------------------------------------------------------------
    # Stage 1. PLANNING - Issue Analysis & Planning.
    # Four planner sub-stages, one call each, each logged separately.
    # ------------------------------------------------------------------
    traj.set_phase("1.0_ISSUE_ANALYSIS")
    architecture = discover_architecture(repo_dir, problem)

    traj.set_phase(TRIAGE_PHASE)
    relevant_files = locate_files(repo_dir, problem, architecture)
    if not relevant_files:
        recorder.close(resolved=False, rounds=0,
                       elapsed_s=time.time() - start_time, final_patch="")
        return None

    # Pool for Eq. 9 re-activation: what bug localization surfaced, whether or not
    # the first plan activated it. Kept because the Critic otherwise has no idea
    # which files exist and returns invented paths under `missing_components`
    # (`tests/settings.py`, `test_settings.py` were the observed ones), which are
    # dropped, so the active set could only ever shrink.
    localized_pool = {r["file"]: (r.get("reason") or "")[:160]
                      for r in relevant_files}

    if ABLATE_ON_DEMAND:
        # w/o On-Demand Activation: the same localization runs, but every
        # component it identified is activated up front and the set is then frozen
        # for the whole trajectory (see the guard in the re-planning branch).
        # Ranking still orders the pool, it just no longer gates activation.
        n_before = len(relevant_files)
        if ABLATE_ON_DEMAND_CAP:
            relevant_files = relevant_files[:ABLATE_ON_DEMAND_CAP]
        traj.log("note", text=f"ablation w/o on-demand activation: activating "
                              f"{len(relevant_files)} of {n_before} identified "
                              f"components up front; active set frozen "
                              f"(cap={ABLATE_ON_DEMAND_CAP or 'none'})")
    elif len(relevant_files) > MAX_RELEVANT_FILES:
        # Own phase so the call is kept in the trajectory: triage calls are
        # dropped by phase, and this used to run under the triage phase and
        # vanish with them.
        traj.set_phase("1.1_BUG_LOCALIZATION_RANK")
        before = [r["file"] for r in relevant_files]
        # The whole candidate list does not fit: with ~500 candidates and their
        # reasons the prompt exceeds the context window and the server answers
        # 400, aborting the run. Shorten each entry and hard-cap how many are
        # offered, preferring candidates whose path the issue text itself names
        # and deprioritising config files, which are almost never the defect.
        named = set()
        low = problem.lower()
        for r in relevant_files:
            stem = Path(r["file"]).stem.lower()
            if len(stem) > 3 and (r["file"].lower() in low or stem in low):
                named.add(r["file"])
        ordered = sorted(
            relevant_files,
            key=lambda r: (r["file"] not in named,
                           r["file"].endswith(CONFIG_SUFFIXES)))
        candidates = ordered[:RANK_CANDIDATES]
        file_list_str = '\n'.join(f"  {r['file']}: {r['reason'][:160]}"
                                  for r in candidates)
        # A weak model spends its whole budget thinking here and returns nothing,
        # which silently degrades to "first MAX_RELEVANT_FILES in completion
        # order". Skip thinking. Ask for the top N only: ordering all candidates
        # runs past the token limit and truncates into invalid JSON.
        rank = llm_json(
            f"Rank these candidate files by how likely each one must be edited "
            f"to fix the bug. Put the file that contains the actual defect first.\n"
            f"BUG: {problem[:500]}\nFILES:\n{file_list_str}\n"
            f"Return JSON: {{\"ranked\": [\"path\", ...]}} with only the "
            f"{MAX_RELEVANT_FILES} most important paths, most important first.",
            TRIAGE_MODEL, 1024, no_think=True)
        ranked = rank.get("ranked", [])
        # Handle case where LLM returns [{"file": "x"}] instead of ["x"]
        if ranked and isinstance(ranked[0], dict):
            ranked = [r.get("file", r.get("name", "")) for r in ranked]
        ranked = [r for r in ranked if isinstance(r, str)]
        fm = {r["file"]: r for r in relevant_files}
        kept = [fm[f] for f in ranked[:MAX_RELEVANT_FILES] if f in fm]
        relevant_files = kept or candidates[:MAX_RELEVANT_FILES]
        traj.log("rank", before=before, after=[r["file"] for r in relevant_files],
                 rank_failed=not kept, n_ranked=len(ranked),
                 n_candidates=len(candidates), named_in_issue=sorted(named))
        localized_pool = {r["file"]: (r.get("reason") or "")[:160]
                          for r in candidates}

    traj.set_phase("1.2_TASK_DECOMPOSITION")
    tasks = decompose_tasks(repo_dir, problem, architecture, relevant_files)

    traj.set_phase("1.3_DEPENDENCY_ANALYSIS")
    deps = analyze_dependencies(repo_dir, problem, architecture, tasks)
    for mf in deps["missing_files"]:
        tasks.append({"file": mf,
                      "role": "Tests" if is_test_file(mf) else "Other",
                      "task": ("none" if is_test_file(mf) and not ALLOW_TEST_EDITS else
                               "change required by a dependency of another sub-task"),
                      "changes": ALLOW_TEST_EDITS or not is_test_file(mf),
                      "why": "added by dependency analysis"})

    # The scope handed to the file agents. `lines` comes from bug localization,
    # `reason` from task decomposition, `role` is the figure's primitive label.
    located_lines = {rf["file"]: rf.get("lines", []) for rf in relevant_files}
    scope = [{"file": t["file"], "lines": located_lines.get(t["file"], []),
              "reason": t.get("task", ""), "role": t.get("role", "Other")}
             for t in tasks]
    roles = {t["file"]: t.get("role", "Other") for t in tasks}

    traj.set_phase("1.4_EDIT_PLANNING")
    plan_obj = plan_edits(repo_dir, problem, architecture, tasks, deps)
    plan = render_plan(plan_obj)

    # ------------------------------------------------------------------
    # The task-visible test signal. Built before anything is edited, from the
    # issue text and the repository only, and confirmed to FAIL on the
    # unmodified repository - a reproduction that already passes proves nothing
    # and would let round 1 "succeed" without changing behaviour. The
    # benchmark's own FAIL_TO_PASS / PASS_TO_PASS tests are not run until after
    # the loop terminates.
    # ------------------------------------------------------------------
    repro, repo_test_cmd, baseline_test_rc = "", "", None
    if has_docker and ABLATE_EXECUTION_FEEDBACK:
        # w/o Execution Feedback: the signal is never read, so it is never built.
        # Constructing a reproduction script and then discarding its output would
        # charge this configuration for LLM calls the ablation does not use and
        # would make its cost column incomparable with the complete system's.
        traj.log("note", text="ablation w/o execution feedback: no reproduction "
                              "script and no repository test command are built")
    elif has_docker:
        traj.set_phase("1.5_REPRODUCTION")
        changed = [t["file"] for t in tasks if t.get("changes")]
        repro_fb, cmd_fb, reproduces, cmd_ok = "", "", False, False
        for attempt in range(1, 4):
            if not cmd_ok:
                repo_test_cmd = choose_repo_test_cmd(repo_dir, problem, changed,
                                                     feedback=cmd_fb)
            if not reproduces:
                repro = build_reproduction(repo_dir, problem, architecture,
                                           feedback=repro_fb)
            base = runner.run_task_visible(repo_dir, repro, repo_test_cmd)

            setup_broke = repro_setup_failed(base.get("repro_output", ""))
            reproduces = (base.get("repro_rc") not in (None, 0)
                          and not setup_broke)
            cmd_ok = bool(repo_test_cmd) and test_command_ran_anything(
                base.get("test", ""))
            if cmd_ok:
                baseline_test_rc = base.get("repo_test_rc")
            traj.log("baseline", attempt=attempt,
                     repro_rc=base.get("repro_rc"),
                     repo_test_cmd=repo_test_cmd,
                     repo_test_rc=base.get("repo_test_rc"),
                     reproduces=reproduces, repo_test_ran=cmd_ok,
                     repro_setup_failed=setup_broke,
                     repro_output=base.get("repro_output", ""),
                     test_output=base.get("test", ""),
                     shell=base.get("shell", ""))
            print(f"  [{instance_id}] baseline {attempt}: "
                  f"reproduces={reproduces} repo_tests_ran={cmd_ok}")
            if reproduces and cmd_ok:
                break
            # A script that exits 0 against the buggy code does not exercise the
            # reported behaviour, and a test command that runs nothing reads as
            # "no regressions" forever. Retry each with what actually happened.
            repro_fb = (f"exit code {base.get('repro_rc')}\n"
                        + ("The script never reached the reported behaviour: it "
                           "failed while setting up. Fix the setup - configure "
                           "settings inline, import what you use, and do not "
                           "depend on a settings module that does not exist.\n"
                           if setup_broke else "")
                        + base.get("repro_output", "")[:2000]
                        + "\n" + base.get("shell", "")[:800])
            cmd_fb = (f"`{repo_test_cmd}` exited {base.get('repo_test_rc')}\n"
                      + base.get("test", "")[:1500])
        if not reproduces:
            traj.log("note", text="no reproduction script ever failed on the "
                                  "unmodified repository; discarded")
            repro = ""
        if not cmd_ok:
            traj.log("note", text="no repository test command actually ran "
                                  "tests; discarded")
            repo_test_cmd = ""
        if not repro and not repo_test_cmd:
            traj.log("note", text="no task-visible test signal could be "
                                  "established; the Critic will judge from the "
                                  "diff and shell evidence alone")

    max_iter = MAX_ITERATIONS if has_docker else 1
    test_feedback = ""
    resolved = False
    accepted = False
    report = None

    for iteration in range(max_iter):
        rnd = iteration + 1
        traj.log("round", round=rnd, status="begin",
                 files_in_scope=[rf["file"] for rf in scope], roles=roles)
        # --------------------------------------------------------------
        # Stage 2. COLLABORATION - natural-language inter-file communication
        # --------------------------------------------------------------
        traj.set_phase(f"2_INTER_FILE_COMMUNICATION (round {rnd})")
        if ABLATE_COMMUNICATION:
            # w/o Inter-Primitive Communication: each primitive keeps its local
            # reasoning and editing but receives nothing from its peers, so C_i is
            # empty and there is no intent beyond the objective the Planner
            # assigned. No message means no promotion either, so the writers are
            # exactly the components the Planner gave an edit spec to.
            intents, adjusted = {}, set()
            traj.log("note", round=rnd,
                     text="ablation w/o inter-primitive communication: C_i empty, "
                          "no messages exchanged, no observer promotion")
        else:
            intents, adjusted = negotiate(repo_dir, problem, architecture, plan,
                                          scope, test_feedback, round_idx=rnd,
                                          deps=deps, roles=roles)

        # Each Dev-Primitive rewrites its own source. Who writes: whoever the
        # planner gave an edit spec, plus any observer that negotiation talked
        # into changing its intent - the one place stage 2. can widen the patch.
        traj.set_phase(f"2.5_SELF_MODIFY (round {rnd})")
        edits = plan_obj.get("edits") or {}
        # Negotiation may promote an observer to a writer, but not a config file:
        # a behaviour bug is essentially never fixed in tox.ini or setup.cfg, and
        # one peer message claiming otherwise is how a config file that bug
        # localization only weakly flagged ends up in the submitted patch. A
        # config file the *planner* assigned an edit to still writes.
        writers = [rf["file"] for rf in scope
                   if (rf["file"] in edits
                       or (rf["file"] in adjusted
                           and not rf["file"].endswith(CONFIG_SUFFIXES)))
                   and (ALLOW_TEST_EDITS or not is_test_file(rf["file"]))]
        promoted = [f for f in writers if f not in edits]
        traj.log("self_modify_scope", round=rnd, writers=writers,
                 observers=[rf["file"] for rf in scope if rf["file"] not in writers],
                 promoted_by_negotiation=promoted)
        edited = 0
        failed_writers = []
        for fname in writers:
            if not (repo_dir / fname).exists():
                continue
            if edit_file(repo_dir, fname, problem, architecture, plan,
                         test_feedback, round_idx=rnd,
                         intent=intents.get(fname, ""),
                         role=roles.get(fname, "Other"),
                         edit_spec=edits.get(fname)):
                edited += 1
            else:
                failed_writers.append(fname)

        if edited == 0:
            # Carry the concrete apply errors into the next round. A generic
            # "no edits applied" told the primitives nothing, so they produced
            # the same unmatched anchor three rounds in a row and the whole
            # re-planning budget was spent without a single execution.
            errs = [e for e in traj.records_of("edit")
                    if e.get("round") == rnd and e.get("error")]
            detail = "\n".join(f"  {e['file']}: {e['error']}" for e in errs[-6:])
            test_feedback = (
                "No edit could be applied to the repository this round. The "
                "failures were:\n" + (detail or "  (no error recorded)") +
                "\nQuote a shorter anchor that certainly appears in the file, or "
                "target a different function.")
            traj.log("round", round=rnd, status="end", edited=0, resolved=False,
                     failed_writers=failed_writers)
            continue

        if not has_docker:
            traj.log("round", round=rnd, status="end", edited=edited, resolved=None)
            traj.log("note", text="no container image: stopping after one edit round, "
                                  "no test feedback was available")
            break

        # --------------------------------------------------------------
        # Stage 3. EXECUTION - o = EXECUTE(R') = (shell, test, runtime, trace).
        # Task-visible only: the reproduction script this run wrote and a
        # command from the repository's own suite. The Shell/Terminal box below
        # adds a diagnostic script when there is a failure to explain.
        # --------------------------------------------------------------
        traj.set_phase(f"3_EXECUTION (round {rnd})")
        if ABLATE_EXECUTION_FEEDBACK:
            # w/o Execution Feedback: the repository is still modified and still
            # graded at the end, but no observation produced after the edits is
            # handed to the Critic or to re-planning, so the Critic must judge the
            # result by static inspection of the diff and the plan. The execution
            # is skipped rather than run and discarded: running it would spend the
            # environment budget on evidence this configuration may not read, and
            # the deterministic-failure override below has nothing to act on
            # either way.
            obs = {"shell": "", "test": "", "runtime": "", "trace": "",
                   "repro_output": "", "repro_rc": None,
                   "repo_test_cmd": None, "repo_test_rc": None}
            traj.log("note", round=rnd,
                     text="ablation w/o execution feedback: o withheld from the "
                          "Critic and from re-planning; no post-edit execution")
        else:
            obs = runner.run_task_visible(repo_dir, repro, repo_test_cmd)
        traj.log("runtime", round=rnd, env_image=runner.env_image,
                 repro_rc=obs.get("repro_rc"),
                 repo_test_cmd=obs.get("repo_test_cmd"),
                 repo_test_rc=obs.get("repo_test_rc"),
                 shell=obs.get("shell", ""), runtime=obs.get("runtime", ""),
                 trace=obs.get("trace", ""))
        traj.log("verify", round=rnd,
                 test_cmd=obs.get("repo_test_cmd") or "(reproduction script only)",
                 repro_rc=obs.get("repro_rc"),
                 repo_test_rc=obs.get("repo_test_rc"),
                 baseline_repo_test_rc=baseline_test_rc,
                 repro_output=obs.get("repro_output", ""),
                 output=obs.get("test", ""),
                 output_chars=len(obs.get("output", "")))
        print(f"  [{instance_id}] Round {rnd}: repro_rc={obs.get('repro_rc')} "
              f"repo_tests_rc={obs.get('repo_test_rc')} "
              f"(baseline {baseline_test_rc})")

        # --------------------------------------------------------------
        # Stage 5. REFINED PATCH - what this round's self-modification produced
        # --------------------------------------------------------------
        our_diff = subprocess.run(["git", "diff", "--no-color"],
                                  capture_output=True, text=True, cwd=repo_dir).stdout

        # Stage 3., Shell/Terminal box: run code in the environment to print the
        # intermediate values the test output does not show. Always available to
        # the Critic, not only on the rounds that are followed by a replan.
        traj.set_phase(f"3_EXECUTION_SHELL (round {rnd})")
        if ABLATE_EXECUTION_FEEDBACK:
            # Same reason as above: the diagnostic script is an execution
            # observation, so this configuration does not get one either.
            debug_section = ""
        else:
            debug_script = runner.generate_debug_script(
                problem, (obs.get("repro_output", "") + "\n"
                          + obs.get("test", ""))[:4000])
            debug_result = runner.run_task_visible(repo_dir, debug_script, "")
            debug_section = debug_result.get("repro_output", "")
            traj.log("debug_script", round=rnd, script=debug_script,
                     output=debug_section)
            obs["shell"] = (obs.get("shell", "") + "\n--- diagnostic script ---\n"
                            + debug_section)[:6000]

        # --------------------------------------------------------------
        # Stage 4. CRITIC - v = CRITIC(q, Pi, R', o), and on FAIL the feedback phi.
        # --------------------------------------------------------------
        traj.set_phase(f"4_CRITIC (round {rnd})")
        prev_report = report
        if ABLATE_CRITIC:
            # w/o Critic Feedback: execution still runs and its raw output still
            # reaches the next round, but nothing analyses it into phi = (e, c, u).
            # There is therefore no verdict to accept on, so the trajectory runs
            # the full budget B and the Planner re-plans from the raw evidence.
            # The record keeps the deterministic signals so the run stays
            # comparable, but carries no diagnosis, no suspected causes and no
            # guidance - which is exactly what the Planner is left without.
            report = no_critic_report(obs, our_diff, rnd, baseline_test_rc)
            print(f"  [{instance_id}] Critic disabled (ablation); "
                  f"repro_rc={obs.get('repro_rc')} "
                  f"repo_test_rc={obs.get('repo_test_rc')}")
        else:
            report = critique(problem, plan_obj, obs, our_diff,
                              [rf["file"] for rf in scope], rnd,
                              prev_feedback=test_feedback,
                              baseline_repo_test_rc=baseline_test_rc,
                              prev_report=prev_report,
                              inactive_candidates=localized_pool)
            print(f"  [{instance_id}] Critic: {report['status']}"
                  + (f" ({report['overridden']})" if report.get("overridden") else ""))

        traj.set_phase(f"5_REFINED_PATCH (round {rnd})")
        traj.log("patch", round=rnd, diff=our_diff, chars=len(our_diff),
                 files_changed=edited, passed=report["status"] == "PASS")

        if report["status"] == "PASS":
            # The Critic accepts the repository state as the refined patch. Note
            # this is not the benchmark verdict - that is computed once, after
            # the loop, and may disagree.
            accepted = True
            traj.log("round", round=rnd, status="end", edited=edited,
                     critic="PASS")
            print(f"  [{instance_id}] Critic accepted in round {rnd}")
            break

        traj.log("round", round=rnd, status="end", edited=edited, critic="FAIL")

        if iteration < max_iter - 1:
            # With no Critic there is no phi to render, so the Planner sees the raw
            # evidence and nothing else - that is the whole content of the
            # ablation. With no execution feedback the evidence sections are empty
            # and only the diff survives.
            test_feedback = (("EXECUTION EVIDENCE (no critic analysis available)"
                              if ABLATE_CRITIC else render_critic_report(report))
                             + f"\n\nYOUR DIFF:\n{our_diff}"
                             + f"\n\nREPRODUCTION OUTPUT:\n"
                             + obs.get("repro_output", "")[:2000]
                             + f"\n\nREPOSITORY TEST OUTPUT:\n"
                             + obs.get("test", "")[:2000]
                             + f"\n\nDIAGNOSTIC:\n{debug_section[:2000]}")

            # Pi' = PLANNER(q, R', Pi, phi). The active set moves first - the Critic
            # may have named a component nobody activated - then the objectives
            # are rewritten for whoever is active now.
            traj.set_phase(f"1.3_ACTIVE_SET_REVISION (round {rnd})")
            if ABLATE_ON_DEMAND:
                # The set was activated up front and stays that way for the whole
                # trajectory: no activation, no removal.
                traj.log("active_set", round=rnd, activated=[], removed=[],
                         unresolved=[], frozen=True,
                         active=[t["file"] for t in tasks])
            elif ABLATE_CRITIC:
                # Removal needs positive evidence of irrelevance, and the only
                # thing that produces it is the Critic. Without one, the Planner
                # may still rewrite objectives but the set itself does not move.
                traj.log("active_set", round=rnd, activated=[], removed=[],
                         unresolved=[], no_critic=True,
                         active=[t["file"] for t in tasks])
            else:
                tasks = revise_active_set(repo_dir, tasks, report, rnd,
                                          localized_pool=localized_pool)
            scope = [{"file": t["file"], "lines": located_lines.get(t["file"], []),
                      "reason": t.get("task", ""), "role": t.get("role", "Other")}
                     for t in tasks]
            roles = {t["file"]: t.get("role", "Other") for t in tasks}

            traj.set_phase(f"1.4_EDIT_PLANNING (replan after round {rnd})")
            # `critic=None` under the ablation: there is no phi, and rendering the
            # stand-in record would hand the Planner an empty CRITIC block that
            # reads as "the Critic found nothing" rather than "there is no Critic".
            # The raw evidence still reaches it through `test_feedback`.
            plan_obj = plan_edits(repo_dir, problem, architecture, tasks, deps,
                                  critic=None if ABLATE_CRITIC else report,
                                  round_idx=rnd)
            plan = render_plan(plan_obj)
            traj.log("replan", round=rnd, text=plan,
                     active=[t["file"] for t in tasks])

    # Get final patch
    diff = subprocess.run(["git", "diff"], capture_output=True, text=True, cwd=repo_dir)
    patch = diff.stdout

    # ------------------------------------------------------------------
    # Evaluation, after the trajectory has terminated. This is the first and
    # only point at which the benchmark's held-out FAIL_TO_PASS / PASS_TO_PASS
    # tests are executed, and the result feeds nothing back into the run.
    # ------------------------------------------------------------------
    if has_docker and patch.strip():
        traj.set_phase("6_HELD_OUT_EVALUATION")
        final = runner.run_tests(repo_dir)
        resolved = final["passed"]
        traj.log("held_out_evaluation", passed=resolved,
                 error=final.get("error"),
                 test_status=final.get("test_status", {}),
                 output=final.get("output", "")[-8000:],
                 note="run after termination; not visible to any component")

    elapsed = time.time() - start_time
    status = "RESOLVED" if resolved else f"patch={len(patch)}b"
    print(f"  [{instance_id}] Done {elapsed:.0f}s, iter={iteration+1}, "
          f"critic={'PASS' if accepted else 'FAIL'}, {status}")

    recorder.close(resolved=resolved, rounds=iteration + 1,
                   elapsed_s=elapsed, final_patch=patch)

    if not patch.strip():
        return None

    return {
        "instance_id": instance_id,
        "model_patch": patch,
        "model_name_or_path": MODEL_NAME,
    }


# ============================================================
# Batch Runner
# ============================================================
def load_completed() -> set:
    done = set()
    if OUTPUT_PATH.exists():
        for line in OUTPUT_PATH.read_text().strip().split('\n'):
            if line.strip():
                try:
                    done.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
    return done


def save_prediction(pred: dict):
    with open(OUTPUT_PATH, "a") as f:
        f.write(json.dumps(pred) + "\n")


def run_batch(instance_ids=None, resume=True):
    with open(DATASET_PATH) as f:
        all_instances = json.load(f)

    # Load HF data
    print("Loading HuggingFace data...")
    hf_data = {}
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    for item in ds:
        hf_data[item["instance_id"]] = {"base_commit": item.get("base_commit")}

    # Load test specs
    print("Loading test specs...")
    from swebench.harness.docker_build import make_test_spec
    test_specs = {}
    for item in ds:
        test_specs[item["instance_id"]] = make_test_spec(item)
    print(f"  {len(test_specs)} specs loaded")

    # Filter instances
    if instance_ids:
        instances = [i for i in all_instances if i["instance_id"] in set(instance_ids)]
    else:
        instances = all_instances

    if resume:
        done = load_completed()
        instances = [i for i in instances if i["instance_id"] not in done]
        if done:
            print(f"Resuming: {len(done)} done, {len(instances)} remaining")

    print(f"\nRunning {len(instances)} instances")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)

    success, fail = 0, 0
    start_time = time.time()

    for i, inst in enumerate(instances):
        iid = inst["instance_id"]
        print(f"\n[{i+1}/{len(instances)}] {iid}")
        try:
            pred = solve_instance(iid, inst, hf_data, test_specs)
            if pred:
                save_prediction(pred)
                success += 1
            else:
                save_prediction({"instance_id": iid, "model_patch": "", "model_name_or_path": MODEL_NAME})
                fail += 1
        except Exception as e:
            print(f"  [{iid}] ERROR: {e}")
            traceback.print_exc()
            save_prediction({"instance_id": iid, "model_patch": "", "model_name_or_path": MODEL_NAME})
            fail += 1

        elapsed = time.time() - start_time
        rate = (success + fail) / elapsed * 3600 if elapsed > 0 else 0
        print(f"  Progress: {success} ok / {fail} fail | {rate:.0f}/hr")

    print(f"\n{'='*70}")
    print(f"DONE: {success}/{success+fail} in {(time.time()-start_time)/3600:.1f}h")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", nargs="*")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--ollama", metavar="MODEL", nargs="?", const="qwen3:8b",
                        help="run the pipeline against a local Ollama model "
                             "instead of Bedrock (default: qwen3:8b)")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama chat URL (default localhost:11434/api/chat)")
    parser.add_argument("--ollama-num-ctx", type=int, default=32768,
                        help="Ollama context length (default 32768)")
    parser.add_argument("--vllm", metavar="MODEL", nargs="?", const="Qwen/Qwen3-8B",
                        help="run against an OpenAI-compatible vLLM server "
                             "(default model: Qwen/Qwen3-8B)")
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM chat-completions URL "
                             "(default http://localhost:8000/v1/chat/completions)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override triage concurrency")
    # Backbone selection for the hosted backend (the backbone-scale table). The
    # two slots stay separate because localization is one cheap call per file and
    # everything else is a reasoning call; a sweep that moves only --model keeps
    # localization fixed and isolates the effect of the reasoning backbone.
    parser.add_argument("--model", default=None, metavar="LITELLM_ID",
                        help="reasoning backbone: OpenAI, Claude, or LiteLLM id "
                             f"(default {THINKING_MODEL})")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=("none", "minimal", "low", "medium",
                                 "high", "xhigh", "max"),
                        help="effort for direct OpenAI or Claude API calls")
    parser.add_argument("--triage-model", default=None, metavar="LITELLM_ID",
                        help="bug-localization backbone "
                             f"(default {TRIAGE_MODEL})")
    # Per-role slots for the backbone-scaling table. Unset slots follow --model,
    # so `--critic-model X` alone is the "scale the Critic only" row.
    parser.add_argument("--planner-model", default=None, metavar="LITELLM_ID",
                        help="Planner backbone (default: the reasoning backbone)")
    parser.add_argument("--primitive-model", default=None, metavar="LITELLM_ID",
                        help="Dev-Primitive backbone (default: the reasoning "
                             "backbone)")
    parser.add_argument("--critic-model", default=None, metavar="LITELLM_ID",
                        help="Critic backbone (default: the reasoning backbone)")
    parser.add_argument("--traj-suffix", default="",
                        help="suffix for the trajectory directory, so runs with "
                             "different models do not overwrite each other")
    # Component ablations (Appendix: Ablation Settings). Each flag removes one
    # mechanism and leaves the rest of the loop intact, so a row of the
    # component-ablation table corresponds to exactly one flag.
    parser.add_argument("--ablate-communication", action="store_true",
                        help="w/o Inter-Primitive Communication: primitives get "
                             "their Planner objective only, C_i is always empty")
    parser.add_argument("--ablate-on-demand", action="store_true",
                        help="w/o On-Demand Activation: activate every component "
                             "localization considered relevant instead of the "
                             "ranked active set")
    parser.add_argument("--ablate-execution-feedback", action="store_true",
                        help="w/o Execution Feedback: never run the repository "
                             "tests or the reproduction script; the Critic sees "
                             "the diff only")
    parser.add_argument("--ablate-critic", action="store_true",
                        help="w/o Critic Feedback: raw execution evidence is fed "
                             "back with no structured phi and no PASS verdict")
    parser.add_argument("--ablate-on-demand-cap", type=int,
                        default=ABLATE_ON_DEMAND_CAP,
                        help="hard cap on components activated under "
                             "--ablate-on-demand, to keep the run finite "
                             "(0 = no cap)")
    args = parser.parse_args()

    ABLATE_COMMUNICATION = args.ablate_communication
    ABLATE_ON_DEMAND = args.ablate_on_demand
    ABLATE_EXECUTION_FEEDBACK = args.ablate_execution_feedback
    ABLATE_CRITIC = args.ablate_critic
    ABLATE_ON_DEMAND_CAP = args.ablate_on_demand_cap
    if ABLATE_EXECUTION_FEEDBACK and ABLATE_CRITIC:
        parser.error("--ablate-execution-feedback and --ablate-critic together "
                     "remove the whole feedback loop; run --max-iterations 1 "
                     "instead so the configuration is unambiguous")
    print(f"configuration={ablation_label()}")

    if args.no_resume and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    MAX_ITERATIONS = args.max_iterations
    if args.vllm:
        use_vllm(args.vllm, url=args.vllm_url)
    elif args.ollama:
        use_ollama(args.ollama, num_ctx=args.ollama_num_ctx,
                   url=args.ollama_url)
    if args.ollama_url:
        OLLAMA_URL = args.ollama_url
    if args.vllm_url:
        VLLM_URL = args.vllm_url
    if args.concurrency:
        TRIAGE_CONCURRENCY = args.concurrency
    # After use_vllm/use_ollama, so an explicit id wins over the backend default
    # and a sweep can point one slot at a hosted model and the other at a served
    # one without editing the source.
    if args.model:
        THINKING_MODEL = args.model
        MODEL_NAME = f"hermes-{args.model.split('/')[-1]}"
    HOSTED_REASONING_EFFORT = args.reasoning_effort
    if args.triage_model:
        TRIAGE_MODEL = args.triage_model
    PLANNER_MODEL = args.planner_model
    PRIMITIVE_MODEL = args.primitive_model
    CRITIC_MODEL = args.critic_model
    if args.vllm or args.ollama or args.model or args.triage_model:
        print(f"backend={BACKEND} model={THINKING_MODEL} triage={TRIAGE_MODEL} "
              f"file_ctx={FILE_CONTEXT_CHARS} concurrency={TRIAGE_CONCURRENCY}")
    if PLANNER_MODEL or PRIMITIVE_MODEL or CRITIC_MODEL:
        print(f"roles: planner={role_model('planner')} "
              f"primitive={role_model('primitive')} critic={role_model('critic')}")
    if args.traj_suffix:
        TRAJ_SUFFIX = args.traj_suffix

    run_batch(instance_ids=args.instance, resume=not args.no_resume)
