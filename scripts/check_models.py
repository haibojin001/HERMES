#!/usr/bin/env python3
"""Make one small JSON call to each configured model before starting a run."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hermes import pipeline as p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hosted", "ollama", "vllm"),
                        default=os.environ.get("HERMES_BACKEND", "hosted"))
    parser.add_argument("--model", action="append", required=True,
                        help="repeat for every Planner, primitive, and Critic model")
    parser.add_argument("--url", help="URL for the selected Ollama or vLLM backend")
    parser.add_argument("--ollama-url", help="Ollama /api/chat URL for a mixed-model run")
    parser.add_argument("--vllm-url", help="vLLM /v1/chat/completions URL for a mixed-model run")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=("none", "minimal", "low", "medium",
                                 "high", "xhigh", "max"))
    args = parser.parse_args()
    if args.backend == "ollama":
        p.use_ollama(args.model[0], url=args.url)
    elif args.backend == "vllm":
        p.use_vllm(args.model[0], url=args.url)
    else:
        p.use_hosted(args.model[0], reasoning_effort=args.reasoning_effort)
    if args.ollama_url:
        p.OLLAMA_URL = args.ollama_url
    if args.vllm_url:
        p.VLLM_URL = args.vllm_url
    if args.backend == "hosted" and args.url:
        if any(model.startswith("ollama/") for model in args.model):
            p.OLLAMA_URL = args.url
        elif any(model.startswith("vllm/") for model in args.model):
            p.VLLM_URL = args.url
    if args.backend != "hosted":
        p.HOSTED_REASONING_EFFORT = args.reasoning_effort
    for model in args.model:
        response = p.llm_json(
            'Connectivity check. Return {"ready": true}.',
            model, max_tokens=512,
            no_think=(args.backend in ("ollama", "vllm")
                      or model.startswith(("ollama/", "vllm/"))))
        if response.get("ready") is not True:
            raise RuntimeError(f"{model}: unexpected response: {response}")
        print(f"{model}: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
