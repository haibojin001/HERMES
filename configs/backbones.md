# Backbone slots and serving

`role_model()` selects the Planner, Dev-Primitive and Critic model. An unset role uses the default reasoning model. Localization uses the triage slot; it is accounted for in the trajectory summary. The SWE-bench solver accepts `--model`, `--planner-model`, `--primitive-model`, `--critic-model`, `--triage-model`, and `--reasoning-effort`. Official Harbor and legacy `tb` wrappers read the corresponding `HERMES_MODEL`, `HERMES_PLANNER_MODEL`, `HERMES_PRIMITIVE_MODEL`, `HERMES_CRITIC_MODEL`, `HERMES_TRIAGE_MODEL`, and `HERMES_REASONING_EFFORT` variables.

The paper's local Qwen3-8B configuration is Ollama with a 32K context, temperature 0.6, top-p 0.95 and top-k 20. `use_ollama()` applies those settings:

```bash
ollama serve
python -m hermes.pipeline --ollama qwen3:8b --ollama-num-ctx 32768 \
  --instance django__django-13512 --max-iterations 4
```

The official benchmark wrappers read `HERMES_BACKEND`, `HERMES_MODEL` and optionally `HERMES_OLLAMA_URL`. The cluster bundle shows how to serve the model on a GPU node while a machine with the benchmark containers runs the agent and grader.

For hosted models, install `pip install -e '.[hosted]'`. Prefix OpenAI model IDs with `openai/` to use the Responses API and `OPENAI_API_KEY`; prefix Claude IDs with `anthropic/` to use the Messages API and `ANTHROPIC_API_KEY`. Bare IDs beginning with `gpt-` or `claude-` also use the corresponding native API. Other IDs, including `bedrock/...`, use LiteLLM and its provider credentials. Set effort only when the chosen model supports that value. The manuscript's display names are not necessarily API IDs; use the exact ID enabled for your account.

| Manuscript model family | Interface | Model ID form |
|---|---|---|
| GPT | OpenAI Responses | `openai/YOUR_API_MODEL_ID` |
| Claude | Anthropic Messages | `anthropic/YOUR_API_MODEL_ID` |
| DeepSeek | LiteLLM DeepSeek provider | `deepseek/YOUR_API_MODEL_ID` |
| Gemini rows | LiteLLM Gemini provider | `gemini/YOUR_API_MODEL_ID` |
| Qwen3-8B | Ollama or vLLM | `ollama/qwen3:8b` or `vllm/YOUR_SERVED_MODEL_ID` |

Role slots can select different providers in one run. `ollama/MODEL` routes to the local Ollama endpoint and `vllm/MODEL` routes to the local vLLM endpoint even when the default backbone is hosted. Use `--ollama-url` or `--vllm-url` with the SWE-bench solver if those endpoints are remote. The official Harbor and legacy `tb` wrappers take `HERMES_OLLAMA_URL` and `HERMES_VLLM_URL` for the respective endpoints. For the paper's heterogeneous Qwen primitive condition, run Ollama as the default and point the activation and diagnosis roles at their hosted IDs:

```bash
export HERMES_BACKEND=ollama HERMES_MODEL=qwen3:8b
export HERMES_PLANNER_MODEL='openai/YOUR_API_MODEL_ID'
export HERMES_CRITIC_MODEL='openai/YOUR_API_MODEL_ID'
python scripts/check_models.py --backend ollama \
  --model qwen3:8b --model "$HERMES_PLANNER_MODEL" \
  --model "$HERMES_CRITIC_MODEL"
experiments/terminal_bench_4/run_official.sh all
```

Set `HERMES_TRIAGE_MODEL` as well when the SWE-bench localization calls need their own backbone. Use `HERMES_PRIMITIVE_MODEL` to override the primitive backbone independently. `HERMES_REASONING_EFFORT` applies to hosted calls; the local Qwen route keeps its Ollama generation settings.

Check every model before scheduling benchmark jobs:

```bash
python scripts/check_models.py --backend hosted \
  --model 'openai/YOUR_API_MODEL_ID' --reasoning-effort medium
python scripts/check_models.py --backend ollama \
  --model qwen3:8b --url http://127.0.0.1:11434/api/chat
```

`--vllm` selects an OpenAI-compatible server and currently uses deterministic decoding. For cost, use per-model reported usage and the price applicable to that model; do not apply one price pair to a mixed-model trajectory.
