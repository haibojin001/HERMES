#!/bin/bash
#SBATCH --job-name=serve_qwen3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
#
# Serve Qwen3-8B with vLLM on one GPU node and expose an OpenAI-compatible
# endpoint on port 8000. The solver runs elsewhere (the SWE-bench environment
# images are local to the machine that built them) and reaches the server through
# an SSH tunnel:
#
#     ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
#         -L 8000:<node>:8000 <login-host>
#     python -m hermes.pipeline --vllm Qwen/Qwen3-8B --concurrency 32 ...
#
# Two version pins, both found the hard way on a CUDA 12.8 driver:
#
#   vllm 0.11.0, not 0.29 - 0.29 depends on a CUDA 13 torch and the engine dies
#     with "NVIDIA driver on your system is too old (found version 12080)".
#     0.11.0 brings torch 2.8.0+cu128.
#   transformers 4.56.2, not 5.x - transformers 5 removed
#     Tokenizer.all_special_tokens_extended, which vllm 0.11 still calls:
#     "AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended".
#
# Install on the compute node, never on a login node: every login-node install we
# attempted was killed part-way through "Installing collected packages".
set -euo pipefail

VENV=${VENV:-$HOME/.venv-vllm}
MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-8000}
# Qwen3-8B's own limit is 40960 (max_position_embeddings, no rope scaling). The
# value below is ours, chosen so the KV cache allows enough concurrency; raise it
# if the localization prompts get clipped, lower it for more parallel requests.
MAX_LEN=${MAX_LEN:-32768}

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "vllm==0.11.0" "transformers==4.56.2"
fi

# The model must already be in the HF cache: compute nodes have no egress here.
export HF_HUB_OFFLINE=1

echo "node: $(hostname)   port: $PORT   model: $MODEL   max_len: $MAX_LEN"

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64
