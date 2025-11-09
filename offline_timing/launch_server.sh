#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

# MODEL_NAME="Qwen/Qwen3-8B"
# python3 -u -m sgl_jax.launch_server \
# --model-path ${MODEL_NAME} \
# --trust-remote-code \
# --tp-size=4 \
# --device=tpu \
# --mem-fraction-static=0.8 \
# --chunked-prefill-size=2048 \
# --dtype=bfloat16 \
# --max-running-requests 256 \
# --skip-server-warmup \
# --page-size=128 \
# --disable-radix-cache

MODEL_NAME="Qwen/Qwen3-30B-A3B"
python3 -u -m sgl_jax.launch_server \
--model-path ${MODEL_NAME} \
--trust-remote-code \
--tp-size=4 \
--ep-size=2 \
--device=tpu \
--mem-fraction-static=0.8 \
--chunked-prefill-size=2048 \
--dtype=bfloat16 \
--max-running-requests 256 \
--skip-server-warmup \
--page-size=128 \
--disable-radix-cache
