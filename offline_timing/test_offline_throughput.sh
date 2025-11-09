#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH

python -m sgl_jax.bench_offline_throughput --model-path Qwen/Qwen3-30B-A3B \
                                           --dataset-name random \
                                           --num-prompts 32 \
                                           --random-range-ratio 1 \
                                           --random-input 1024 \
                                           --random-output 1024


