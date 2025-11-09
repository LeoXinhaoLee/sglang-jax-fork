#!/bin/bash

num_prompts_per_concurrency=3
max_concurrency=128
num_prompts=$((num_prompts_per_concurrency * max_concurrency))
python -m sgl_jax.bench_serving --backend sgl-jax \
                                --dataset-name random \
                                --num-prompts ${num_prompts} \
                                --max-concurrency ${max_concurrency} \
                                --random-range-ratio 1 \
                                --random-input 1024 \
                                --random-output 1024 \
                                --warmup-requests 0
