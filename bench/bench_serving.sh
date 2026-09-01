#!/usr/bin/env bash
# Clean online benchmark: 8k in / 1k out.
#
# Two corrections over the first attempt:
#   * vLLM's prefix caching is DISABLED. With it on and a fixed --seed, every
#     concurrency run replayed the same prompts and hit the cache, so "TTFT"
#     was a cache lookup (147 ms for 8000 tokens) rather than real prefill.
#   * each run gets its own seed, so no prompt is ever served twice.
# Prefill is measured separately with output-len 1 at concurrency 1, where TTFT
# is the time to prefill 8000 tokens.
set -u
S=/tmp/claude-1002/-home-shadeform-vllm/7a1849fb-7001-480e-a286-d749285cd88e/scratchpad
VENV=/home/shadeform/vllm/.venv/bin
M=/home/shadeform/vllm/models/Qwen3.8-27B-EXL3-5.5bpw

bench () {  # $1=port $2=conc $3=num_prompts $4=outlen $5=seed
  $VENV/vllm bench serve --backend openai --model $M --endpoint /v1/completions \
    --host 127.0.0.1 --port $1 --dataset-name random \
    --random-input-len 8000 --random-output-len $4 \
    --max-concurrency $2 --num-prompts $3 --ignore-eos --seed $5 2>&1 | \
    grep -E "Successful requests|Benchmark duration|Output token throughput|Total Token throughput|Mean TTFT|Mean TPOT"
}

sweep () {  # $1=port
  echo "--- PREFILL (8k in, 1 out, conc 1) ---"
  bench $1 1 4 1 101
  for C in 1 4 16 64; do
    case $C in 1) NP=4;; 4) NP=8;; 16) NP=16;; 64) NP=64;; esac
    echo "=== concurrency $C (num_prompts $NP) ==="
    bench $1 $C $NP 1000 $((200 + C))
  done
}

start_vllm () {  # $1=tp $2=port $3=tag
  VLLM_EXL3_BACKEND=native $VENV/vllm serve $M \
    --port $2 --tensor-parallel-size $1 --dtype bfloat16 \
    --max-model-len 10240 --max-num-batched-tokens 8192 --max-num-seqs 128 \
    --gpu-memory-utilization 0.90 --trust-remote-code --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' > $S/clean_$3.server.log 2>&1 &
  echo $!
}

wait_up () {  # $1=port $2=pid $3=path
  for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:$1$3 >/dev/null 2>&1 && { echo "up after $((i*5))s"; return 0; }
    kill -0 $2 2>/dev/null || { echo "SERVER DIED"; return 1; }
    sleep 5
  done
  return 1
}

echo "########## VLLM-EXL3 TP=1 (no prefix cache) ##########"
P=$(start_vllm 1 8000 tp1)
if wait_up 8000 $P /health; then sweep 8000; fi
kill $P 2>/dev/null; sleep 25

echo "########## VLLM-EXL3 TP=4 (no prefix cache) ##########"
P=$(start_vllm 4 8100 tp4)
if wait_up 8100 $P /health; then sweep 8100; fi
kill $P 2>/dev/null; sleep 30

echo "########## EXLLAMAV3 / TabbyAPI (1 GPU, chunk_size 2048) ##########"
python3 - <<'PY'
p = "/home/shadeform/tabbyAPI/config.yml"
s = open(p).read()
s = s.replace("chunk_size: 8192", "chunk_size: 2048").replace("cache_size: 655360", "cache_size: 262144")
open(p, "w").write(s)
PY
cd /home/shadeform/tabbyAPI
CUDA_VISIBLE_DEVICES=0 $VENV/python main.py > $S/clean_tabby.server.log 2>&1 &
TB=$!
if wait_up 5000 $TB /v1/models; then sweep 5000; fi
kill $TB 2>/dev/null
echo "CLEANDONE"
