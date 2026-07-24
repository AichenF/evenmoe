#!/usr/bin/env bash
# Single-node smoke launch for --ep-dispatch-algorithm=lp (moe_load_balancer-backed LPLB).
# Run inside the lmsysorg/sglang:dev container with PYTHONPATH pointing at the
# combined glue-tree + kernel-symlinks setup.
set -euo pipefail

MODEL=${MODEL:-/raid/model/DeepSeek-R1}
PORT=${PORT:-30001}
DIST_PORT=${DIST_PORT:-29501}
LOG=${LOG:-/workspace/moe_load_balancer/sglang-lplb-server.log}

unset NCCL_SOCKET_IFNAME
export PYTHONPATH=/workspace/sglang-glue/python

exec python -u -m sglang.launch_server \
    --model-path "$MODEL" \
    --tokenizer-path "$MODEL" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    --dist-init-addr "127.0.0.1:$DIST_PORT" \
    --tp-size 8 --dp-size 8 --ep-size 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --ep-num-redundant-experts 8 \
    --ep-dispatch-algorithm lp \
    --mem-fraction-static 0.75 \
    --max-running-requests 64 \
    --watchdog-timeout 1800 \
    --disable-radix-cache \
    --disable-cuda-graph \
    > "$LOG" 2>&1
