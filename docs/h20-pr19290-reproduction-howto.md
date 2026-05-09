# H20 Reproduction Howto: SGLang PR #19290 (DeepEP Waterfill)

This document captures the end-to-end procedure for running the SGLang PR #19290
reproduction (DeepEP Waterfill throughput benchmark) on the H20 cluster
(`10.6.131.5` + `10.6.131.6`). It is the operational companion to the canonical
`REPRODUCE_PR19290.md` doc that lives inside the worktree on the cluster — this
file documents how an agent or new operator should **find** and **drive** that
canonical reproduction.

Last validated: **2026-05-09** — baseline server brought up successfully on a
2-node 16×H20 cluster, `tput_bench.py` smoke run reported ~29.7–30.5k tok/s,
matching PR #19290's published baseline (~29.7k tok/s).

---

## 1. SSH onto H20 (node 0)

The H20 cluster has two head nodes:

| Node | IP | Hostname |
|---|---|---|
| Node 0 | `10.6.131.5` | `H20-GPU-05` |
| Node 1 | `10.6.131.6` | `H20-GPU-06` |

User: `xutingz` (passwordless SSH already configured).

```bash
ssh xutingz@10.6.131.5
```

Node 0 is the canonical entry point. Cross-node operations (kicking off the
node-1 server, cleaning state on both, etc.) are done by chaining
`ssh 10.6.131.6 ...` from inside node 0.

```bash
# From local workstation, you can chain through:
ssh -o StrictHostKeyChecking=no 10.6.131.5 "ssh -o BatchMode=yes 10.6.131.6 'echo node-1 OK'"
```

Both nodes have the SGLang dev container `sglang_waterfill` running
(`lmsysorg/sglang:dev` image). Verify before doing anything else:

```bash
ssh 10.6.131.5 "docker ps --filter name=sglang_waterfill"
ssh 10.6.131.6 "docker ps --filter name=sglang_waterfill"
```

If either container is missing or stopped, recreate it per Step 0.1 of the
canonical doc (referenced below).

## 2. Find the reproduction doc

The canonical reproduction guide lives **inside the worktree on the cluster**
(it is not committed to PR #19290's branch by design):

```
/lustre/raplab/client/xutingz/workspace/gitsrc/sglang_pr19290_clean/REPRODUCE_PR19290.md
```

The worktree on this path is checked out at branch `pr-19290`, commit
`43fe3d965d rm static rank load` (the canonical PR HEAD). Other related
directories on the cluster:

| Path | Contents |
|---|---|
| `gitsrc/sglang` | Working sglang clone, branch `feat/deepep-waterfill-eplb-balance`, used as the canonical install source per the doc's Step 0.2 |
| `gitsrc/sglang_pr19290_clean` | Read-only PR HEAD checkout — **the reproduction doc is here** |
| `gitsrc/sglang_moe_lb_waterfill` | PR HEAD + an extra "moe-lb integration" commit; this is what the running container currently has installed editably |
| `gitsrc/moe_load_balancer` | Local copy of the moe_load_balancer module (skeleton-era state on the cluster as of 2026-05-09; refresh from `xutizhou/moe_load_balancer` if you need the latest) |
| `bench/waterfill/` | Benchmark prompts, EPLB distribution file, `tput_bench.py`, and historical run logs |

Read the doc:

```bash
ssh 10.6.131.5 "cat /lustre/raplab/client/xutingz/workspace/gitsrc/sglang_pr19290_clean/REPRODUCE_PR19290.md" | less
```

The doc covers (at a glance): hardware/software requirements, container
creation, NVSHMEM/NCCL/IB env vars, MMLU prompt prep, EPLB recording, the
A/B throughput benchmark (Methods A / B / C), MMLU accuracy verification,
torch profile tracing, expected results, and an extensive troubleshooting
section.

## 3. Run the reproduction (Method B baseline smoke test)

Method B is the manual launch + `tput_bench.py` flow; it is the path that
exactly reproduces PR #19290's published numbers. Below is the **minimum**
you need to verify the reproduction works end-to-end. For a publishable A/B
comparison, follow the full doc (4 warmup + 8 measurement rounds for both
baseline and waterfill, then trimmed mean).

### 3.1 Verify environment

```bash
ssh 10.6.131.5 "docker exec sglang_waterfill bash -c '
  python3 -c \"
import torch, sgl_kernel, flashinfer, deep_ep, sglang
from sglang.srt.server_args import ServerArgs
print(\\\"PyTorch:\\\", torch.__version__)
print(\\\"sgl-kernel:\\\", sgl_kernel.__version__)
print(\\\"flashinfer:\\\", flashinfer.__version__)
print(\\\"sglang:\\\", sglang.__file__)
print(\\\"waterfill flag:\\\", hasattr(ServerArgs, \\\"enable_deepep_waterfill\\\"))
  \"
'"
```

Expected output (versions may shift; the `waterfill flag: True` check is the key one):

```
PyTorch: 2.9.1+cu129
sgl-kernel: 0.4.1
flashinfer: 0.6.4
sglang: /lustre/.../python/sglang/__init__.py
waterfill flag: True
```

### 3.2 Pre-flight: confirm the EPLB / prompt files exist

```bash
ssh 10.6.131.5 "ls -la \
  /lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_expert_dist/ep16_mmlu_logical_count.pt \
  /lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_bench_2k.json \
  /lustre/raplab/client/xutingz/workspace/bench/waterfill/tput_bench.py"
```

If any are missing, see Steps 1–2 of the canonical doc to (re)generate them.

### 3.3 Clean stale state

```bash
ssh 10.6.131.5 "for ip in 10.6.131.5 10.6.131.6; do
  ssh -o BatchMode=yes \$ip 'docker exec sglang_waterfill bash -c \"
    pkill -9 -f sglang 2>/dev/null
    rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null
  \"' &
done; wait"
```

### 3.4 Pick a non-conflicting port

The canonical doc uses port `40000` (port `30000` is often already taken by
other tenants on this cluster). If the chosen port is occupied, increment by
2 and update both the launch command and `tput_bench.py`'s `SERVER` variable.

```bash
# Quick check from host (containers use --network=host):
ssh 10.6.131.5 "netstat -tlnp 2>/dev/null | grep :40000"
ssh 10.6.131.6 "netstat -tlnp 2>/dev/null | grep :40000"
```

### 3.5 Launch the baseline server

Launch on **node 1 first**, sleep 10s, **then node 0**. Per the canonical
doc, the three NVSHMEM environment variables MUST be exported in the same
`docker exec` shell as the launch — they are not persisted across calls.

```bash
# Node 1 (10.6.131.6)
ssh 10.6.131.6 "docker exec -d sglang_waterfill bash -c '
  export NVSHMEM_IB_GID_INDEX=3
  export NVSHMEM_IBGDA_NIC_HANDLER=cpu
  export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
  python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --trust-remote-code --host 0.0.0.0 --port 40000 \
    --tp 16 --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --chunked-prefill-size -1 --disable-radix-cache \
    --max-prefill-tokens 8192 --max-running-requests 2048 \
    --load-balance-method round_robin \
    --watchdog-timeout 1800 --mem-fraction-static 0.75 \
    --disable-cuda-graph --skip-server-warmup \
    --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_expert_dist/ep16_mmlu_logical_count.pt \
    --dist-init-addr 10.6.131.5:20060 --nnodes 2 --node-rank 1 \
    > /tmp/baseline_node1_$(date +%F).log 2>&1
'"

sleep 10

# Node 0 (10.6.131.5)
ssh 10.6.131.5 "docker exec -d sglang_waterfill bash -c '
  export NVSHMEM_IB_GID_INDEX=3
  export NVSHMEM_IBGDA_NIC_HANDLER=cpu
  export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0
  python3 -m sglang.launch_server \
    --model-path /lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3 \
    --trust-remote-code --host 0.0.0.0 --port 40000 \
    --tp 16 --dp-size 16 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode normal \
    --chunked-prefill-size -1 --disable-radix-cache \
    --max-prefill-tokens 8192 --max-running-requests 2048 \
    --load-balance-method round_robin \
    --watchdog-timeout 1800 --mem-fraction-static 0.75 \
    --disable-cuda-graph --skip-server-warmup \
    --init-expert-location /lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_expert_dist/ep16_mmlu_logical_count.pt \
    --dist-init-addr 10.6.131.5:20060 --nnodes 2 --node-rank 0 \
    > /tmp/baseline_node0_$(date +%F).log 2>&1
'"
```

> **Pin all three** `NVSHMEM_*` env vars on every launch. Skipping them
> typically manifests as `socketStartConnect: exceeded retries` or
> `nvshmem setup connections failed` after a few minutes of waiting.

> **`--init-expert-location` for baseline too**: PR #19290's published gain
> (~+4%) is measured against an EPLB-warmed baseline. Without it the baseline
> uses round-robin dispatch and is slower, artificially inflating waterfill's
> apparent gain.

### 3.6 Wait for the server to be ready

Server load + warmup is typically 3–5 minutes for DeepSeek-V3 on this cluster.

```bash
until ssh 10.6.131.5 "curl -s -o /dev/null -w '%{http_code}' http://10.6.131.5:40000/health" | grep -q 200; do
  echo "$(date +%H:%M:%S) waiting..."
  sleep 30
done
echo "$(date +%H:%M:%S) server ready"
```

If the loop runs more than ~10 minutes, tail both nodes' logs and look for
`socketStartConnect`, `Capture cuda graph failed`, or NCCL/NVSHMEM errors —
see the canonical doc's Troubleshooting section.

### 3.7 Run the throughput benchmark

The canonical script is hard-coded to port 30000; if you used a different port
(e.g. 40000), patch `SERVER` inline:

```bash
ssh 10.6.131.5 "docker exec sglang_waterfill bash -c '
  sed s,http://10.6.131.5:30000,http://10.6.131.5:40000, \
    /lustre/raplab/client/xutingz/workspace/bench/waterfill/tput_bench.py > /tmp/tput_bench_40000.py
  CUDA_VISIBLE_DEVICES=99 python3 /tmp/tput_bench_40000.py baseline 2 2
'"
```

Arguments are `<tag> <warmup_rounds> <measurement_rounds>`. For a smoke test,
`2 2` is enough (~5 minutes total). For a publishable number, use `4 8` per
the canonical doc, then take the trimmed mean.

Expected smoke-test output (validated 2026-05-09):

```
[baseline] Loaded 2000 prompts, warmup=2, rounds=2
warmup-1: 15863 tok/s (584166 toks in 36.8s, 0 errs)   # first round absorbs JIT
warmup-2: 29708 tok/s (591149 toks in 19.9s, 0 errs)
round-1:  29648 tok/s
round-2:  30481 tok/s
```

`round-1`/`round-2` should land near the published ~29.7k tok/s baseline
(within a few percent of run-to-run variance).

### 3.8 Run the waterfill mode (optional, for the A/B comparison)

```bash
# Tear down baseline
ssh 10.6.131.5 "for ip in 10.6.131.5 10.6.131.6; do
  ssh -o BatchMode=yes \$ip 'docker exec sglang_waterfill bash -c \"
    pkill -9 -f sglang 2>/dev/null
    rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null
  \"' &
done; wait"

# Relaunch with --enable-deepep-waterfill (and a fresh --dist-init-addr port,
# e.g. 20062, to dodge any lingering TCP TIME_WAIT from the baseline run).
# Same launch command on both nodes, just add `--enable-deepep-waterfill`.

# After the server is ready, run:
ssh 10.6.131.5 "docker exec sglang_waterfill bash -c '
  CUDA_VISIBLE_DEVICES=99 python3 /tmp/tput_bench_40000.py waterfill 2 2
'"
```

Expected gain over baseline (with EPLB-warmed baseline): **~+3–4%**.
Compare the two `tput_bench.py` "Trimmed mean" lines — that's the headline
number from PR #19290.

### 3.9 Tear down

```bash
ssh 10.6.131.5 "for ip in 10.6.131.5 10.6.131.6; do
  ssh -o BatchMode=yes \$ip 'docker exec sglang_waterfill bash -c \"
    pkill -9 -f sglang 2>/dev/null
    rm -f /dev/shm/nccl* /dev/shm/nvshmem* 2>/dev/null
  \"' &
done; wait"
```

## 4. Common gotchas (short list)

The canonical doc has a 12-item troubleshooting section; the four most likely
to bite an agent driving this remotely:

1. **`http://10.6.131.5:30000` is unreachable** — port 30000 is in use by
   another tenant. Use 40000 (or any free port) and `sed`-patch
   `tput_bench.py`'s `SERVER` constant.
2. **`socketStartConnect: exceeded retries`** — one of the three
   `NVSHMEM_*` vars wasn't exported in the same `docker exec` shell as the
   launch. Re-export them and relaunch.
3. **First mode reports ~15k tok/s, second mode reports ~29k tok/s** —
   DeepGEMM JIT cache wasn't pre-warmed. Run a discard round (or the first
   warmup) before measuring; the cache persists across runs in
   `/root/.cache/deep_gemm/`.
4. **Waterfill shows +9–10% gain instead of +3–4%** — the baseline forgot
   `--init-expert-location`. Both modes need the EPLB file for a fair
   A/B.

## 5. What I did on 2026-05-09 (validation log)

- SSH'd to `10.6.131.5` from local workstation.
- Located the canonical doc at
  `/lustre/raplab/client/xutingz/workspace/gitsrc/sglang_pr19290_clean/REPRODUCE_PR19290.md`.
- Verified `sglang_waterfill` containers were up on both nodes, env was
  correct (`waterfill flag: True`), and the EPLB / prompt files were in
  place.
- Cleaned stale state on both nodes.
- Launched the baseline server on port 40000, `--dist-init-addr 10.6.131.5:20060`.
- Polled `/health` for ~3.5 minutes until the server came up.
- Patched `tput_bench.py` to point at port 40000 and ran with
  `2 warmup + 2 measurement` rounds (smoke test).
- Got `round-1: 29648 tok/s`, `round-2: 30481 tok/s` — matches PR #19290's
  published ~29.7k baseline.
- Cleanly tore the server down.

The full A/B comparison (baseline vs waterfill, 4+8 rounds each) was not run
in this validation pass — the goal was to verify the pipeline works
end-to-end, which it does.
