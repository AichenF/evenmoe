# Task Goal: SGLang Main Waterfill Integration

## Objective

Rebuild the Waterfill integration on top of latest SGLang `main`, not on the old PR #19290 branch, while using `moe_load_balancer` as an imported package instead of copying module code into SGLang.

## Required Shape

- Create/use a new SGLang worktree based on latest `origin/main`.
- Keep SGLang changes as minimal glue only:
  - server arg / feature switch
  - TopK routed-only hook
  - ModelRunner runtime injection
  - latest-main-specific empty-batch compatibility guard
  - small env flag needed by the adapter runtime
- Put SGLang/DeepEP Waterfill runtime and materializer code in `moe_load_balancer.adapters.sglang`, not in the SGLang tree.
- Preserve the previously validated Waterfill behavior from `sglang-pr19290-moe-lb`, including static/dynamic semantics and dynamic low-batch all-reduce ordering.

## Current Local Result

SGLang worktree:

- Path: `/Users/xutingz/workspace/gitsrc/sglang-main-moe-lb-waterfill`
- Branch: `moe-lb-waterfill-main`
- Base: latest `origin/main` at `1610aa77a`
- Commit: `8dd0d498d Integrate MoE load balancer Waterfill on main`
- Scope: SGLang glue only, `5 files changed, 119 insertions(+), 5 deletions(-)`

Module repo:

- Path: `/Users/xutingz/workspace/gitsrc/moe_load_balancer`
- Commit: `f1226ea Add SGLang DeepEP Waterfill adapter runtime`
- Scope: adds `src/moe_load_balancer/adapters/sglang/deepep_waterfill.py`

## Important Constraint

The old large SGLang commit `77fce52f2` was amended away. The current SGLang branch history contains only the smaller glue commit `8dd0d498d` (now `abe15ed1f` on `xutizhou/sglang`, after a rebase onto `1610aa77a`) on top of latest main.

## H20 Deployment Status (2026-05-09)

Partial validation completed against the H20 `sglang_waterfill` container.

### What works

- `xutizhou/sglang` `moe-lb-waterfill-main` (`abe15ed1f`) cloned as a sibling
  worktree at `/lustre/raplab/client/xutingz/workspace/gitsrc/sglang_main_moe_lb_waterfill`
  (via `git worktree add` from the existing `sglang_moe_lb_waterfill` clone, since
  `upstream` already points at `xutizhou/sglang`).
- `moe_load_balancer` cloned at
  `/lustre/raplab/client/xutingz/workspace/gitsrc/moe_load_balancer_main`,
  pulled to commit `c6a9d21` (`fix(policies): drop stale eager imports of
  removed waterfill names` — see "Bug found and fixed" below).
- `pip install --no-deps --no-build-isolation -e <path>` for both packages
  succeeds inside `sglang_waterfill` on both nodes.
- All Python-level imports succeed on both nodes:
  - `import sglang` resolves to the new worktree
  - `import moe_load_balancer` resolves to `moe_load_balancer_main`
  - `ServerArgs.enable_deepep_waterfill` flag is exposed
  - `moe_load_balancer.adapters.sglang.waterfill.runtime.WaterfillRuntime` imports
  - `moe_load_balancer.policies.l2.waterfill.WaterfillL2Router` imports

### What does NOT work yet

`python3 -m sglang.launch_server --enable-deepep-waterfill ...` fails to boot
because the H20 `sglang_waterfill` container is pinned to a PR #19290-era
dependency stack that is **ABI-incompatible with SGLang `main`'s pinned deps**:

| Component | Container has | SGLang main requires |
|---|---|---|
| `torch` | `2.9.1+cu129` | `2.11.0` (per `python/pyproject.toml`) |
| `sglang-kernel` | `0.4.1` | `0.4.2.post1` |
| `deep_gemm` (Python module) | namespace fragment at `/usr/local/lib/python3.12/dist-packages/deep_gemm` (no `__init__.py`, missing `utils/`) | full package incl. `deep_gemm.utils.layout.get_mn_major_tma_aligned_tensor` |

The torch mismatch is the load-bearing one — even after upgrading
`sglang-kernel` to `0.4.2.post1+cu129` (the cu129 build) and prepending
`PYTHONPATH=/lustre/.../DeepGEMM:$PYTHONPATH` to pick up the cluster's full
DeepGEMM source, the kernel `.so` cannot resolve
`c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, unsigned int, bool)`
against the container's torch 2.9.1.

### Bug found and fixed

`src/moe_load_balancer/policies/__init__.py` was eagerly importing
`SharedExpertWaterfillPolicy` and `WaterfillPolicyConfig` from
`policies.l2`, but the Stage 2 refactor in commit `4d6a547` had already
renamed the class to `WaterfillL2Router` and dropped `WaterfillPolicyConfig`.
The eager import broke any downstream import of
`moe_load_balancer.policies` (and transitively the SGLang adapter runtime).
`policies/l2/__init__.py`'s docstring already established the deferred-import
contract — `policies/__init__.py` now mirrors it: empty `__all__`, no eager
imports, and a docstring documenting the contract. Pushed as
`xutizhou/moe_load_balancer@c6a9d21`.

### What was left in place / rolled back

- New worktrees (`sglang_main_moe_lb_waterfill`, `moe_load_balancer_main`) are
  left in place on the cluster — they're harmless when not active, and the
  next attempt can pick them up.
- The container's editable installs were rolled back to
  `gitsrc/sglang_moe_lb_waterfill/python` and
  `gitsrc/moe_load_balancer` (the original PR #19290 setup), and
  `sglang-kernel` was rolled back to `0.4.1`. Re-verified `import sgl_kernel`
  works post-rollback. PR #19290 baseline reproduction is unaffected.
- All launched server processes were killed and `/dev/shm/{nccl,nvshmem}*`
  cleaned on both nodes.

### Path forward

To complete the on-cluster smoke test of the main-based glue, the container
needs to be torch 2.11 / sglang-kernel 0.4.2.post1+cu129 / DeepGEMM-with-utils
compatible. Two reasonable shapes:

1. **Bake a sibling container `sglang_waterfill_main` from a torch-2.11
   image** (e.g. `lmsysorg/sglang:dev` once the upstream image moves to
   torch 2.11, or `nvcr.io/nvidia/pytorch:25.10-py3`+`pip install sglang
   sglang-kernel==0.4.2.post1`). Same NFS mount, same env vars, but a
   parallel container so PR #19290 reproduction stays untouched.
2. **In-place upgrade the existing container's torch + deps** — riskier;
   destroys PR #19290 reproduction unless we snapshot the container first.

Option (1) is recommended.

