# CUDA Kernel Learning History

This file is intentionally reserved as the durable learning record for the student. Future agents should read this file before giving advice or running commands.

Use this together with:

- [CUDA Kernel Learning Roadmap](./cuda-kernel-learning-roadmap.md)
- [Agent Guide](./agent-guide.md)

## Current Student State

Last updated: not started

Current stage: Stage 0 - Environment And Project Orientation

Current topic: none

Current target: run the first simple CUDA kernel and understand the project workflow.

Known environment:

- Repo: `/Users/tangchenyu/LeetCUDA`
- GPU: unknown
- CUDA version: unknown
- PyTorch version: unknown
- Preferred `TORCH_CUDA_ARCH_LIST`: unknown

Known strengths:

- Starting from zero CUDA kernel background.

Known gaps:

- CUDA execution model.
- PyTorch extension workflow.
- Kernel correctness and benchmark interpretation.

Next recommended action:

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/elementwise
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

Then set `TORCH_CUDA_ARCH_LIST` and run:

```bash
python3 elementwise.py
```

## Session Log

Append new entries at the top of this section.

### YYYY-MM-DD - Stage X - Topic Name

Status: not started | in progress | completed | blocked

Student goal:

- Fill in the requested learning target.

Files studied:

- `path/to/file`

Commands run:

```bash
# command here
```

Observed output:

- Summarize correctness, timings, errors, or build failures.

Concepts explained:

- Concept 1
- Concept 2

Student demonstrated understanding:

- What the student could explain or implement independently.

Still confusing:

- Question 1
- Question 2

Agent notes:

- Important context for the next agent.
- Avoid repeating explanations that already worked poorly.
- Mention any local environment quirks.

Next step:

- Concrete next action.

## Concept Checklist

Mark items as done only when the student can explain them without merely repeating code.

### Stage 0: Workflow

- [ ] Locate topic README, CUDA source, and Python test.
- [ ] Run one kernel.
- [ ] Interpret correctness output.
- [ ] Identify GPU and compute capability.

### Stage 1: Elementwise

- [ ] Explain grid, block, and thread index.
- [ ] Explain boundary check.
- [ ] Explain fp32 vs fp16 kernel variants.
- [ ] Explain vectorized load/store.
- [ ] Implement a tiny new elementwise kernel.

### Stage 2: Memory Layout

- [ ] Convert 2D indices to linear offsets.
- [ ] Identify contiguous access.
- [ ] Explain coalescing.
- [ ] Explain why transpose is harder than elementwise.
- [ ] Explain a simple atomic operation use case.

### Stage 3: Reduction

- [ ] Explain warp reduction.
- [ ] Explain block reduction.
- [ ] Explain `__shfl_down_sync`.
- [ ] Explain shared memory reduction.
- [ ] Compare fp16 and fp32 accumulation.

### Stage 4: Softmax And Norm

- [ ] Derive safe softmax.
- [ ] Explain online softmax.
- [ ] Explain LayerNorm reductions.
- [ ] Explain RMSNorm.
- [ ] Compare custom kernel output with PyTorch output.

### Stage 5: GEMV

- [ ] Explain GEMV as many dot products.
- [ ] Explain K dimension splitting.
- [ ] Explain why GEMV can be memory-bound.
- [ ] Compare sgemv and hgemv variants.

### Stage 6: CUDA Core GEMM

- [ ] Explain naive GEMM.
- [ ] Explain block tile.
- [ ] Explain thread tile.
- [ ] Explain shared memory reuse.
- [ ] Explain double buffering.
- [ ] Explain async copy at a high level.

### Stage 7: Tensor Core HGEMM

- [ ] Explain Tensor Core purpose.
- [ ] Explain WMMA.
- [ ] Explain MMA PTX atom shape.
- [ ] Explain warp tile vs block tile.
- [ ] Explain shared memory padding/swizzle.
- [ ] Compare custom HGEMM with cuBLAS.

### Stage 8: FlashAttention

- [ ] Explain attention formula.
- [ ] Explain why full attention matrix is expensive.
- [ ] Explain tiled QK.
- [ ] Explain online softmax in attention.
- [ ] Explain PV accumulation.
- [ ] Explain split-KV vs split-Q.
- [ ] Explain shared KV/QKV.

### Stage 9: FFPA

- [ ] Explain large head dimension problem.
- [ ] Explain Split-D at a high level.
- [ ] Run or inspect FFPA examples.
- [ ] Compare FFPA with SDPA.

## Troubleshooting Notes

Record recurring issues here.

### Build Or Import Failures

- No known issue yet.

### GPU Architecture Issues

- No known issue yet.

### Version Issues

- No known issue yet.

