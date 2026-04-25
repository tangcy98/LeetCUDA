# CUDA Kernel Learning Roadmap

This roadmap is designed for students learning CUDA kernels from zero with this repository. It should be used together with:

- [Learning History](./learning-history.md)
- [Agent Guide](./agent-guide.md)
- [Project README](../README.md)

The main loop for every topic is:

1. Read the topic README and skim the `.cu` / `.py` files.
2. Run the Python test or benchmark.
3. Explain the kernel mapping: tensor shape, grid, block, thread responsibility.
4. Compare output with PyTorch or cuBLAS reference.
5. Modify one small thing and rerun.
6. Record the result in [Learning History](./learning-history.md).

## Stage 0: Environment And Project Orientation

Goal: understand how this project is organized and how one CUDA kernel is exposed to Python.

Read:

- [../README.md](../README.md)
- [../kernels/elementwise/README.md](../kernels/elementwise/README.md)
- [../kernels/nvidia-nsight/README.md](../kernels/nvidia-nsight/README.md)

Run:

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/elementwise
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise.py
```

Adjust `TORCH_CUDA_ARCH_LIST` to the student's GPU architecture when needed, such as `Ampere`, `Ada`, or `Hopper`.

Must understand:

- What files usually exist for one topic: `README.md`, `.cu`, `.py`, and sometimes `setup.py` / pybind code.
- How Python calls a custom CUDA extension.
- How correctness and timing are printed.
- How to rerun after editing CUDA code.

Completion check:

- Student can run one existing kernel.
- Student can identify where the CUDA kernel, Python wrapper, and test data are defined.

## Stage 1: Elementwise Kernels

Goal: learn the CUDA execution model and simple memory access.

Study in this order:

1. [../kernels/elementwise](../kernels/elementwise)
2. [../kernels/relu](../kernels/relu)
3. [../kernels/sigmoid](../kernels/sigmoid)
4. [../kernels/elu](../kernels/elu)
5. [../kernels/gelu](../kernels/gelu)
6. [../kernels/swish](../kernels/swish)
7. [../kernels/hardswish](../kernels/hardswish)
8. [../kernels/hardshrink](../kernels/hardshrink)

Must understand:

- `blockIdx`, `threadIdx`, `blockDim`, `gridDim`.
- One thread per element.
- Boundary checks.
- `float`, `half`, `half2`.
- Vectorized load/store such as `float4` and packed `half`.
- Why memory bandwidth matters for elementwise kernels.

Hands-on tasks:

- Change elementwise add into multiply.
- Add a scalar affine kernel: `y = alpha * x + beta`.
- Compare scalar, `x2`, `x4`, and packed versions.

Completion check:

- Student can write a simple elementwise kernel without copying an existing implementation.
- Student can explain why a vectorized version can be faster.

## Stage 2: Memory Layout And Data Movement

Goal: understand contiguous access, coalescing, transpose, and shared memory motivation.

Study in this order:

1. [../kernels/embedding](../kernels/embedding)
2. [../kernels/mat-transpose](../kernels/mat-transpose)
3. [../kernels/histogram](../kernels/histogram)

Must understand:

- Row-major tensor layout.
- Coalesced vs non-coalesced global memory access.
- Alignment requirements for vectorized load/store.
- Why transpose is memory-bound and easy to make inefficient.
- When atomics are needed, using histogram as the entry point.

Hands-on tasks:

- Draw the mapping from a 2D matrix index to linear memory.
- Modify matrix sizes and observe timing.
- Add comments to one transpose kernel explaining read and write patterns.

Completion check:

- Student can predict which memory access is contiguous.
- Student can explain why transpose needs more care than elementwise add.

## Stage 3: Warp And Block Reduction

Goal: learn the core pattern behind dot product, softmax, layer norm, and attention.

Study in this order:

1. [../kernels/reduce](../kernels/reduce)
2. [../kernels/dot-product](../kernels/dot-product)

Must understand:

- Warp-level reduction.
- Block-level reduction.
- `__shfl_down_sync`.
- Shared memory reduction.
- Accumulation dtype: fp16 input with fp32 accumulate.
- Packed reductions for fp16, bf16, fp8, and int8 variants.

Hands-on tasks:

- Reimplement a simple fp32 sum reduction.
- Change block size and record performance.
- Compare fp16 accumulation vs fp32 accumulation for accuracy.

Completion check:

- Student can explain a tree reduction.
- Student can trace how one block produces one reduced value.

## Stage 4: Softmax, Online Softmax, And Normalization

Goal: combine reductions with numerically stable formulas used in deep learning.

Study in this order:

1. [../kernels/softmax](../kernels/softmax)
2. [../kernels/layer-norm](../kernels/layer-norm)
3. [../kernels/rms-norm](../kernels/rms-norm)
4. [../kernels/rope](../kernels/rope)

Must understand:

- Safe softmax: subtract row max.
- Online softmax: update max and denominator incrementally.
- Per-token kernel structure.
- Mean, variance, and `rsqrt`.
- RMSNorm vs LayerNorm.
- Why fp32 accumulation is usually used even when input is fp16.

Hands-on tasks:

- For one row of softmax, manually compute max, denominator, and output.
- Explain the two reductions used by LayerNorm.
- Compare PyTorch output and custom kernel output, including max absolute error.

Completion check:

- Student can derive safe softmax.
- Student can identify reduction phases in norm kernels.

## Stage 5: GEMV

Goal: bridge simple reductions and matrix multiplication.

Study in this order:

1. [../kernels/sgemv](../kernels/sgemv)
2. [../kernels/hgemv](../kernels/hgemv)
3. [../kernels/hgemv/hgemv_cute.cu](../kernels/hgemv/hgemv_cute.cu), only after the basic versions are clear.

Must understand:

- How one row or tile maps to one block.
- K dimension splitting.
- Dot product as the inner loop of GEMV.
- fp16 load with fp16 or fp32 accumulation.
- Why GEMV can be memory-bound.

Hands-on tasks:

- Draw the mapping from matrix row and vector column to threads.
- Compare `k16`, `k32`, and `k128` variants.
- Add a small shape case and verify output.

Completion check:

- Student can explain how a vector is reused across rows.
- Student can connect dot product to GEMV.

## Stage 6: CUDA Core GEMM

Goal: understand tiling before entering Tensor Core implementations.

Study in this order:

1. [../kernels/sgemm](../kernels/sgemm)
2. [../kernels/hgemm/naive](../kernels/hgemm/naive)

Must understand:

- Naive GEMM.
- C tile ownership by a block.
- Thread tile, such as `8x8`.
- Shared memory tile.
- Loop over K.
- Double buffering.
- Async copy, after the synchronous version is clear.

Hands-on tasks:

- Start from naive GEMM and identify repeated global memory loads.
- Explain what data is stored in shared memory.
- Compare versions from naive to tiled to double-buffered.

Completion check:

- Student can draw block tile, thread tile, and K loop.
- Student can explain why tiled GEMM improves data reuse.

## Stage 7: Tensor Core HGEMM

Goal: learn WMMA, MMA PTX, shared-memory swizzle, and performance-oriented GEMM.

Study in this order:

1. [../kernels/hgemm/README.md](../kernels/hgemm/README.md)
2. [../kernels/hgemm/wmma/hgemm_wmma.cu](../kernels/hgemm/wmma/hgemm_wmma.cu)
3. [../kernels/hgemm/wmma/hgemm_wmma_stage.cu](../kernels/hgemm/wmma/hgemm_wmma_stage.cu)
4. [../kernels/hgemm/mma/basic/hgemm_mma.cu](../kernels/hgemm/mma/basic/hgemm_mma.cu)
5. [../kernels/hgemm/mma/basic/hgemm_mma_stage.cu](../kernels/hgemm/mma/basic/hgemm_mma_stage.cu)
6. [../kernels/hgemm/mma/swizzle](../kernels/hgemm/mma/swizzle)
7. [../kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu](../kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu)
8. [../HGEMM/README.md](../HGEMM/README.md)

Must understand:

- What Tensor Cores accelerate.
- WMMA API vs MMA PTX.
- MMA atom shape, such as `m16n8k16`.
- Warp tile and block tile.
- Shared memory padding and swizzle.
- Multi-stage pipeline.
- Block swizzle and L2 locality.
- cuBLAS comparison methodology.

Hands-on tasks:

- Run a default WMMA benchmark.
- Run a default MMA benchmark.
- Pick one shape and compare cuBLAS, WMMA, MMA, and CuTe versions if the environment supports it.
- Draw the hierarchy: block tile -> warp tile -> MMA tile.

Completion check:

- Student can explain what each warp computes.
- Student can explain why MMA PTX can outperform a simple WMMA implementation.

## Stage 8: FlashAttention And Attention Kernels

Goal: understand how GEMM, softmax, and tiling combine in attention.

Study in this order:

1. [../kernels/flash-attn/README.md](../kernels/flash-attn/README.md)
2. [../kernels/flash-attn/cutlass/flash_attn_cute.cu](../kernels/flash-attn/cutlass/flash_attn_cute.cu)
3. [../kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu)
4. [../kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu)
5. [../kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu)
6. [../kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv.cu)
7. [../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu)
8. [../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu)

Must understand:

- Standard attention: `softmax(QK^T / sqrt(d)) V`.
- Why materializing the full attention matrix is expensive.
- Online softmax inside tiled attention.
- Split-KV vs split-Q.
- Shared KV and shared QKV memory reuse.
- Fine-grained QK and QKV tiling.
- Accuracy checks against PyTorch SDPA or FlashAttention reference.

Hands-on tasks:

- Explain one attention kernel with shapes `(B, H, N, D)`.
- Identify where Q, K, V, and O are loaded and stored.
- Trace one block over a Q tile and K/V tiles.

Completion check:

- Student can explain FlashAttention as tiled GEMM + online softmax + value accumulation.
- Student can distinguish memory-saving design from compute optimization.

## Stage 9: FFPA And Large Head Dimension Attention

Goal: study an advanced attention implementation after the FlashAttention basics are clear.

Study:

- [../ffpa-attn/README.md](../ffpa-attn/README.md)
- [../ffpa-attn/examples/run_ffpa_attn.py](../ffpa-attn/examples/run_ffpa_attn.py)

Must understand:

- Why standard FlashAttention-2 is limited for large head dimensions.
- What Split-D means in this project.
- O(1) SRAM complexity claim at a high level.
- Self-attention, cross-attention, GQA/MQA, and causal mode.

Hands-on tasks:

- Run the example if the local CUDA/PyTorch versions satisfy the requirements.
- Compare FFPA output with PyTorch SDPA.
- Record shape, dtype, GPU, runtime, and error.

Completion check:

- Student can explain what problem FFPA targets.
- Student can identify when FFPA is relevant and when normal FlashAttention is enough.

## Optional Track: Triton And CUTLASS

Use this track after the CUDA C++ basics are stable.

Study:

- [../kernels/openai-triton](../kernels/openai-triton)
- [../kernels/cutlass](../kernels/cutlass)
- [../third-party/cutlass/media/docs/cpp/quickstart.md](../third-party/cutlass/media/docs/cpp/quickstart.md)
- [../third-party/cutlass/media/docs/cpp/cute/00_quickstart.md](../third-party/cutlass/media/docs/cpp/cute/00_quickstart.md)

Purpose:

- Triton helps express kernels faster in Python.
- CUTLASS/CuTe helps understand production-grade Tensor Core tiling abstractions.

## Suggested Milestones

Milestone 1:

- Can run one elementwise kernel.
- Can edit and rerun it.

Milestone 2:

- Can explain warp/block reduction.
- Can write a simple sum reduction.

Milestone 3:

- Can explain softmax and layer norm kernels.
- Can reason about fp32 accumulation.

Milestone 4:

- Can explain tiled SGEMM.
- Can draw block tile, warp/thread work, and K loop.

Milestone 5:

- Can run and explain a WMMA HGEMM kernel.

Milestone 6:

- Can explain an MMA PTX HGEMM kernel and its tiling hierarchy.

Milestone 7:

- Can explain FlashAttention as a fusion of tiled QK, online softmax, and PV.

Milestone 8:

- Can investigate FFPA or another advanced kernel with the project docs and learning history.

## Recording Rules

After every session, update [Learning History](./learning-history.md) with:

- Date and environment.
- Topic and files studied.
- Commands run.
- Whether the run passed.
- What the student understood.
- What remains confusing.
- Next recommended step.

