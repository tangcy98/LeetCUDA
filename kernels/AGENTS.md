# KERNELS — 200+ CUDA Kernel Implementations

## OVERVIEW
30 topic directories, each self-contained: `{name}.cu` + `{name}.py` + `README.md`. Progressive difficulty from element-wise ops to FlashAttention with Tensor Cores.

## STRUCTURE

```
kernels/
├── elementwise/     # ⭐ Template for new kernels
├── relu,sigmoid,gelu,elu,swish,hardswish,hardshrink/  # ⭐ Activation ops
├── embedding,histogram/  # ⭐ Lookup/counting ops
├── reduce/          # ⭐⭐ Warp/block reduce (f32/f16/bf16/fp8/i8)
├── dot-product/     # ⭐⭐ Dot product variants
├── softmax/         # ⭐⭐ Online safe softmax
├── layer-norm/      # ⭐⭐ Layer normalization
├── rms-norm/        # ⭐⭐ RMS normalization
├── rope/            # ⭐⭐ Rotary position embedding
├── mat-transpose/   # ⭐⭐ Transpose (incl. CuTe variant)
├── nms/             # ⭐⭐ Non-maximum suppression
├── sgemv,hgemv/     # ⭐⭐⭐ GEMV (K-dim variants)
├── sgemm/           # ⭐⭐⭐ FP32 GEMM (naive→WMMA TF32)
├── hgemm/           # ⭐⭐⭐ FP16 GEMM (→AGENTS.md, 10 subdirs)
├── flash-attn/      # ⭐⭐⭐⭐ FlashAttention-2 MMA (→AGENTS.md, 6 subdirs)
├── swizzle/         # ⭐⭐⭐ SMEM swizzle examples
├── ws-hgemm/        # ⭐⭐⭐ Warp-specialized HGEMM (Hopper CuTe)
├── openai-triton/   # ⭐⭐⭐ Triton: vector-add, softmax, layernorm, merge-attn
├── cutlass/         # ⭐⭐⭐ CuTe vector-add example
├── nvidia-nsight/   # nsys/ncu profiling guide
├── transformer/     # (empty, placeholder)
└── notes-v1.cu      # DEPRECATED — do not reference
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new simple kernel | Copy `elementwise/` pattern | `.cu` + `.py` + `README.md` |
| Add data type variant | Existing `.cu` file | Follow f32→f16→f16x8_pack progression |
| Add Tensor Core GEMM | `hgemm/` | See `hgemm/AGENTS.md` |
| Add attention variant | `flash-attn/` | See `flash-attn/AGENTS.md` |
| Add Triton kernel | `openai-triton/{name}/` | Python-only, `.py` + `README.md` |
| Benchmark kernel | Each dir's `.py` file | Uses `torch.cuda.Event` timing |

## CONVENTIONS

### Adding a New Kernel
1. Create `kernels/{topic}/` directory
2. Write `{topic}.cu` with kernel implementations
3. Write `{topic}.py` with PyTorch binding via `torch.utils.cpp_extension.load()`
4. Write `README.md` with kernel table (name, elem dtype, acc dtype, level)
5. Add `.gitignore` ignoring `build/` and `*.so`

### Python Binding Pattern
```python
lib = torch.utils.cpp_extension.load(
    name='my_kernel',
    sources=['my_kernel.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math',
                       '-U__CUDA_NO_HALF_OPERATORS__',
                       '-U__CUDA_NO_HALF_CONVERSIONS__']
)
```
Each `.py` tests correctness against PyTorch reference, prints max absolute error.

### Kernel Variant Naming
```
{op}_f32_kernel           # Basic float32
{op}_f32x4_kernel         # 128-bit vectorized load
{op}_f16_kernel           # Half precision
{op}_f16x2_kernel         # 2-element half vector
{op}_f16x8_kernel         # 8-element half vector
{op}_f16x8_pack_kernel    # Packed 128-bit with half2 arithmetic
{op}_bf16x8_pack_kernel   # BFloat16 packed
{op}_fp8_e4m3x16_pack_kernel  # FP8 packed
```

### Difficulty Progression
- **Easy ⭐**: Element-wise, activation — one kernel per dtype variant
- **Medium ⭐⭐**: Reduction, normalization — warp/block coordination
- **Hard ⭐⭐⭐**: GEMM/GEMV — tiling, shared memory, Tensor Cores (WMMA/MMA)
- **Hard+ ⭐⭐⭐⭐**: FlashAttention — online softmax, multi-stage pipeline, QKV tiling
- **Hard++ ⭐⭐⭐⭐⭐**: FFPA, WGMMA — O(1) SRAM, Hopper architecture

## ANTI-PATTERNS

- **DO NOT** add files to `transformer/` — placeholder, no pattern established
- **DO NOT** reference `notes-v1.cu` — deprecated, patterns don't match current style
- Each kernel dir must be self-contained — no cross-directory imports between topic dirs
- Shared utilities go in `{topic}/utils/` only for complex topics (hgemm, flash-attn)
