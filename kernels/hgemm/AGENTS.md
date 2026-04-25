# HGEMM — Half-Precision GEMM (98-100% cuBLAS)

## OVERVIEW
Progressive HGEMM optimization from naive CUDA cores to Tensor Cores (WMMA→MMA→WGMMA→CuTe), achieving 98-100% of cuBLAS on L20/4090/3080. Also builds as standalone `toy-hgemm` Python package.

## STRUCTURE

```
hgemm/
├── naive/           # CUDA core implementations (f16, sliced-K, async)
├── wmma/            # WMMA Tensor Cores (m16n16k16)
├── mma/             # MMA PTX Tensor Cores (m16n8k16)
│   ├── basic/       #   Naive MMA, mma2x4, multi-stage
│   ├── swizzle/     #   SMEM swizzle variants (NN/TN layout)
│   └── others/      #   Experimental variants
├── wgmma/           # Hopper WGMMA (m64n128k16) with TMA
├── cutlass/         # CuTe high-level implementation
├── cublas/          # cuBLAS reference for benchmarking
├── bench/           # Profiling scripts (prof.py)
├── pybind/          # PyTorch C++ binding (hgemm_pybind.cc)
├── tools/           # Swizzle layout visualization
├── utils/           # utils.h — timing, cuBLAS wrappers
├── hgemm.cu         # Top-level includes (aggregates all kernels)
├── hgemm.py         # Test/benchmark driver (JIT or library mode)
├── setup.py         # Build as toy-hgemm wheel
└── README.md        # Comprehensive docs with benchmark tables
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add naive kernel variant | `naive/hgemm.cu` or `naive/hgemm_async.cu` | CUDA cores only |
| Add WMMA variant | `wmma/hgemm_wmma.cu` or `wmma/hgemm_wmma_stage.cu` | m16n16k16 |
| Add MMA variant | `mma/basic/hgemm_mma.cu` or `mma/basic/hgemm_mma_stage.cu` | m16n8k16 PTX |
| Add swizzle variant | `mma/swizzle/` | Bank-conflict-free SMEM |
| Add Hopper kernel | `wgmma/` | m64n128k16, requires sm_90 |
| Benchmark | `python hgemm.py --mma --plot` | Plots TFLOPS vs cuBLAS |
| Build library | `python setup.py bdist_wheel` | Produces `toy-hgemm` package |

## CONVENTIONS

### Optimization Progression (follow this order)
1. **Naive** → `hgemm_naive_f16` — baseline, no tiling
2. **Sliced-K** → `hgemm_sliced_k_f16` — loop over K dimension
3. **Thread tiling** → `hgemm_t_8x8_sliced_k_f16x4` — 8×8 per thread
4. **Packed loads** → `..._f16x8_pack` — 128-bit vectorized
5. **Double buffer** → `..._dbuf` — hide memory latency
6. **Async copy** → `..._async` — `cp.async` instructions
7. **WMMA** → `hgemm_wmma_m16n16k16_...` — first Tensor Core step
8. **MMA** → `hgemm_mma_m16n8k16_...` — PTX-level control
9. **Multi-stage** → `..._stages` — 2-4 stage pipeline
10. **Swizzle** → `..._swizzle` — SMEM bank conflict elimination
11. **CuTe** → `hgemm_mma_stages_..._cute` — CUTLASS 3.x abstraction
12. **WGMMA** → `hgemm_wgmma_m64n128k16_...` — Hopper only

### Kernel Naming
```
hgemm_{api}_{shape}_{features}_kernel
  api:      wmma | mma | wgmma | cute
  shape:    m16n16k16 | m16n8k16 | m64n128k16
  features: naive | mma2x4 | stages | dbuf | swizzle | tn
```

### Layout Convention
- Default: **NN** layout (both row-major)
- TN variants: `_tn` suffix — A transposed, B normal
- TN required for SMEM swizzle and CuTe implementations

## ANTI-PATTERNS

- `hgemm.cu` at root is an **include aggregator** — add new kernels in subdirs, not here
- `utils/utils.h` has cuBLAS dependency — include only when benchmarking
- Don't mix NN/TN layouts in same kernel — pick one per file
- WGMMA kernels require `sm_90` — guard with `#if` or separate gencode
