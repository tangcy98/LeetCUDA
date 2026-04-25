# HGEMM：半精度 GEMM（98-100% cuBLAS）

## 概览

本目录展示 HGEMM 从朴素 CUDA Core 到 Tensor Core（WMMA → MMA → WGMMA → CuTe）的渐进优化过程，在 L20 / 4090 / 3080 上可达到 cuBLAS 约 98-100% 的性能。它也可以构建为独立的 `toy-hgemm` Python package。

## 目录结构

```
hgemm/
├── naive/           # CUDA Core 实现（f16、sliced-K、async）
├── wmma/            # WMMA Tensor Cores（m16n16k16）
├── mma/             # MMA PTX Tensor Cores（m16n8k16）
│   ├── basic/       #   Naive MMA、mma2x4、multi-stage
│   ├── swizzle/     #   SMEM swizzle 变体（NN/TN layout）
│   └── others/      #   实验性变体
├── wgmma/           # Hopper WGMMA（m64n128k16）+ TMA
├── cutlass/         # CuTe 高层实现
├── cublas/          # cuBLAS benchmark reference
├── bench/           # Profiling 脚本（prof.py）
├── pybind/          # PyTorch C++ binding（hgemm_pybind.cc）
├── tools/           # Swizzle layout 可视化
├── utils/           # utils.h：计时、cuBLAS wrappers
├── hgemm.cu         # 顶层 include 聚合文件（汇总所有 kernels）
├── hgemm.py         # 测试/benchmark driver（JIT 或 library 模式）
├── setup.py         # 构建 toy-hgemm wheel
└── README.md        # 完整文档和 benchmark 表格
```

## 去哪里看

| 任务 | 位置 | 说明 |
|------|------|------|
| 新增 naive kernel 变体 | `naive/hgemm.cu` 或 `naive/hgemm_async.cu` | 仅 CUDA cores |
| 新增 WMMA 变体 | `wmma/hgemm_wmma.cu` 或 `wmma/hgemm_wmma_stage.cu` | m16n16k16 |
| 新增 MMA 变体 | `mma/basic/hgemm_mma.cu` 或 `mma/basic/hgemm_mma_stage.cu` | m16n8k16 PTX |
| 新增 swizzle 变体 | `mma/swizzle/` | 避免 SMEM bank conflict |
| 新增 Hopper kernel | `wgmma/` | m64n128k16，需要 sm_90 |
| Benchmark | `python hgemm.py --mma --plot` | 绘制 TFLOPS 并与 cuBLAS 对比 |
| 构建库 | `python setup.py bdist_wheel` | 生成 `toy-hgemm` package |

## 约定

### 优化递进顺序

1. **Naive** → `hgemm_naive_f16`：baseline，无 tiling
2. **Sliced-K** → `hgemm_sliced_k_f16`：沿 K 维循环
3. **Thread tiling** → `hgemm_t_8x8_sliced_k_f16x4`：每线程 8×8
4. **Packed loads** → `..._f16x8_pack`：128-bit 向量化
5. **Double buffer** → `..._dbuf`：隐藏内存延迟
6. **Async copy** → `..._async`：使用 `cp.async` 指令
7. **WMMA** → `hgemm_wmma_m16n16k16_...`：第一步 Tensor Core
8. **MMA** → `hgemm_mma_m16n8k16_...`：PTX 级控制
9. **Multi-stage** → `..._stages`：2-4 stage pipeline
10. **Swizzle** → `..._swizzle`：消除 SMEM bank conflict
11. **CuTe** → `hgemm_mma_stages_..._cute`：CUTLASS 3.x 抽象
12. **WGMMA** → `hgemm_wgmma_m64n128k16_...`：仅 Hopper

### Kernel 命名

```
hgemm_{api}_{shape}_{features}_kernel
  api:      wmma | mma | wgmma | cute
  shape:    m16n16k16 | m16n8k16 | m64n128k16
  features: naive | mma2x4 | stages | dbuf | swizzle | tn
```

### Layout 约定

- 默认：**NN** layout（A、B 都按 row-major）
- TN 变体：`_tn` 后缀，A transposed，B normal
- TN 是 SMEM swizzle 和 CuTe 实现所需的 layout

## 反模式

- 根目录的 `hgemm.cu` 是 **include 聚合文件**，新增 kernel 应放在子目录里，不要直接放这里
- `utils/utils.h` 依赖 cuBLAS，只在 benchmark 时 include
- 不要在同一个 kernel 里混用 NN/TN layout，每个文件选择一种 layout
- WGMMA kernels 需要 `sm_90`，请使用 `#if` guard 或单独的 gencode
