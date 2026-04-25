# KERNELS：200+ 个 CUDA Kernel 实现

## 概览

`kernels/` 下包含约 30 个主题目录，每个目录都是自包含的：`{name}.cu` + `{name}.py` + `README.md`。难度从逐元素算子逐步推进到基于 Tensor Cores 的 FlashAttention。

## 目录结构

```
kernels/
├── elementwise/     # ⭐ 新 kernel 的模板
├── relu,sigmoid,gelu,elu,swish,hardswish,hardshrink/  # ⭐ 激活函数
├── embedding,histogram/  # ⭐ 查表/计数类算子
├── reduce/          # ⭐⭐ Warp/block reduce（f32/f16/bf16/fp8/i8）
├── dot-product/     # ⭐⭐ Dot product 变体
├── softmax/         # ⭐⭐ Online safe softmax
├── layer-norm/      # ⭐⭐ Layer normalization
├── rms-norm/        # ⭐⭐ RMS normalization
├── rope/            # ⭐⭐ Rotary position embedding
├── mat-transpose/   # ⭐⭐ Transpose（包含 CuTe 变体）
├── nms/             # ⭐⭐ Non-maximum suppression
├── sgemv,hgemv/     # ⭐⭐⭐ GEMV（多种 K 维变体）
├── sgemm/           # ⭐⭐⭐ FP32 GEMM（naive → WMMA TF32）
├── hgemm/           # ⭐⭐⭐ FP16 GEMM（见 AGENTS.md，10 个子目录）
├── flash-attn/      # ⭐⭐⭐⭐ FlashAttention-2 MMA（见 AGENTS.md，6 个子目录）
├── swizzle/         # ⭐⭐⭐ SMEM swizzle 示例
├── ws-hgemm/        # ⭐⭐⭐ Warp-specialized HGEMM（Hopper CuTe）
├── openai-triton/   # ⭐⭐⭐ Triton：vector-add、softmax、layernorm、merge-attn
├── cutlass/         # ⭐⭐⭐ CuTe vector-add 示例
├── nvidia-nsight/   # nsys / ncu profiling 指南
├── transformer/     # 空目录，占位
└── notes-v1.cu      # 已废弃，不要参考
```

## 去哪里看

| 任务 | 位置 | 说明 |
|------|------|------|
| 新增简单 kernel | 复制 `elementwise/` 模式 | `.cu` + `.py` + `README.md` |
| 新增数据类型变体 | 现有 `.cu` 文件 | 按 f32 → f16 → f16x8_pack 的顺序扩展 |
| 新增 Tensor Core GEMM | `hgemm/` | 查看 `hgemm/AGENTS.md` |
| 新增 attention 变体 | `flash-attn/` | 查看 `flash-attn/AGENTS.md` |
| 新增 Triton kernel | `openai-triton/{name}/` | 纯 Python，`.py` + `README.md` |
| Benchmark kernel | 每个目录自己的 `.py` 文件 | 使用 `torch.cuda.Event` 计时 |

## 约定

### 新增 Kernel

1. 创建 `kernels/{topic}/` 目录
2. 编写 `{topic}.cu`，包含 kernel 实现
3. 编写 `{topic}.py`，通过 `torch.utils.cpp_extension.load()` 绑定 PyTorch
4. 编写 `README.md`，包含 kernel 表格（name、elem dtype、acc dtype、level）
5. 添加 `.gitignore`，忽略 `build/` 和 `*.so`

### Python Binding 模式

```python
lib = torch.utils.cpp_extension.load(
    name='my_kernel',
    sources=['my_kernel.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math',
                       '-U__CUDA_NO_HALF_OPERATORS__',
                       '-U__CUDA_NO_HALF_CONVERSIONS__']
)
```

每个 `.py` 都应与 PyTorch reference 对比正确性，并打印最大绝对误差。

### Kernel 变体命名

```
{op}_f32_kernel           # 基础 float32
{op}_f32x4_kernel         # 128-bit 向量化 load
{op}_f16_kernel           # 半精度
{op}_f16x2_kernel         # 2 元素 half vector
{op}_f16x8_kernel         # 8 元素 half vector
{op}_f16x8_pack_kernel    # packed 128-bit，使用 half2 arithmetic
{op}_bf16x8_pack_kernel   # BFloat16 packed
{op}_fp8_e4m3x16_pack_kernel  # FP8 packed
```

### 难度递进

- **Easy ⭐**：逐元素、激活函数，每种 dtype 一个 kernel 变体
- **Medium ⭐⭐**：Reduction、normalization，需要 warp/block 协作
- **Hard ⭐⭐⭐**：GEMM/GEMV，需要 tiling、shared memory、Tensor Cores（WMMA/MMA）
- **Hard+ ⭐⭐⭐⭐**：FlashAttention，需要 online softmax、multi-stage pipeline、QKV tiling
- **Hard++ ⭐⭐⭐⭐⭐**：FFPA、WGMMA，需要 O(1) SRAM、Hopper 架构知识

## 反模式

- **不要** 往 `transformer/` 添加文件，它只是占位目录，还没有稳定模式
- **不要** 参考 `notes-v1.cu`，它已经废弃，模式不符合当前代码风格
- 每个 kernel 目录必须自包含，不要在不同 topic 目录之间做 cross-directory import
- 共享工具只应放在复杂 topic 的 `{topic}/utils/` 中，例如 hgemm、flash-attn
