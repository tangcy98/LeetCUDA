# 项目知识库

**生成时间：** 2026-04-25
**Commit：** 067f25e
**分支：** main

## 概览

LeetCUDA 是一个面向学习的 CUDA kernel 仓库，包含 200+ 个带 PyTorch 绑定的 CUDA kernel 实现，并附带两个独立库：HGEMM（达到 cuBLAS 98-100% 性能）和 FFPA-Attention（大 head dimension 场景下比 SDPA 快 1.8-3 倍）。项目按主题组织，难度从朴素 CUDA Core kernel 逐步推进到 Tensor Core（WMMA → MMA → WGMMA → CuTe）。

## 授课入口：必须先查看 docs

当 agent 要帮助用户学习 CUDA kernel、跑通项目内容、解释代码或制定下一步学习计划时，必须先查阅 `docs/` 下的三份核心文档，并据此开始授课：

1. `docs/learning-history.md`：读取学生当前阶段、已知环境、已完成内容、仍然困惑的问题。
2. `docs/cuda-kernel-learning-roadmap.md`：定位用户当前主题属于哪个学习阶段，并确定前置知识和下一步练习。
3. `docs/agent-guide.md`：按照其中的会话流程执行：读历史、映射路线图、检查项目文件、运行最小测试、解释输出、做小修改、更新学习历史。

如果聊天上下文缺失，以 `docs/learning-history.md` 为准；如果历史为空，默认从阶段 0 开始，先检查环境并跑通 `kernels/elementwise`。不要只根据本文件直接授课，本文件只是入口，真正的教学流程以 `docs/` 为准。

## 目录结构

```
LeetCUDA/
├── kernels/           # 30 个主题目录，每个目录通常包含 X.cu + X.py + README.md
│   ├── elementwise/   # ⭐ Easy：激活函数/逐元素算子
│   ├── reduce/        # ⭐⭐ Medium：warp/block reduction
│   ├── sgemm/         # ⭐⭐⭐ Hard：FP32 GEMM
│   ├── hgemm/         # ⭐⭐⭐ Hard：FP16 GEMM（见 AGENTS.md）
│   ├── flash-attn/    # ⭐⭐⭐⭐ Hard+：FA2 MMA（见 AGENTS.md）
│   ├── openai-triton/ # Triton kernel 实现
│   ├── swizzle/       # SMEM swizzle 模式示例
│   └── ws-hgemm/      # Warp-specialized HGEMM（Hopper）
├── ffpa-attn/         # 独立 FFPA 库（见 AGENTS.md）
├── HGEMM/             # 独立 HGEMM 仓库副本（与 kernels/hgemm 对应）
├── third-party/       # CUTLASS submodule，不要修改
├── others/            # TensorRT / PyTorch distributed 示例
├── docs/              # 文档资料
└── slides/            # 演示材料
```

## 去哪里看

| 任务 | 位置 | 说明 |
|------|------|------|
| 新增简单 kernel | `kernels/{topic}/` | 参考 `kernels/elementwise/` 模式 |
| 新增 Tensor Core kernel | `kernels/hgemm/` 或 `kernels/flash-attn/` | 查看子目录的 `AGENTS.md` |
| 构建 HGEMM 库 | `kernels/hgemm/setup.py` | `python setup.py bdist_wheel` |
| 构建 FlashAttention | `kernels/flash-attn/setup.py` | 使用同类构建模式 |
| 构建 FFPA 库 | `ffpa-attn/setup.py` | 使用 `tools/build_fast.sh` 启用 ccache |
| CUTLASS/CuTe 参考 | `third-party/cutlass/` | Git submodule，只读参考 |
| Profiling 指南 | `kernels/nvidia-nsight/` | nsys / ncu 用法 |
| Swizzle 模式 | `kernels/swizzle/` | 避免 SMEM bank conflict |
| Triton kernels | `kernels/openai-triton/` | 纯 Python Triton 实现 |

## 约定

### Kernel 文件模式

每个 kernel 主题通常遵循：`{name}.cu`（CUDA）+ `{name}.py`（PyTorch 绑定与测试）+ `README.md`。
Python 文件使用 `torch.utils.cpp_extension.load()` 进行 JIT 编译。

### CUDA 命名

- **Kernel 函数**：`{op}_{dtype}[x{vec}][_pack]_kernel`，例如 `relu_f16x8_pack_kernel`
- **变体后缀**：`_naive`、`_sliced_k`、`_dbuf`（double buffer）、`_async`、`_stages`、`_swizzle`
- **Tensor Core 标记**：README 表格中的 `*` 表示使用 Tensor Cores
- **常量**：使用 `k` 前缀，例如 `kHeadDim`、`kMmaAtomM`、`kMmaTileSeqLenQ`
- **寄存器数组**：`R_Q`、`R_K`、`R_V`、`R_O`、`R_S`

### 数据类型变体

每个 kernel 通常按以下顺序扩展：`f32` → `f32x4` → `f16` → `f16x2` → `f16x8` → `f16x8_pack`。高级 kernel 会继续加入 `bf16`、`fp8_e4m3`、`fp8_e5m2`、`i8`。

### 代码风格

- **2 空格缩进**（clang-format 强制，基于 Google 风格）
- **100 字符行宽限制**
- 指针左对齐：`int* ptr`
- 使用 `_Pragma("unroll")`，不要使用 `#pragma unroll`
- Kernel 参数使用 `__restrict__`
- 使用 `__launch_bounds__` 提示 occupancy
- Python：black formatter，80 字符行宽，isort

### 编译参数

- C++：`-O3 -std=c++17`
- NVCC：`-U__CUDA_NO_HALF_OPERATORS__`、`-U__CUDA_NO_HALF_CONVERSIONS__`、`-U__CUDA_NO_BFLOAT16_CONVERSIONS__`
- Gencode：sm_80（Ampere）、sm_89（Ada）、sm_90（Hopper）

## 反模式（本项目）

- **不要** 编辑 `ffpa-attn/csrc/cuffpa/generated/` 中的文件，这些文件由 `env.py` 自动生成
- **不要** 修改 `third-party/`，这是 Git submodule，只读参考
- **不要** 修改 clang-format 的 `SortIncludes: false`，手动 include 顺序是有意保留的
- **不要** 提交 `.nsys*`、`.ncu*`、`.sqlite*`、构建产物；详见 `.gitignore`
- **绝不要** 在 CUDA/C++ 中使用 4 空格缩进，本项目使用 2 空格
- `notes-v1.cu` 已**废弃**，不要把它作为当前代码模式参考

## 常用命令

```bash
# 初始化 submodule
git submodule update --init --recursive --force

# 配置 pre-commit
pip install pre-commit && pre-commit install

# 运行一个简单 kernel（JIT 编译 + 测试）
cd kernels/relu && python relu.py

# 将 HGEMM 构建为库
cd kernels/hgemm && python setup.py bdist_wheel

# 构建 FFPA-ATTN（快速模式，使用 ccache）
cd ffpa-attn && bash tools/build_fast.sh bdist_wheel

# 构建 FFPA-ATTN（开发模式，指定 head dimensions）
FFPA_DEV_HEADDIMS=256,512 bash tools/build_fast.sh

# 跳过 CUDA extension（仅文档）
FFPA_SKIP_CUDA_EXT=1 pip install ".[docs]"
```

## 备注

- `HGEMM/` 是 `kernels/hgemm/` 的独立发布副本，避免两边同时修改
- Tensor Core 学习路径：WMMA（m16n16k16，更简单）→ MMA（m16n8k16，PTX 级控制）→ WGMMA（m64n128k16，Hopper）→ CuTe（CUTLASS 3.x 高层抽象）
- FlashAttention-2 MMA kernels 在小 attention（D≤64）上可能快于 FA2；大规模场景仍有差距。FFPA 面向 D>256
- `AGENTS.md` 在 `.gitignore` 中，属于本地知识文件
- `ffpa-attn` 需要 PyTorch >= 2.7.0 和 CUDA >= 13.0；`kernels/` 中的 kernel 可用较新的 PyTorch + CUDA JIT 编译
