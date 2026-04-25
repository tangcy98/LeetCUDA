# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指导。

## 项目概览

LeetCUDA 是一个结构化的 CUDA kernel 学习仓库，包含 200+ 个 kernel，并按 9 个阶段组织学习路径：从简单逐元素算子（Stage 1），到 Tensor Core HGEMM（Stage 7）、FlashAttention（Stage 8），再到大 head dimension 的 FFPA（Stage 9）。

Submodules：

- `third-party/cutlass`：NVIDIA CUTLASS，许多 kernel 使用其中的 CuTe include
- `ffpa-attn/`：FFPA，Faster Flash Prefill Attention，支持大 head dimension（≤1024）
- `HGEMM/`：独立 HGEMM 子库

## 构建方式

仓库根目录没有统一的构建入口。每个 kernel 目录都是自包含的。

**简单 kernels**（Stages 1-6）：首次运行时由 PyTorch JIT 编译，不需要单独构建：

```bash
cd kernels/elementwise
python3 elementwise.py
```

**带 `setup.py` 的复杂 kernels**（hgemm、flash-attn）：

```bash
cd kernels/hgemm
pip install -e .          # 构建 C extension
python3 hgemm.py --mma    # 运行 MMA kernels
```

**通过 `makefile` 构建二进制**（用于特定 SM target）：

```bash
cd kernels/hgemm
make mma_89               # 为 Ada Lovelace（SM 89）编译
./hgemm_mma_stage.89.bin  # 运行二进制
```

不同 kernel 支持的 SM target 可能不同：`80`（Ampere）、`89`（Ada）、`90`（Hopper）。JIT 构建时需要相应设置 `TORCH_CUDA_ARCH_LIST`，例如 `export TORCH_CUDA_ARCH_LIST=Ada`。

## 运行 Kernels / 测试

没有集中式测试入口。每个 kernel 都自带测试：Python 脚本会将自定义 kernel 与 PyTorch 或 cuBLAS reference 对比，并打印 correctness 与耗时。

```bash
# 简单 kernels
python3 kernels/relu/relu.py

# HGEMM（包含许多 CLI flags）
python3 kernels/hgemm/hgemm.py --wmma --M 4096 --N 4096 --K 2048 --check --verbose

# Flash Attention
python3 kernels/flash-attn/flash_attn_mma.py --check --sdpa

# Triton kernels
python3 kernels/openai-triton/fused-softmax/fused_softmax.py
```

常见 flags：`--check`（验证正确性）、`--warmup N`、`--iters N`、`--verbose`。

## 代码架构

### Kernel 模式（各主题基本一致）

```
kernels/<topic>/
├── <topic>.cu      # CUDA kernel + pybind11 bindings（PYBIND11_MODULE）
├── <topic>.py      # 通过 torch.utils.cpp_extension.load() JIT 编译 .cu，
│                   # 并与 PyTorch reference 做 benchmark
└── README.md       # 预期输出、运行命令、实现说明
```

复杂 kernels（hgemm、flash-attn）会按实现家族扩展子目录：

```
kernels/hgemm/
├── naive/          # CUDA Core baseline
├── wmma/           # WMMA API（m16n16k16）
├── mma/basic/      # 原始 MMA PTX，分阶段 pipeline
├── mma/swizzle/    # 加入 SMEM swizzle，避免 bank conflict
├── cutlass/        # 基于 CuTe
├── wgmma/          # SM90 WGMMA（仅 Hopper）
└── cublas/         # cuBLAS reference
```

### AI 辅导基础设施（`docs/`）

当 Claude 作为 CUDA 学习辅导 agent 工作时，会使用以下三份互相关联的文件：

- `docs/agent-guide.md`：AI agent 的 session 流程与操作说明
- `docs/cuda-kernel-learning-roadmap.md`：9 阶段课程路线图和每阶段主题映射
- `docs/learning-history.md`：持久化的学生学习进度记录，每次 session 更新

作为辅导 agent 时，应优先阅读 `agent-guide.md`，了解预期 session 协议。

## 代码风格

提交前运行 pre-commit，统一格式：

```bash
pre-commit run --all-files
```

- **CUDA/C++**：`clang-format` v18（排除 `ffpa-attn-mma/*`、`third-party/*`、`slides/*`）
- **Python**：`black`，80 字符行宽；`isort` 使用 black profile
- **C++ 标准**：大多数 kernel 使用 C++17；SM90/Hopper 代码使用 C++20

## 贡献新 Kernel

使用 `kernels/elementwise/` 作为模板。最小结构是：一个包含 `PYBIND11_MODULE` 的 `.cu` 文件，以及一个负责 JIT 编译和 benchmark 的 `.py` 文件。
