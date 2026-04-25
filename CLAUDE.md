# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeetCUDA is a structured CUDA kernel learning repository with 200+ kernels organized in a 9-stage curriculum — from simple elementwise ops (Stage 1) through Tensor Core HGEMM (Stage 7) and FlashAttention (Stage 8) to FFPA with large head dims (Stage 9).

Submodules:
- `third-party/cutlass` — NVIDIA CUTLASS (CuTe includes used by many kernels)
- `ffpa-attn/` — FFPA: Faster Flash Prefill Attention (large head dim ≤1024)
- `HGEMM/` — standalone HGEMM sub-library

## Building

There is no single root-level build. Each kernel directory is self-contained.

**Simple kernels** (Stages 1–6): JIT-compiled by PyTorch on first run — no separate build step:
```bash
cd kernels/elementwise
python3 elementwise.py
```

**Complex kernels with a `setup.py`** (hgemm, flash-attn):
```bash
cd kernels/hgemm
pip install -e .          # builds C extension
python3 hgemm.py --mma    # run MMA kernels
```

**Binary builds via `makefile`** (for SM-specific targets):
```bash
cd kernels/hgemm
make mma_89               # compile for Ada Lovelace (SM 89)
./hgemm_mma_stage.89.bin  # run the binary
```

Available SM targets vary by kernel: `80` (Ampere), `89` (Ada), `90` (Hopper). Set `TORCH_CUDA_ARCH_LIST` accordingly for JIT builds (e.g., `export TORCH_CUDA_ARCH_LIST=Ada`).

## Running Kernels / Tests

There is no centralized test runner. Each kernel is self-testing — the Python script compares the custom kernel against a PyTorch or cuBLAS reference and prints correctness + timing:

```bash
# Simple kernels
python3 kernels/relu/relu.py

# HGEMM (many CLI flags)
python3 kernels/hgemm/hgemm.py --wmma --M 4096 --N 4096 --K 2048 --check --verbose

# Flash Attention
python3 kernels/flash-attn/flash_attn_mma.py --check --sdpa

# Triton kernels
python3 kernels/openai-triton/fused-softmax/fused_softmax.py
```

Common flags across kernels: `--check` (verify correctness), `--warmup N`, `--iters N`, `--verbose`.

## Code Architecture

### Kernel Pattern (consistent across all topics)

```
kernels/<topic>/
├── <topic>.cu      # CUDA kernel + pybind11 bindings (PYBIND11_MODULE)
├── <topic>.py      # JIT-compiles the .cu via torch.utils.cpp_extension.load(),
│                   # then benchmarks vs. PyTorch reference
└── README.md       # Expected output, run commands, implementation notes
```

Complex kernels (hgemm, flash-attn) extend this with subdirectories per implementation family:
```
kernels/hgemm/
├── naive/          # CUDA Core baseline
├── wmma/           # WMMA API (m16n16k16)
├── mma/basic/      # Raw MMA PTX, staged pipelining
├── mma/swizzle/    # + SMEM swizzle for bank-conflict avoidance
├── cutlass/        # CuTe-based
├── wgmma/          # SM90 WGMMA (Hopper only)
└── cublas/         # cuBLAS reference
```

### AI Tutoring Infrastructure (`docs/`)

Three interlocking files used when Claude acts as a CUDA tutor:
- `docs/agent-guide.md` — session workflow and instructions for the AI agent
- `docs/cuda-kernel-learning-roadmap.md` — 9-stage curriculum with per-stage topic mapping
- `docs/learning-history.md` — persistent per-student progress log (updated each session)

When acting as tutor, read `agent-guide.md` first for the expected session protocol.

## Code Style

Pre-commit enforces all formatting — run before committing:
```bash
pre-commit run --all-files
```

- **CUDA/C++**: `clang-format` v18 (excludes `ffpa-attn-mma/*`, `third-party/*`, `slides/*`)
- **Python**: `black` at 80-char line length, `isort` with black profile
- **C++ standard**: C++17 for most kernels; C++20 for SM90/Hopper code

## Contributing a New Kernel

Use `kernels/elementwise/` as the template. The minimal structure is a `.cu` file with a `PYBIND11_MODULE` and a `.py` file that JIT-compiles and benchmarks it.
