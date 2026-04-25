# FLASH-ATTN：基于 MMA PTX 的 FlashAttention-2

## 概览

这里实现的是使用纯 MMA PTX 指令（m16n8k16）的 FlashAttention-2。变体从 Split-KV（FA1 风格）逐步优化到 Split-Q（FA2 风格），再加入 shared SMEM 和细粒度 QKV tiling。小 attention（D≤64）上可能快于 FA2；D>256 的场景由 FFPA 处理。本目录也可以构建为 Python package。

## 目录结构

```
flash-attn/
├── mma/
│   ├── basic/       # 核心变体：split_kv、split_q、share_kv、share_qkv、tiling_qk、tiling_qkv
│   │                # 以及 F32 accumulation 变体（*_F32F16F16F32.cu）
│   ├── swizzle/     # Q/K/V 的 SMEM swizzle（12 种组合 × 2 种 acc 类型）
│   └── others/      # 减少寄存器用量（_rr）、实验性变体（_Os2g、_smooth）
├── cutlass/         # flash_attn_cute.cu，朴素 CuTe reference
├── pybind/          # PyTorch C++ binding
├── bench/           # Benchmark scripts
├── tools/           # Swizzle layout 可视化
├── utils/           # utils.h，共享工具
├── flash_attn_mma.py  # 测试驱动（所有变体，正确性 + 性能）
├── setup.py         # 构建为 Python package
└── README.md
```

## 去哪里看

| 任务 | 位置 | 说明 |
|------|------|------|
| 新增基础 attention 变体 | `mma/basic/` | 参考 `flash_attn_mma_split_q.cu` 模式 |
| 新增 swizzle 组合 | `mma/swizzle/` | 命名：`..._swizzle_{q|qk|qkv}.cu` |
| 新增 F32 accumulation | `mma/basic/` | 文件名追加 `_F32F16F16F32` |
| 新增 register-reduced 变体 | `mma/others/` | 追加 `_rr` 后缀 |
| 测试所有变体 | `python flash_attn_mma.py --D 64` | 与 SDPA 对比正确性 |

## 约定

### Attention 算法变体（渐进优化）

1. **Split KV**：Q、K、V 跨 warps 拆分（FA1 风格，较慢）
2. **Split Q**：Q 跨 warps 拆分，KV 共享（FA2 风格，更快）
3. **Shared KV**：K、V 共用同一块 SMEM（相比 FA2 节省 1/2 SRAM）
4. **Shared QKV**：Q、K、V 完全共用 SMEM（相比 FA2 节省 1/4 SRAM）
5. **Tiling QK**：MMA 级别的细粒度 Q@K^T tiling（O(16×d) SRAM，D 可到 1024）
6. **Tiling QKV**：完整 Q@K^T 与 P@V tiling（O(Br×16) SRAM，O(1) 复杂度）

### 文件命名模式

```
flash_attn_mma_{variant}[_{swizzle}][_{acc}][_{modifier}].cu

variant:   split_kv | split_q | share_kv | share_qkv | tiling_qk | tiling_qkv
swizzle:   swizzle_q | swizzle_qk | swizzle_qkv
acc:       F32F16F16F32（MMA acc F32，elem F16；默认是 F16 acc）
modifier:  rr（为 d>128 减少寄存器用量）| Os2g | smooth
```

### MMA Warp Tiling 布局

```
Split Q：4 个 warps 分别拥有 Br/4 行 Q，所有 warp 共享完整 K/V
| warp 0 | MMA 0 ... MMA 0 (×Bc/8) |
| warp 1 | MMA 1 ... MMA 1 (×Bc/8) |
| warp 2 | MMA 2 ... MMA 2 (×Bc/8) |
| warp 3 | MMA 3 ... MMA 3 (×Bc/8) |
```

### 关键常量

- `Br`、`Bc`：Q 与 K/V 的 block tile 行数/列数
- `kMmaAtomM=16, kMmaAtomN=8, kMmaAtomK=16`：MMA 指令形状
- `kMmaTileSeqLenQ`、`kMmaTileSeqLenK`：warp tiling 因子

## 反模式

- 不要在同一个 kernel 里混用 Split-KV 和 Split-Q 模式，它们的 warp 分工本质不同
- `_rr` 变体用更低 register pressure 换取更大 D 支持，不一定总是更快
- Swizzle 变体只影响 SMEM 访问，不改变算法逻辑；算法代码应保持一致
- Tiling QKV 已重构进 `ffpa-attn/`，新的 tiling 工作优先放那里
