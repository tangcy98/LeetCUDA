# FLASH-ATTN — FlashAttention-2 with MMA PTX

## OVERVIEW
FlashAttention-2 implemented with pure MMA PTX instructions (m16n8k16). Variants progress from Split-KV (FA1) to Split-Q (FA2) with shared SMEM and fine-grained QKV tiling. Faster than FA2 for small attention (D≤64); FFPA handles D>256. Also builds as Python package.

## STRUCTURE

```
flash-attn/
├── mma/
│   ├── basic/       # Core variants: split_kv, split_q, share_kv, share_qkv, tiling_qk, tiling_qkv
│   │                # + F32 accumulation variants (*_F32F16F16F32.cu)
│   ├── swizzle/     # SMEM swizzle for Q/K/V (12 combinations × 2 acc types)
│   └── others/      # Register-reduced (_rr), experimental (_Os2g, _smooth)
├── cutlass/         # flash_attn_cute.cu — naive CuTe reference
├── pybind/          # PyTorch C++ binding
├── bench/           # Benchmark scripts
├── tools/           # Swizzle layout visualization
├── utils/           # utils.h — shared utilities
├── flash_attn_mma.py  # Test driver (all variants, correctness + perf)
├── setup.py         # Build as Python package
└── README.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add basic attention variant | `mma/basic/` | Follow `flash_attn_mma_split_q.cu` pattern |
| Add swizzle combination | `mma/swizzle/` | Name: `..._swizzle_{q|qk|qkv}.cu` |
| Add F32 accumulation | `mma/basic/` | Append `_F32F16F16F32` to filename |
| Add register-reduced | `mma/others/` | Append `_rr` suffix |
| Test all variants | `python flash_attn_mma.py --D 64` | Tests correctness vs SDPA |

## CONVENTIONS

### Attention Algorithm Variants (progressive optimization)
1. **Split KV** — Q,K,V split across warps (FA1-style, slower)
2. **Split Q** — Q split across warps, KV shared (FA2-style, faster)
3. **Shared KV** — K,V share same SMEM (½ SRAM vs FA2)
4. **Shared QKV** — Q,K,V fully share SMEM (¼ SRAM vs FA2)
5. **Tiling QK** — Fine-grained MMA-level Q@K^T tiling (O(16×d) SRAM, D→1024)
6. **Tiling QKV** — Full Q@K^T and P@V tiling (O(Br×16) SRAM, O(1) complexity)

### File Naming Pattern
```
flash_attn_mma_{variant}[_{swizzle}][_{acc}][_{modifier}].cu

variant:   split_kv | split_q | share_kv | share_qkv | tiling_qk | tiling_qkv
swizzle:   swizzle_q | swizzle_qk | swizzle_qkv
acc:       F32F16F16F32 (= MMA acc F32, elem F16, default is F16 acc)
modifier:  rr (register-reduced for d>128) | Os2g | smooth
```

### MMA Warp Tiling Layout
```
Split Q: 4 warps each own Br/4 rows of Q, all share full K/V
| warp 0 | MMA 0 ... MMA 0 (×Bc/8) |
| warp 1 | MMA 1 ... MMA 1 (×Bc/8) |
| warp 2 | MMA 2 ... MMA 2 (×Bc/8) |
| warp 3 | MMA 3 ... MMA 3 (×Bc/8) |
```

### Key Constants
- `Br`, `Bc` — block tile rows/columns for Q and K/V
- `kMmaAtomM=16, kMmaAtomN=8, kMmaAtomK=16` — MMA instruction shape
- `kMmaTileSeqLenQ`, `kMmaTileSeqLenK` — warp tiling factors

## ANTI-PATTERNS

- Don't mix Split-KV and Split-Q patterns in same kernel — fundamentally different warp assignments
- `_rr` variants trade register pressure for larger D support — not always faster
- Swizzle variants only apply to SMEM access, not algorithm logic — keep algorithm code identical
- Tiling QKV was refactored into `ffpa-attn/` — new tiling work should go there
