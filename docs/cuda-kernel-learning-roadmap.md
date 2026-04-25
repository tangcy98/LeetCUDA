# CUDA Kernel 学习路线图

这份路线图面向从零开始学习 CUDA kernel 的学生，必须与以下文档配合使用：

- [学习历史记录](./learning-history.md)
- [Agent 指导文档](./agent-guide.md)
- [项目 README](../README.md)

每个主题都按同一套学习循环推进：

1. 阅读主题 `README.md`，快速浏览 `.cu` 和 `.py` 文件。
2. 运行 Python 测试或 benchmark。
3. 解释 kernel 映射关系：tensor 形状、grid、block、每个 thread 负责什么。
4. 与 PyTorch、cuBLAS 或 SDPA reference 对比输出。
5. 做一个很小的修改并重新运行。
6. 将结果记录到 [学习历史记录](./learning-history.md)。

## 阶段 0：环境与项目入门

目标：理解本项目的组织方式，以及一个 CUDA kernel 如何暴露给 Python 调用。

阅读：

- [../README.md](../README.md)
- [../kernels/elementwise/README.md](../kernels/elementwise/README.md)
- [../kernels/nvidia-nsight/README.md](../kernels/nvidia-nsight/README.md)

运行：

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/elementwise
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise.py
```

如果学生的 GPU 不是 Ada，需要按实际架构调整 `TORCH_CUDA_ARCH_LIST`，例如 `Ampere`、`Ada` 或 `Hopper`。

必须理解：

- 一个主题目录通常包含什么：`README.md`、`.cu`、`.py`，复杂主题还可能有 `setup.py` 或 pybind 代码。
- Python 如何调用自定义 CUDA extension。
- correctness 和 timing 输出怎么看。
- 修改 CUDA 代码后如何重新运行。

完成标准：

- 学生能跑通一个已有 kernel。
- 学生能指出 CUDA kernel、Python wrapper、测试数据分别在哪里。

## 阶段 1：Elementwise Kernels

目标：学习 CUDA 执行模型和最简单的全局内存访问。

学习顺序：

1. [../kernels/elementwise](../kernels/elementwise)
2. [../kernels/relu](../kernels/relu)
3. [../kernels/sigmoid](../kernels/sigmoid)
4. [../kernels/elu](../kernels/elu)
5. [../kernels/gelu](../kernels/gelu)
6. [../kernels/swish](../kernels/swish)
7. [../kernels/hardswish](../kernels/hardswish)
8. [../kernels/hardshrink](../kernels/hardshrink)

必须理解：

- `blockIdx`、`threadIdx`、`blockDim`、`gridDim`。
- 一个 thread 处理一个元素的基本模式。
- 边界检查。
- `float`、`half`、`half2`。
- `float4` 和 packed `half` 这类向量化 load/store。
- 为什么 elementwise kernel 通常受 memory bandwidth 限制。

动手任务：

- 把 elementwise add 改成 multiply。
- 新增一个 scalar affine kernel：`y = alpha * x + beta`。
- 对比 scalar、`x2`、`x4`、packed 版本。

完成标准：

- 学生能不照抄现有代码，写出一个简单 elementwise kernel。
- 学生能解释为什么向量化版本可能更快。

## 阶段 2：内存布局与数据搬运

目标：理解连续访问、coalescing、transpose，以及为什么 shared memory 有价值。

学习顺序：

1. [../kernels/embedding](../kernels/embedding)
2. [../kernels/mat-transpose](../kernels/mat-transpose)
3. [../kernels/histogram](../kernels/histogram)

必须理解：

- Row-major tensor layout。
- Coalesced 与 non-coalesced global memory access。
- 向量化 load/store 的对齐要求。
- 为什么 transpose 是 memory-bound，且容易写低效。
- 什么时候需要 atomic，可从 histogram 入门。

动手任务：

- 画出二维矩阵索引到线性内存地址的映射。
- 修改矩阵大小并观察 timing。
- 给一个 transpose kernel 添加注释，说明读写访问模式。

完成标准：

- 学生能判断哪一类访问是连续的。
- 学生能解释 transpose 为什么比 elementwise add 更麻烦。

## 阶段 3：Warp 与 Block Reduction

目标：掌握 dot product、softmax、layer norm、attention 背后的核心并行模式。

学习顺序：

1. [../kernels/reduce](../kernels/reduce)
2. [../kernels/dot-product](../kernels/dot-product)

必须理解：

- Warp-level reduction。
- Block-level reduction。
- `__shfl_down_sync`。
- Shared memory reduction。
- Accumulation dtype：fp16 输入配 fp32 accumulate。
- fp16、bf16、fp8、int8 的 packed reduction 变体。

动手任务：

- 重新实现一个简单 fp32 sum reduction。
- 修改 block size 并记录性能。
- 对比 fp16 accumulate 与 fp32 accumulate 的精度差异。

完成标准：

- 学生能解释 tree reduction。
- 学生能追踪一个 block 如何产生一个 reduced value。

## 阶段 4：Softmax、Online Softmax 与 Normalization

目标：把 reduction 与深度学习中常见的数值稳定公式结合起来。

学习顺序：

1. [../kernels/softmax](../kernels/softmax)
2. [../kernels/layer-norm](../kernels/layer-norm)
3. [../kernels/rms-norm](../kernels/rms-norm)
4. [../kernels/rope](../kernels/rope)

必须理解：

- Safe softmax：先减 row max。
- Online softmax：增量更新 max 和 denominator。
- Per-token kernel 结构。
- Mean、variance、`rsqrt`。
- RMSNorm 与 LayerNorm 的差别。
- 为什么输入是 fp16 时通常仍使用 fp32 accumulation。

动手任务：

- 手算一行 softmax 的 max、denominator 和输出。
- 解释 LayerNorm 中的两次 reduction。
- 对比 PyTorch 输出与自定义 kernel 输出，包括最大绝对误差。

完成标准：

- 学生能推导 safe softmax。
- 学生能指出 norm kernel 中的 reduction 阶段。

## 阶段 5：GEMV

目标：从简单 reduction 过渡到矩阵乘。

学习顺序：

1. [../kernels/sgemv](../kernels/sgemv)
2. [../kernels/hgemv](../kernels/hgemv)
3. [../kernels/hgemv/hgemv_cute.cu](../kernels/hgemv/hgemv_cute.cu)，等基础版本理解清楚后再看。

必须理解：

- 一行或一个 tile 如何映射到一个 block。
- K 维如何拆分。
- GEMV 的内层就是 dot product。
- fp16 load 搭配 fp16 或 fp32 accumulation。
- 为什么 GEMV 常常是 memory-bound。

动手任务：

- 画出 matrix row、vector column 到 threads 的映射。
- 对比 `k16`、`k32`、`k128` 变体。
- 添加一个小 shape 并验证输出。

完成标准：

- 学生能解释 vector 如何跨多行复用。
- 学生能把 dot product 与 GEMV 联系起来。

## 阶段 6：CUDA Core GEMM

目标：在进入 Tensor Core 前，先理解 tiling。

学习顺序：

1. [../kernels/sgemm](../kernels/sgemm)
2. [../kernels/hgemm/naive](../kernels/hgemm/naive)

必须理解：

- Naive GEMM。
- 一个 block 负责 C 的一个 tile。
- Thread tile，例如 `8x8`。
- Shared memory tile。
- 沿 K 维循环。
- Double buffering。
- 在同步版本理解清楚后，再理解 async copy。

动手任务：

- 从 naive GEMM 开始，找出重复 global memory load。
- 解释 shared memory 里存了什么数据。
- 对比 naive、tiled、double-buffered 版本。

完成标准：

- 学生能画出 block tile、thread tile 和 K loop。
- 学生能解释 tiled GEMM 如何提高数据复用。

## 阶段 7：Tensor Core HGEMM

目标：学习 WMMA、MMA PTX、shared memory swizzle 和面向性能的 GEMM。

学习顺序：

1. [../kernels/hgemm/README.md](../kernels/hgemm/README.md)
2. [../kernels/hgemm/wmma/hgemm_wmma.cu](../kernels/hgemm/wmma/hgemm_wmma.cu)
3. [../kernels/hgemm/wmma/hgemm_wmma_stage.cu](../kernels/hgemm/wmma/hgemm_wmma_stage.cu)
4. [../kernels/hgemm/mma/basic/hgemm_mma.cu](../kernels/hgemm/mma/basic/hgemm_mma.cu)
5. [../kernels/hgemm/mma/basic/hgemm_mma_stage.cu](../kernels/hgemm/mma/basic/hgemm_mma_stage.cu)
6. [../kernels/hgemm/mma/swizzle](../kernels/hgemm/mma/swizzle)
7. [../kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu](../kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu)
8. [../HGEMM/README.md](../HGEMM/README.md)

必须理解：

- Tensor Core 加速什么。
- WMMA API 与 MMA PTX 的区别。
- MMA atom shape，例如 `m16n8k16`。
- Warp tile 与 block tile。
- Shared memory padding 与 swizzle。
- Multi-stage pipeline。
- Block swizzle 与 L2 locality。
- 与 cuBLAS 做对比的方法。

动手任务：

- 运行一个默认 WMMA benchmark。
- 运行一个默认 MMA benchmark。
- 如果环境支持，选一个 shape 对比 cuBLAS、WMMA、MMA、CuTe。
- 画出层次结构：block tile -> warp tile -> MMA tile。

完成标准：

- 学生能解释每个 warp 计算什么。
- 学生能解释 MMA PTX 为什么可能优于简单 WMMA 实现。

## 阶段 8：FlashAttention 与 Attention Kernels

目标：理解 GEMM、softmax、tiling 如何组合成 attention。

学习顺序：

1. [../kernels/flash-attn/README.md](../kernels/flash-attn/README.md)
2. [../kernels/flash-attn/cutlass/flash_attn_cute.cu](../kernels/flash-attn/cutlass/flash_attn_cute.cu)
3. [../kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_split_kv.cu)
4. [../kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu)
5. [../kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_share_kv.cu)
6. [../kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_share_qkv.cu)
7. [../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qk.cu)
8. [../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu](../kernels/flash-attn/mma/basic/flash_attn_mma_tiling_qkv.cu)

必须理解：

- 标准 attention：`softmax(QK^T / sqrt(d)) V`。
- 为什么不能轻易 materialize 完整 attention matrix。
- Tiled attention 中的 online softmax。
- Split-KV 与 Split-Q。
- Shared KV 与 shared QKV 的内存复用。
- 细粒度 QK 和 QKV tiling。
- 如何与 PyTorch SDPA 或 FlashAttention reference 做精度检查。

动手任务：

- 用形状 `(B, H, N, D)` 解释一个 attention kernel。
- 找到 Q、K、V、O 的 load/store 位置。
- 追踪一个 block 如何遍历一个 Q tile 和多个 K/V tile。

完成标准：

- 学生能把 FlashAttention 解释为 tiled GEMM + online softmax + value accumulation。
- 学生能区分节省内存的设计和提升计算效率的设计。

## 阶段 9：FFPA 与大 Head Dimension Attention

目标：在理解 FlashAttention 基础后，学习更高级的 attention 实现。

学习：

- [../ffpa-attn/README.md](../ffpa-attn/README.md)
- [../ffpa-attn/examples/run_ffpa_attn.py](../ffpa-attn/examples/run_ffpa_attn.py)

必须理解：

- 为什么标准 FlashAttention-2 对 large head dimension 有限制。
- 本项目中的 Split-D 是什么。
- O(1) SRAM complexity 的高层含义。
- Self-attention、cross-attention、GQA/MQA、causal mode。

动手任务：

- 如果本地 CUDA/PyTorch 版本满足要求，运行 example。
- 对比 FFPA 输出与 PyTorch SDPA。
- 记录 shape、dtype、GPU、runtime 和 error。

完成标准：

- 学生能解释 FFPA 解决什么问题。
- 学生能判断何时使用 FFPA，何时普通 FlashAttention 已足够。

## 可选路线：Triton 与 CUTLASS

当 CUDA C++ 基础稳定后再学习这一部分。

学习：

- [../kernels/openai-triton](../kernels/openai-triton)
- [../kernels/cutlass](../kernels/cutlass)
- [../third-party/cutlass/media/docs/cpp/quickstart.md](../third-party/cutlass/media/docs/cpp/quickstart.md)
- [../third-party/cutlass/media/docs/cpp/cute/00_quickstart.md](../third-party/cutlass/media/docs/cpp/cute/00_quickstart.md)

用途：

- Triton 能帮助用 Python 更快表达 kernel。
- CUTLASS/CuTe 能帮助理解生产级 Tensor Core tiling 抽象。

## 推荐里程碑

里程碑 1：

- 能跑通一个 elementwise kernel。
- 能修改并重新运行它。

里程碑 2：

- 能解释 warp/block reduction。
- 能写一个简单 sum reduction。

里程碑 3：

- 能解释 softmax 和 layer norm kernels。
- 能说明为什么要关注 fp32 accumulation。

里程碑 4：

- 能解释 tiled SGEMM。
- 能画出 block tile、warp/thread 工作和 K loop。

里程碑 5：

- 能运行并解释一个 WMMA HGEMM kernel。

里程碑 6：

- 能解释一个 MMA PTX HGEMM kernel 及其 tiling 层级。

里程碑 7：

- 能把 FlashAttention 解释为 tiled QK、online softmax 与 PV 的融合。

里程碑 8：

- 能结合项目文档和学习历史，独立调查 FFPA 或其他高级 kernel。

## 记录规则

每次学习 session 结束后，更新 [学习历史记录](./learning-history.md)，至少写入：

- 日期与环境。
- 学习主题和读过的文件。
- 运行过的命令。
- 是否跑通。
- 学生已经理解的内容。
- 仍然困惑的问题。
- 下一步推荐动作。
