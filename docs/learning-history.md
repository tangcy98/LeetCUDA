# CUDA Kernel 学习历史记录

本文件专门用于长期记录学生的学习状态。后续 agent 在给建议或运行命令前，必须先阅读本文件。

请与以下文档配合使用：

- [CUDA Kernel 学习路线图](./cuda-kernel-learning-roadmap.md)
- [Agent 指导文档](./agent-guide.md)

## 当前学生状态

最后更新：尚未开始

当前阶段：阶段 0：环境与项目入门

当前主题：无

当前目标：跑通第一个简单 CUDA kernel，并理解项目工作流。

已知环境：

- Repo：`/Users/tangchenyu/LeetCUDA`
- GPU：未知
- CUDA version：未知
- PyTorch version：未知
- 推荐 `TORCH_CUDA_ARCH_LIST`：未知

已知优势：

- 从零开始学习 CUDA kernel，学习路径可以按路线图完整推进。

已知短板：

- CUDA 执行模型。
- PyTorch extension 工作流。
- Kernel correctness 与 benchmark 输出解读。

下一步推荐动作：

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/elementwise
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

然后设置 `TORCH_CUDA_ARCH_LIST` 并运行：

```bash
python3 elementwise.py
```

## 学习记录

新记录追加在本节顶部。

### YYYY-MM-DD - 阶段 X - 主题名称

状态：未开始 | 进行中 | 已完成 | 阻塞

学生目标：

- 填写本次学习目标。

阅读过的文件：

- `path/to/file`

运行过的命令：

```bash
# command here
```

观察到的输出：

- 总结 correctness、timing、错误信息或构建失败。

解释过的概念：

- 概念 1
- 概念 2

学生已经展示出的理解：

- 学生能独立解释或实现的内容。

仍然困惑：

- 问题 1
- 问题 2

Agent 备注：

- 给下一个 agent 的重要上下文。
- 避免重复使用效果不好的解释方式。
- 记录本地环境的特殊问题。

下一步：

- 一个具体、可执行的下一步动作。

## 概念 Checklist

只有当学生能主动解释，而不是照着代码复述时，才勾选对应项目。

### 阶段 0：工作流

- [ ] 找到主题 README、CUDA source 和 Python test。
- [ ] 跑通一个 kernel。
- [ ] 解读 correctness 输出。
- [ ] 识别 GPU 与 compute capability。

### 阶段 1：Elementwise

- [ ] 解释 grid、block、thread index。
- [ ] 解释边界检查。
- [ ] 解释 fp32 与 fp16 kernel 变体。
- [ ] 解释向量化 load/store。
- [ ] 实现一个很小的新 elementwise kernel。

### 阶段 2：内存布局

- [ ] 将二维索引转换为线性 offset。
- [ ] 判断连续访问。
- [ ] 解释 coalescing。
- [ ] 解释 transpose 为什么比 elementwise 更难。
- [ ] 解释一个简单 atomic 使用场景。

### 阶段 3：Reduction

- [ ] 解释 warp reduction。
- [ ] 解释 block reduction。
- [ ] 解释 `__shfl_down_sync`。
- [ ] 解释 shared memory reduction。
- [ ] 对比 fp16 与 fp32 accumulation。

### 阶段 4：Softmax 与 Norm

- [ ] 推导 safe softmax。
- [ ] 解释 online softmax。
- [ ] 解释 LayerNorm 中的 reductions。
- [ ] 解释 RMSNorm。
- [ ] 对比 custom kernel 与 PyTorch 输出。

### 阶段 5：GEMV

- [ ] 解释 GEMV 是许多 dot product。
- [ ] 解释 K 维拆分。
- [ ] 解释 GEMV 为什么可能 memory-bound。
- [ ] 对比 sgemv 与 hgemv 变体。

### 阶段 6：CUDA Core GEMM

- [ ] 解释 naive GEMM。
- [ ] 解释 block tile。
- [ ] 解释 thread tile。
- [ ] 解释 shared memory reuse。
- [ ] 解释 double buffering。
- [ ] 高层解释 async copy。

### 阶段 7：Tensor Core HGEMM

- [ ] 解释 Tensor Core 的作用。
- [ ] 解释 WMMA。
- [ ] 解释 MMA PTX atom shape。
- [ ] 解释 warp tile 与 block tile。
- [ ] 解释 shared memory padding / swizzle。
- [ ] 对比 custom HGEMM 与 cuBLAS。

### 阶段 8：FlashAttention

- [ ] 解释 attention 公式。
- [ ] 解释为什么完整 attention matrix 很贵。
- [ ] 解释 tiled QK。
- [ ] 解释 attention 中的 online softmax。
- [ ] 解释 PV accumulation。
- [ ] 解释 split-KV 与 split-Q。
- [ ] 解释 shared KV/QKV。

### 阶段 9：FFPA

- [ ] 解释 large head dimension 问题。
- [ ] 高层解释 Split-D。
- [ ] 运行或阅读 FFPA examples。
- [ ] 对比 FFPA 与 SDPA。

## 故障排查记录

在这里记录反复出现的问题。

### Build 或 Import 失败

- 暂无已知问题。

### GPU 架构问题

- 暂无已知问题。

### 版本问题

- 暂无已知问题。
