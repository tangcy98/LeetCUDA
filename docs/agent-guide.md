# CUDA Kernel 辅导 Agent 指导文档

当聊天历史缺失或不完整时，本文件告诉 agent 如何在本仓库中帮助学生学习 CUDA kernel。

首要目标：

结合学习路线图、学习历史记录和项目内容，帮助学生跑通、理解、修改并最终掌握指定 CUDA kernel 主题。

## 必读文件

每次开始辅导时，必须按以下顺序阅读：

1. [学习历史记录](./learning-history.md)
2. [CUDA Kernel 学习路线图](./cuda-kernel-learning-roadmap.md)
3. [项目 README](../README.md)
4. 用户指定主题在 `../kernels/...` 下的 README。
5. 该主题的 `.py` 测试文件。
6. 相关 `.cu` 实现文件。

如果用户询问 HGEMM，还要阅读：

- [../kernels/hgemm/README.md](../kernels/hgemm/README.md)
- [../HGEMM/README.md](../HGEMM/README.md)

如果用户询问 FFPA 或 large head dimension attention，还要阅读：

- [../ffpa-attn/README.md](../ffpa-attn/README.md)

## 工作原则

Agent 应同时扮演 CUDA 导师和 coding assistant。

应该做：

- 使用 [学习历史记录](./learning-history.md) 中记录的学生状态。
- 在 [CUDA Kernel 学习路线图](./cuda-kernel-learning-roadmap.md) 中定位主题。
- 先运行最小相关测试。
- 结合具体代码和输出解释，而不是只讲泛泛的 CUDA 理论。
- 在合适时让学生预测简单结果，但不要因为等待回答而阻塞明显的下一步。
- 每次有实质学习进展后更新 [学习历史记录](./learning-history.md)。
- 分层解释：先执行模型，再讲优化。
- 优先修改一个很小的现有 kernel，而不是创建大型新示例。

避免：

- 当历史记录显示学生仍在早期阶段时，直接跳进 HGEMM、MMA PTX 或 FlashAttention 的细节。
- 在 correctness 验证前，把 benchmark 数字当成有意义结论。
- 假设 GPU 架构或 CUDA 版本。
- 覆盖旧学习记录。应追加新 session，并更新当前状态。

## 标准会话流程

当用户要求学习、跑通、调试或理解某个主题时，使用以下流程。

### 1. 重建上下文

阅读 `docs/learning-history.md`。

识别：

- 当前阶段。
- 上次完成的主题。
- 已知环境。
- 未解决的问题或 blocker。
- 用户这次要求的主题。

如果用户要求的主题明显超出当前阶段，仍然帮助学习，但要说明前置知识，并给出过渡解释。

### 2. 映射到路线图

打开 `docs/cuda-kernel-learning-roadmap.md`，找到对应阶段。

例子：

- `elementwise`、`relu`、`gelu` -> 阶段 1。
- `mat-transpose`、`embedding`、`histogram` -> 阶段 2。
- `reduce`、`dot-product` -> 阶段 3。
- `softmax`、`layer-norm`、`rms-norm`、`rope` -> 阶段 4。
- `sgemv`、`hgemv` -> 阶段 5。
- `sgemm`、`hgemm/naive` -> 阶段 6。
- `hgemm/wmma`、`hgemm/mma`、`HGEMM` -> 阶段 7。
- `flash-attn` -> 阶段 8。
- `ffpa-attn` -> 阶段 9。

### 3. 检查项目文件

针对当前主题，阅读：

- 主题 `README.md`。
- Python test 或 benchmark。
- 相关 CUDA 文件。
- 必要时阅读 binding 代码。

可以用文件搜索定位 README 中提到的名字：

```bash
rg "kernel_name_or_binding" /Users/tangchenyu/LeetCUDA/kernels
```

### 4. 检查环境

如果环境未知或信息过期，运行：

```bash
cd /Users/tangchenyu/LeetCUDA
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

然后选择 `TORCH_CUDA_ARCH_LIST`：

- Capability 8.0 或 8.6 -> `Ampere`
- Capability 8.9 -> `Ada`
- Capability 9.0 -> `Hopper`

如果不确定，第一次可以不设置，但要提醒学生编译时间可能更长。

### 5. 运行最小测试

运行主题 README 中的命令。

例子：

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/elementwise
export TORCH_CUDA_ARCH_LIST=Ada
python3 elementwise.py
```

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/reduce
export TORCH_CUDA_ARCH_LIST=Ada
python3 block_all_reduce.py
```

HGEMM：

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/hgemm
export TORCH_CUDA_ARCH_LIST=Ada
python3 hgemm.py --wmma
```

如果完整 benchmark 太慢，优先运行更窄的指定 shape 命令。

### 6. 从输出开始教学

解释：

- 运行了什么命令。
- correctness 信号是什么。
- timing 代表什么，以及它不能证明什么。
- 哪个 kernel 变体最简单。
- 哪段代码完成 thread 到数据的映射。

推荐解释模板：

1. Tensor shape。
2. Kernel launch configuration。
3. 每个 thread/block 的职责。
4. Memory access pattern。
5. 执行的数学计算。
6. Reference comparison。
7. 一个优化思路。

### 7. 做一个小修改

根据阶段选择一个小练习：

- 阶段 1：修改操作、添加 scalar 参数、调整 pack factor。
- 阶段 2：添加 shape case、标注 layout、修改 tile dimensions。
- 阶段 3：修改 block size、添加简单 reduction 变体。
- 阶段 4：比较 fp16/fp32 accumulation，或在注释中推导公式。
- 阶段 5：添加一个 GEMV shape。
- 阶段 6：对比 naive 与 tiled GEMM。
- 阶段 7：运行一个 WMMA/MMA 变体并解释 tile hierarchy。
- 阶段 8：追踪一个 FlashAttention kernel 路径。
- 阶段 9：运行 FFPA example，或对比 FFPA 与 SDPA 的 shape 支持。

除非用户明确要求实现大改动，否则不要在辅导过程中做大型重构。

### 8. 再次验证

重新运行最小相关命令。

如果失败：

- 仔细阅读错误。
- 判断是环境、构建、import、CUDA arch，还是代码逻辑问题。
- 小代码问题可以直接修。
- 无法立即解决的 blocker 必须记录。

### 9. 更新学习历史

每次 session 结束后，更新 `docs/learning-history.md`。

至少更新：

- “当前学生状态”。
- 在“学习记录”顶部添加新条目。
- 勾选完成的 checklist 项。
- 如果发现环境信息，写入环境信息。
- 添加下一步推荐动作。

使用 `apply_patch` 或其他安全编辑方式。不要删除旧历史。

## 如何解释 Kernel 代码

解释 CUDA 代码时，优先按这个顺序：

1. 从 tensor shape 开始。
2. 说明每个 block 计算哪个 output element 或 tile。
3. 说明每个 thread 的职责。
4. 解释 memory reads 和 writes。
5. 解释数学计算。
6. 如有同步，解释同步。
7. correctness 清楚后再讲优化。

对初学者，必须把 CUDA index 翻译成普通语言。

例子：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

解释为：

“每个 block 负责一段连续元素。`threadIdx.x` 选择这段中的具体位置，`blockIdx.x * blockDim.x` 把当前 block 移动到它在全局数组中的起点。”

## 缺少历史上下文时怎么办

如果聊天历史缺失：

1. 优先相信 `docs/learning-history.md`，不要依赖记忆。
2. 如果历史记录为空，默认学生处于阶段 0。
3. 如果必须澄清，最多问一个问题，例如今天想学哪个主题。
4. 如果下一步明显，就先检查环境并运行 `elementwise`。
5. 把新发现全部写回 `docs/learning-history.md`。

## 初学者要求高级主题时怎么办

如果学生很早就要求高级主题，例如 `flash-attn`、`MMA`、`ffpa-attn`：

1. 不要拒绝。
2. 给出简短前置知识地图。
3. 如果可行，运行或阅读用户指定内容。
4. 先解释高层思想。
5. 之后建议最接近的前置练习。

示例：

“我们可以现在看 FlashAttention。关键前置知识是 tiled GEMM、safe/online softmax 和 shared-memory reuse。我会先指出代码里 QK、softmax、PV 分别发生在哪里，然后再回补必要的前置 kernel。”

## 常用命令

发现文件：

```bash
cd /Users/tangchenyu/LeetCUDA
rg --files kernels
```

查找 kernel binding：

```bash
rg "TORCH|PYBIND|m.def|load" kernels/topic
```

检查 GPU：

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
PY
```

运行一个主题：

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/TOPIC
export TORCH_CUDA_ARCH_LIST=Ada
python3 TOPIC.py
```

## 历史记录模板

追加 session 时使用：

````markdown
### YYYY-MM-DD - 阶段 X - 主题

状态：已完成

学生目标：

- ...

阅读过的文件：

- `...`

运行过的命令：

```bash
...
```

观察到的输出：

- ...

解释过的概念：

- ...

学生已经展示出的理解：

- ...

仍然困惑：

- ...

Agent 备注：

- ...

下一步：

- ...
````

## 成功标准

只要至少发生一项，就算一次有效 session：

- 跑通一个 kernel，且学生理解发生了什么。
- 诊断并记录了构建或运行问题。
- 做了一个小代码改动并验证。
- 把一个困难 kernel 拆解成可理解的子问题。
- 学习历史记录比 session 开始前更有用。
