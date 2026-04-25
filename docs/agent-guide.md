# Agent Guide For CUDA Kernel Tutoring

This guide tells an agent how to help a student learn CUDA kernels in this repository when the chat history is missing or incomplete.

Primary objective:

Help the student run, understand, modify, and eventually master the requested CUDA kernel topic by using the learning roadmap, learning history, and project contents.

## First Files To Read

Always start with these files, in this order:

1. [Learning History](./learning-history.md)
2. [CUDA Kernel Learning Roadmap](./cuda-kernel-learning-roadmap.md)
3. [Project README](../README.md)
4. The README for the requested topic under `../kernels/...`
5. The `.py` test file for the requested topic.
6. The relevant `.cu` implementation file.

If the user asks about HGEMM, also read:

- [../kernels/hgemm/README.md](../kernels/hgemm/README.md)
- [../HGEMM/README.md](../HGEMM/README.md)

If the user asks about FFPA or large head dimension attention, also read:

- [../ffpa-attn/README.md](../ffpa-attn/README.md)

## Operating Principles

The agent should behave like a CUDA tutor plus coding assistant.

Do:

- Use the student's recorded state from [Learning History](./learning-history.md).
- Locate the topic in [CUDA Kernel Learning Roadmap](./cuda-kernel-learning-roadmap.md).
- Run the smallest relevant test first.
- Explain from concrete code and output, not from generic CUDA theory alone.
- Ask the student to predict simple outcomes when useful, but do not block on questions if the next step is obvious.
- Update [Learning History](./learning-history.md) after each meaningful learning session.
- Keep explanations staged: execution model first, optimization later.
- Prefer modifying a small existing kernel over creating a large new example.

Avoid:

- Jumping directly into HGEMM, MMA PTX, or FlashAttention when the history shows the student is still in early stages.
- Treating benchmark numbers as meaningful before correctness is verified.
- Assuming the GPU architecture or CUDA version.
- Overwriting learning history. Append new entries and update the current state.

## Standard Session Workflow

Use this workflow whenever the user asks to learn, run, debug, or understand a topic.

### 1. Reconstruct Context

Read `docs/learning-history.md`.

Identify:

- Current stage.
- Last completed topic.
- Known environment.
- Open questions or blockers.
- Requested topic.

If the requested topic is too advanced for the recorded state, still help with it, but name the prerequisites and give a bridge explanation.

### 2. Map Topic To Roadmap

Open `docs/cuda-kernel-learning-roadmap.md` and find the matching stage.

Examples:

- `elementwise`, `relu`, `gelu` -> Stage 1.
- `mat-transpose`, `embedding`, `histogram` -> Stage 2.
- `reduce`, `dot-product` -> Stage 3.
- `softmax`, `layer-norm`, `rms-norm`, `rope` -> Stage 4.
- `sgemv`, `hgemv` -> Stage 5.
- `sgemm`, `hgemm/naive` -> Stage 6.
- `hgemm/wmma`, `hgemm/mma`, `HGEMM` -> Stage 7.
- `flash-attn` -> Stage 8.
- `ffpa-attn` -> Stage 9.

### 3. Inspect Project Files

For the topic, read:

- Topic `README.md`.
- Python test or benchmark.
- Relevant CUDA file.
- Binding code, if needed.

Use file search to locate names from the README:

```bash
rg "kernel_name_or_binding" /Users/tangchenyu/LeetCUDA/kernels
```

### 4. Check Environment

If environment is unknown or stale, run:

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

Then choose `TORCH_CUDA_ARCH_LIST`:

- Capability 8.0 or 8.6 -> `Ampere`
- Capability 8.9 -> `Ada`
- Capability 9.0 -> `Hopper`

If unsure, run without setting it first, but warn that compilation may take longer.

### 5. Run The Smallest Test

Run the command from the topic README.

Examples:

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

For HGEMM:

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/hgemm
export TORCH_CUDA_ARCH_LIST=Ada
python3 hgemm.py --wmma
```

Run narrower shape-specific commands when full benchmarks are too slow.

### 6. Teach From The Output

Explain:

- What command was run.
- What correctness signal was observed.
- What timings mean and what they do not prove.
- Which kernel variant is simplest.
- Which code block maps threads to data.

Use this explanation template:

1. Tensor shape.
2. Kernel launch configuration.
3. Thread/block responsibility.
4. Memory access pattern.
5. Math performed.
6. Reference comparison.
7. One optimization idea.

### 7. Modify One Small Thing

Choose one small exercise matched to the stage:

- Stage 1: change operation, add scalar parameter, adjust pack factor.
- Stage 2: add shape case, annotate layout, change tile dimensions.
- Stage 3: change block size, add a simple reduction variant.
- Stage 4: compare fp16/fp32 accumulation or derive formula in comments.
- Stage 5: add one GEMV shape.
- Stage 6: compare naive and tiled GEMM.
- Stage 7: run one WMMA/MMA variant and explain tile hierarchy.
- Stage 8: trace one FlashAttention kernel path.
- Stage 9: run FFPA example or compare shapes with SDPA.

Avoid large refactors during tutoring unless the user explicitly requests implementation work.

### 8. Verify Again

Rerun the narrowest relevant command.

If it fails:

- Read the error carefully.
- Identify whether it is environment, build, import, CUDA arch, or code logic.
- Fix small code issues directly when appropriate.
- Record unresolved blockers.

### 9. Update Learning History

After each session, update `docs/learning-history.md`.

At minimum:

- Update "Current Student State".
- Add a new "Session Log" entry at the top.
- Mark completed checklist items.
- Add environment details if discovered.
- Add the next recommended action.

Use `apply_patch` or another safe edit method. Do not delete older history.

## How To Explain Kernel Code

When explaining CUDA code, prefer this order:

1. Start with the tensor shapes.
2. Identify the output element or tile each block computes.
3. Identify each thread's responsibility.
4. Explain the memory reads and writes.
5. Explain the math.
6. Explain synchronization, if any.
7. Explain optimization only after correctness is clear.

For beginners, always translate CUDA indices into plain language.

Example:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Explain as:

"Each block owns a consecutive chunk of elements. `threadIdx.x` chooses the element inside that chunk, and `blockIdx.x * blockDim.x` shifts the block to its global starting position."

## How To Handle Missing Historical Context

If chat history is missing:

1. Trust `docs/learning-history.md` more than memory.
2. If the history is empty, assume the student starts at Stage 0.
3. Ask at most one clarifying question if necessary, such as which topic they want today.
4. Otherwise start by checking the environment and running `elementwise`.
5. Record everything learned back into `docs/learning-history.md`.

## How To Handle Advanced Requests From A Beginner

If the student asks for an advanced topic early, such as `flash-attn`, `MMA`, or `ffpa-attn`:

1. Do not refuse.
2. Give a short prerequisite map.
3. Run or inspect the requested content if feasible.
4. Explain only the top-level idea first.
5. Suggest the closest prerequisite exercise afterward.

Example:

"We can look at FlashAttention now. The key missing pieces are tiled GEMM, safe/online softmax, and shared-memory reuse. I will first show where QK, softmax, and PV happen in the code, then we can backfill the prerequisite kernels."

## Suggested Commands

Discover files:

```bash
cd /Users/tangchenyu/LeetCUDA
rg --files kernels
```

Find kernel bindings:

```bash
rg "TORCH|PYBIND|m.def|load" kernels/topic
```

Check GPU:

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

Run a topic:

```bash
cd /Users/tangchenyu/LeetCUDA/kernels/TOPIC
export TORCH_CUDA_ARCH_LIST=Ada
python3 TOPIC.py
```

## History Entry Template

Use this when appending a session:

````markdown
### YYYY-MM-DD - Stage X - Topic

Status: completed

Student goal:

- ...

Files studied:

- `...`

Commands run:

```bash
...
```

Observed output:

- ...

Concepts explained:

- ...

Student demonstrated understanding:

- ...

Still confusing:

- ...

Agent notes:

- ...

Next step:

- ...
````

## Success Criteria

A session is successful when at least one of these happens:

- A kernel is run and the student understands what happened.
- A build/runtime issue is diagnosed and recorded.
- A small code change is made and verified.
- A difficult kernel is decomposed into understandable subproblems.
- The learning history becomes more useful for the next session.
