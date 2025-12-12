# GPU Code Evaluation - Documentation

## Overview

Added GPU code evaluation tasks to assess the model's understanding of GPU programming concepts (CUDA, Triton) and ability to generate GPU code.

## New Evaluation Tasks

### 1. GPUCode - GPU Concepts Understanding

**Purpose**: Tests model's knowledge of GPU programming concepts, CUDA, and Triton.

**Dataset**: 15 questions covering:
- CUDA fundamentals (`__syncthreads()`, memory types, warp size)
- Memory optimization (coalescing, shared memory, tiling)
- Triton basics (`@triton.jit`, `tl.program_id`, `tl.load/store`)
- Performance concepts (warp divergence, block/grid dimensions)

**Evaluation Method**: Keyword-based matching (50% threshold)

**Example Questions**:
- "What is the purpose of __syncthreads() in CUDA?"
- "What is memory coalescing and why is it important?"
- "In Triton, what does the @triton.jit decorator do?"

---

### 2. GPUCodeGen - GPU Code Generation

**Purpose**: Tests model's ability to generate CUDA and Triton kernel code.

**Dataset**: 4 code generation tasks:
- CUDA vector addition kernel
- Triton element-wise multiplication
- CUDA memory allocation
- Triton matrix multiplication with tiling

**Evaluation Method**: Checks for presence of required keywords (100% threshold)

**Example Task**:
```
Prompt: "Write a simple CUDA kernel to add two vectors A and B"
Required keywords: ["__global__", "threadIdx", "blockIdx", "blockDim"]
```

---

## Usage

### Run GPU Code Evaluation

```bash
# Evaluate GPU concepts understanding
python -m scripts.chat_eval -i sft -a GPUCode

# Evaluate GPU code generation
python -m scripts.chat_eval -i sft -a GPUCodeGen

# Run both GPU tasks
python -m scripts.chat_eval -i sft -a "GPUCode|GPUCodeGen"

# Distributed evaluation
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GPUCode
```

### Include in Full Evaluation

```bash
# Run all tasks including GPU code evaluation
python -m scripts.chat_eval -i sft
```

---

## Files

### Created
- [gpucode.py](file:///Users/ibraniang/Desktop/nanogpu/kpt_nanochat/nanochat/tasks/gpucode.py) - GPU evaluation tasks

### Modified
- [chat_eval.py](file:///Users/ibraniang/Desktop/nanogpu/kpt_nanochat/nanochat/scripts/chat_eval.py) - Added GPU tasks to registry

---

## Evaluation Metrics

Both tasks are **generative** evaluations:
- **GPUCode**: Pass if ≥50% of keywords present in answer
- **GPUCodeGen**: Pass if 100% of required keywords present in generated code

**Baseline Accuracy**: 0% (open-ended tasks)

---

## Example Evaluation Output

```
GPUCode accuracy: 45.00%
GPUCodeGen accuracy: 25.00%
```

---

## Future Improvements

1. **Better Evaluation**: Use semantic similarity instead of keyword matching
2. **Code Execution**: Actually compile and run generated CUDA/Triton code
3. **Performance Prediction**: Add tasks to predict GPU kernel performance
4. **More Examples**: Expand dataset with more diverse GPU programming questions
5. **Real Datasets**: Integrate with existing GPU code benchmarks

---

## Integration with Qwen2.5

The GPU code evaluation complements the Qwen2.5-Coder architecture:
- Qwen2.5-Coder is trained on extensive code data including GPU code
- These evaluations test if the model learned GPU programming concepts
- Useful for assessing code-focused model capabilities

---

## Quick Reference

| Task | Type | Questions | Threshold | Baseline |
|------|------|-----------|-----------|----------|
| GPUCode | Generative | 15 | 50% keywords | 0% |
| GPUCodeGen | Generative | 4 | 100% keywords | 0% |
