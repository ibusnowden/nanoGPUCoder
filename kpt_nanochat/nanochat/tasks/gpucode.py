"""
GPU Code Evaluation Task
Evaluates model's understanding of GPU programming (CUDA, Triton) and performance characteristics.
"""

import re
from tasks.common import Task


# Sample GPU code evaluation dataset
GPU_CODE_EXAMPLES = [
    {
        "prompt": "What is the purpose of __syncthreads() in CUDA?",
        "answer": "synchronize threads within a block",
        "keywords": ["synchronize", "barrier", "block", "threads"],
    },
    {
        "prompt": "In CUDA, what is the difference between global memory and shared memory?",
        "answer": "shared memory is faster and shared within a block, global memory is slower and accessible by all threads",
        "keywords": ["shared", "faster", "block", "global", "slower"],
    },
    {
        "prompt": "What is memory coalescing in CUDA and why is it important?",
        "answer": "accessing contiguous memory locations in a single transaction for better performance",
        "keywords": ["contiguous", "memory", "performance", "transaction", "bandwidth"],
    },
    {
        "prompt": "In Triton, what does the @triton.jit decorator do?",
        "answer": "compiles the function to GPU kernel code",
        "keywords": ["compile", "kernel", "jit", "gpu"],
    },
    {
        "prompt": "What is the purpose of tl.program_id in Triton?",
        "answer": "gets the current program/block ID for parallel execution",
        "keywords": ["block", "id", "parallel", "index"],
    },
    {
        "prompt": "Why is tiling important in GPU matrix multiplication?",
        "answer": "reduces global memory accesses by reusing data in shared memory",
        "keywords": ["shared", "memory", "reuse", "cache", "performance"],
    },
    {
        "prompt": "What is warp divergence in CUDA?",
        "answer": "when threads in a warp take different execution paths causing serialization",
        "keywords": ["divergence", "branch", "warp", "serialization", "performance"],
    },
    {
        "prompt": "What is the typical size of a CUDA warp?",
        "answer": "32 threads",
        "keywords": ["32"],
    },
    {
        "prompt": "In CUDA, what is the difference between blockDim and gridDim?",
        "answer": "blockDim is threads per block, gridDim is blocks per grid",
        "keywords": ["threads", "block", "blocks", "grid"],
    },
    {
        "prompt": "What is the advantage of using Triton over writing CUDA kernels?",
        "answer": "easier to write with Python-like syntax and automatic optimization",
        "keywords": ["python", "easier", "automatic", "optimization", "high-level"],
    },
    {
        "prompt": "Complete this CUDA kernel signature: __global__ void vectorAdd(float* A, float* B, float* C, int N) { int idx = ___; }",
        "answer": "blockIdx.x * blockDim.x + threadIdx.x",
        "keywords": ["blockIdx", "blockDim", "threadIdx"],
    },
    {
        "prompt": "In Triton, what does BLOCK_SIZE typically represent?",
        "answer": "the number of elements processed by each program/block",
        "keywords": ["elements", "block", "program", "tile"],
    },
    {
        "prompt": "What is the purpose of __shared__ memory in CUDA?",
        "answer": "fast on-chip memory shared among threads in a block",
        "keywords": ["fast", "shared", "block", "on-chip", "cache"],
    },
    {
        "prompt": "Why do we need to check if idx < N in CUDA kernels?",
        "answer": "to prevent out-of-bounds access when grid size doesn't evenly divide N",
        "keywords": ["bounds", "check", "overflow", "size"],
    },
    {
        "prompt": "What is the purpose of tl.load and tl.store in Triton?",
        "answer": "load data from and store data to memory with automatic masking",
        "keywords": ["load", "store", "memory", "mask"],
    },
]


class GPUCodeEval(Task):
    """
    Evaluates model's understanding of GPU programming concepts.
    Tests knowledge of CUDA, Triton, and GPU performance optimization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.examples = GPU_CODE_EXAMPLES

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        """Get a single GPU code question."""
        example = self.examples[index]
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        conversation = {
            "messages": messages,
            "keywords": example["keywords"],
        }
        return conversation

    def evaluate(self, conversation, completion):
        """
        Evaluate if the completion demonstrates understanding of GPU concepts.
        Uses keyword matching and semantic similarity.
        """
        keywords = conversation["keywords"]
        completion_lower = completion.lower()
        
        # Count how many keywords are present in the completion
        keyword_matches = sum(1 for kw in keywords if kw.lower() in completion_lower)
        
        # Success if at least 50% of keywords are present
        # This is a simple heuristic - could be improved with better NLP
        threshold = len(keywords) * 0.5
        success = keyword_matches >= threshold
        
        return success


# Additional: Code generation evaluation for GPU kernels
GPU_CODE_GEN_EXAMPLES = [
    {
        "prompt": "Write a simple CUDA kernel to add two vectors A and B and store the result in C.",
        "test_keywords": ["__global__", "threadIdx", "blockIdx", "blockDim"],
    },
    {
        "prompt": "Write a Triton kernel to perform element-wise multiplication of two tensors.",
        "test_keywords": ["@triton.jit", "tl.load", "tl.store", "tl.program_id"],
    },
    {
        "prompt": "Write CUDA code to allocate device memory for an array of 1024 floats.",
        "test_keywords": ["cudaMalloc", "sizeof", "float"],
    },
    {
        "prompt": "Write a Triton kernel for matrix multiplication with tiling.",
        "test_keywords": ["BLOCK_SIZE", "tl.dot", "tl.load", "tl.store"],
    },
]


class GPUCodeGenEval(Task):
    """
    Evaluates model's ability to generate GPU code (CUDA/Triton).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.examples = GPU_CODE_GEN_EXAMPLES

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.examples)

    def get_example(self, index):
        """Get a single GPU code generation task."""
        example = self.examples[index]
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": ""},  # Model should generate
        ]
        conversation = {
            "messages": messages,
            "test_keywords": example["test_keywords"],
        }
        return conversation

    def evaluate(self, conversation, completion):
        """
        Evaluate if the generated code contains essential GPU programming elements.
        """
        test_keywords = conversation["test_keywords"]
        
        # Count how many required keywords are present
        keyword_matches = sum(1 for kw in test_keywords if kw in completion)
        
        # Success if all required keywords are present
        success = keyword_matches == len(test_keywords)
        
        return success
