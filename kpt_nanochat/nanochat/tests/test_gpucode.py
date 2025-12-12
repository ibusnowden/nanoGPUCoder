"""
Tests for GPU code evaluation tasks.
"""

import pytest
from tasks.gpucode import GPUCodeEval, GPUCodeGenEval


def test_gpucode_eval_initialization():
    """Test GPUCodeEval initializes correctly"""
    task = GPUCodeEval()
    assert len(task) == 15, "Should have 15 GPU concept questions"
    assert task.eval_type == 'generative'


def test_gpucode_eval_get_example():
    """Test getting examples from GPUCodeEval"""
    task = GPUCodeEval()
    example = task.get_example(0)
    
    assert 'messages' in example
    assert len(example['messages']) == 2
    assert example['messages'][0]['role'] == 'user'
    assert example['messages'][1]['role'] == 'assistant'
    assert 'keywords' in example
    assert isinstance(example['keywords'], list)


def test_gpucode_eval_evaluation():
    """Test GPUCodeEval evaluation logic"""
    task = GPUCodeEval()
    conversation = task.get_example(0)
    
    # Test with good completion (contains keywords)
    good_completion = "The purpose is to synchronize threads within a block as a barrier"
    result = task.evaluate(conversation, good_completion)
    assert result == True, "Should pass when keywords are present"
    
    # Test with bad completion (no keywords)
    bad_completion = "I don't know what that does"
    result = task.evaluate(conversation, bad_completion)
    assert result == False, "Should fail when keywords are missing"


def test_gpucodegen_eval_initialization():
    """Test GPUCodeGenEval initializes correctly"""
    task = GPUCodeGenEval()
    assert len(task) == 4, "Should have 4 GPU code generation tasks"
    assert task.eval_type == 'generative'


def test_gpucodegen_eval_get_example():
    """Test getting examples from GPUCodeGenEval"""
    task = GPUCodeGenEval()
    example = task.get_example(0)
    
    assert 'messages' in example
    assert 'test_keywords' in example
    assert isinstance(example['test_keywords'], list)


def test_gpucodegen_eval_evaluation():
    """Test GPUCodeGenEval evaluation logic"""
    task = GPUCodeGenEval()
    conversation = task.get_example(0)  # CUDA vector add
    
    # Test with complete code (all keywords present)
    good_code = """
    __global__ void vectorAdd(float* A, float* B, float* C, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            C[idx] = A[idx] + B[idx];
        }
    }
    """
    result = task.evaluate(conversation, good_code)
    assert result == True, "Should pass when all keywords are present"
    
    # Test with incomplete code (missing keywords)
    bad_code = "void vectorAdd() { return; }"
    result = task.evaluate(conversation, bad_code)
    assert result == False, "Should fail when keywords are missing"


def test_gpucode_eval_all_questions():
    """Test that all GPU concept questions are valid"""
    task = GPUCodeEval()
    for i in range(len(task)):
        example = task.get_example(i)
        assert 'messages' in example
        assert 'keywords' in example
        assert len(example['keywords']) > 0, f"Question {i} has no keywords"


def test_gpucodegen_eval_all_tasks():
    """Test that all GPU code generation tasks are valid"""
    task = GPUCodeGenEval()
    for i in range(len(task)):
        example = task.get_example(i)
        assert 'messages' in example
        assert 'test_keywords' in example
        assert len(example['test_keywords']) > 0, f"Task {i} has no test keywords"


def test_gpucode_keyword_matching():
    """Test keyword matching is case-insensitive"""
    task = GPUCodeEval()
    conversation = task.get_example(0)
    
    # Test case insensitivity
    completion_upper = "SYNCHRONIZE THREADS WITHIN A BLOCK"
    result = task.evaluate(conversation, completion_upper)
    assert result == True, "Should be case-insensitive"


def test_gpucodegen_keyword_matching():
    """Test keyword matching is case-sensitive for code"""
    task = GPUCodeGenEval()
    conversation = task.get_example(0)
    
    # Code keywords should be case-sensitive
    code_wrong_case = "__GLOBAL__ void kernel() { THREADIDX.x; }"
    result = task.evaluate(conversation, code_wrong_case)
    assert result == False, "Code keywords should be case-sensitive"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
