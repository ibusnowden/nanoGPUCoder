"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import fcntl
import hashlib
import os
import random
import re
import time

import torch.distributed as dist
from datasets import load_dataset as _hf_load_dataset


# -----------------------------------------------------------------------------
# DDP-safe dataset loading utilities (Fix for HF Hub 429 + NFS race conditions)
# -----------------------------------------------------------------------------

def ddp_rank():
    """Get the current DDP rank, or 0 if not in distributed mode."""
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def ddp_world_size():
    """Get the DDP world size, or 1 if not in distributed mode."""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def ddp_barrier():
    """Synchronize all DDP ranks."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _get_lock_path(args, kwargs):
    """Generate a unique lock file path for a dataset load call."""
    # Create a hash of the arguments to identify this specific dataset
    key = str(args) + str(sorted(kwargs.items()))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    lock_dir = os.environ.get("HF_DATASETS_CACHE", "/tmp")
    return os.path.join(lock_dir, f".dataset_load_{h}.lock")


def load_dataset(*args, **kwargs):
    """
    DDP-safe wrapper around datasets.load_dataset.

    Only rank 0 loads/prepares the dataset. Other ranks load from the cached
    arrow files directly to avoid NFS race conditions.
    """
    from datasets import Dataset
    import glob as glob_module

    r = ddp_rank()
    ws = ddp_world_size()

    # Single process or non-distributed: just load normally
    if ws == 1:
        return _hf_load_dataset(*args, **kwargs)

    ds = None
    cache_path = None

    # Rank 0 loads first (downloads/prepares if needed)
    if r == 0:
        ds = _hf_load_dataset(*args, **kwargs)
        # Get the cache path from the loaded dataset
        if hasattr(ds, '_data_files') and ds._data_files:
            cache_path = str(ds._data_files[0])
        elif hasattr(ds, 'cache_files') and ds.cache_files:
            cache_path = ds.cache_files[0].get('filename', '')
        # Force Python to release any file handles
        import gc
        gc.collect()
        # Wait for NFS to sync
        time.sleep(2.0)

    # Barrier: ensure rank 0 is completely done before others start
    ddp_barrier()

    if r == 0:
        return ds

    # Non-rank-0: small staggered delay
    time.sleep(0.5 * r)

    # Try to load from cache - use retries for NFS issues
    for attempt in range(10):
        try:
            return _hf_load_dataset(*args, **kwargs)
        except OSError as e:
            err_str = str(e).lower()
            if attempt < 9 and ("busy" in err_str or "not empty" in err_str or "resource" in err_str):
                time.sleep(2.0 + attempt)
                continue
            raise


# -----------------------------------------------------------------------------


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError

    def reward(self, conversation, completion):
        """
        Default RL reward: cast evaluate() to float.
        Tasks can override for more permissive parsing.
        """
        return float(self.evaluate(conversation, completion))


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a tax mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        conversation = self.tasks[task_idx][local_idx]
        if isinstance(conversation, dict):
            conversation = dict(conversation)
            conversation["_task_idx"] = task_idx
            conversation["_task_name"] = self.tasks[task_idx].__class__.__name__
        return conversation


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


def extract_choice_letter(text, letters):
    """
    Extract the first matching choice letter from free-form output.
    Returns None if no valid letter is found.
    """
    if not isinstance(text, str):
        return None
    text = text.strip().upper()
    if text in letters:
        return text
    for letter in letters:
        if re.search(rf"\b{re.escape(letter)}\b", text):
            return letter
    return None


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="auxiliary_train", split="train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
