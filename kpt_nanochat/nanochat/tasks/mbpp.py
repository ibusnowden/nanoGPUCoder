"""
MBPP code generation evaluation.
"""

from nanochat.execution import execute_code
from tasks.common import Task, load_dataset
from tasks.humaneval import extract_program, extract_imports


class MBPP(Task):
    """
    MBPP (Mostly Basic Programming Problems) evaluation.
    Uses the public mbpp dataset test split.
    """

    def __init__(self, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("mbpp", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["text"] if "text" in row else row["prompt"]
        tests = row["test_list"] if "test_list" in row else []
        test_block = "\n".join(tests)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row.get("code", "")},
        ]
        return {
            "messages": messages,
            "tests": test_block,
            "entry_point": row.get("task_id", f"task_{index}"),
        }

    def evaluate(self, conversation, completion):
        imports = extract_imports(conversation["messages"][0]["content"])
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation["tests"]
            + "\n"
        )
        result = execute_code(program)
        return result.success

