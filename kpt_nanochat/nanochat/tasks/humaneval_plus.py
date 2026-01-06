"""
HumanEval+ evaluation using the humanevalpack dataset.
"""

from nanochat.execution import execute_code
from tasks.common import Task, load_dataset
from tasks.humaneval import extract_imports, extract_program


class HumanEvalPlus(Task):
    """
    HumanEval+ (harder variants) from bigcode/humanevalpack (config: humanevalplus).
    """

    def __init__(self, split: str = "test", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("bigcode/humanevalpack", name="humanevalplus", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["prompt"]
        solution = row["canonical_solution"]
        entry_point = row["entry_point"]
        test = row["test"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point,
            "test": test,
        }
        return conversation

    def evaluate(self, conversation, completion):
        imports = extract_imports(conversation["messages"][0]["content"])
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation["test"]
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        return result.success

