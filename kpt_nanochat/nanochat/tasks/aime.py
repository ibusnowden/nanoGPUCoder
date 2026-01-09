"""
AIME evaluation task.
Default dataset: Maxwell-Jia/AIME_2024
"""

from tasks.common import Task, load_dataset
from tasks.math import _extract_answer, _normalize_answer, _as_number


_DEFAULT_DATASET = "Maxwell-Jia/AIME_2024"


class AIME(Task):
    """
    AIME problems (train split only in most HF datasets).
    """

    def __init__(self, split: str = "train", dataset_id: str = _DEFAULT_DATASET, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset(dataset_id, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row.get("Problem") or row.get("problem") or row.get("question") or row.get("prompt")
        if problem is None:
            raise KeyError("Missing problem field for AIME row")
        answer = row.get("Answer") or row.get("answer") or row.get("target")
        if answer is None:
            raise KeyError("Missing answer field for AIME row")
        messages = [
            {"role": "user", "content": str(problem)},
            {"role": "assistant", "content": str(answer)},
        ]
        return {"messages": messages, "target": str(answer)}

    def evaluate(self, conversation, assistant_response):
        pred = _normalize_answer(_extract_answer(assistant_response))
        gold = _normalize_answer(conversation.get("target") or conversation["messages"][-1]["content"])
        if pred == gold and pred != "":
            return True
        pred_num = _as_number(pred)
        gold_num = _as_number(gold)
        if pred_num is not None and gold_num is not None:
            return abs(pred_num - gold_num) <= 1e-6
        return False
