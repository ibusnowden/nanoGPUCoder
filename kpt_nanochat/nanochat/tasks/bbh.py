"""
Big-Bench Hard (BBH) generative evaluation.
"""

import re
from typing import List

from datasets import concatenate_datasets
from tasks.common import Task, load_dataset


def _normalize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()


class BBH(Task):
    """
    BBH evaluation treated as generative string match against the provided target.
    subset: "all" (default) concatenates the common BBH tasks, or pass a single task
            name matching a dataset config in lukaemon/bbh.
    """

    DEFAULT_TASKS: List[str] = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]

    def __init__(self, subset: str = "all", split: str = "test", **kwargs):
        super().__init__(**kwargs)
        if subset == "all":
            parts = [load_dataset("lukaemon/bbh", name=name, split=split) for name in self.DEFAULT_TASKS]
            self.ds = concatenate_datasets(parts).shuffle(seed=42)
        else:
            self.ds = load_dataset("lukaemon/bbh", name=subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row["input"]
        target = row["target"]
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        return {"messages": messages, "target": target}

    def evaluate(self, conversation, completion):
        gt = _normalize(conversation["target"])
        pred = _normalize(completion)
        return gt == pred

