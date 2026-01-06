"""
Local JSON/JSONL chat dataset loader.
Expected format: each row has a "messages" list with {role, content}.
"""

import os

from tasks.common import Task, load_dataset


class JsonlChat(Task):
    """
    Load a local JSONL/JSON chat dataset for SFT.
    """

    def __init__(
        self,
        path,
        split="train",
        messages_field="messages",
        prompt_field=None,
        answer_field=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not path:
            raise ValueError("JsonlChat requires a path to a JSON/JSONL file.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL dataset not found: {path}")
        self.messages_field = messages_field
        self.prompt_field = prompt_field
        self.answer_field = answer_field
        data_files = {split: path}
        self.ds = load_dataset("json", data_files=data_files, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        if self.messages_field in row and row[self.messages_field] is not None:
            messages = row[self.messages_field]
            return {"messages": messages}
        if self.prompt_field and self.answer_field:
            prompt = row[self.prompt_field]
            answer = row[self.answer_field]
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            return {"messages": messages}
        raise ValueError("JsonlChat requires messages_field or prompt_field+answer_field.")
