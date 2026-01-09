"""
DAPO-Math-17k dataset loader for GRPO.
https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k
"""

from tasks.common import Task, load_dataset
from tasks.math import _extract_answer, _normalize_answer, _as_number


_DEFAULT_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"


def _normalize_role(role):
    role = str(role).strip().lower()
    if role in {"assistant", "model", "bot"}:
        return "assistant"
    if role in {"system", "sys"}:
        return "system"
    return "user"


class DAPOMath(Task):
    """
    DAPO-Math-17k problems with ground-truth answers in reward_model.

    The dataset provides a chat-style prompt and a ground_truth answer.
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
        prompt = row.get("prompt")
        messages = []
        if isinstance(prompt, list):
            for msg in prompt:
                if not isinstance(msg, dict):
                    continue
                role = _normalize_role(msg.get("role", "user"))
                content = msg.get("content") or msg.get("value") or msg.get("text")
                if content:
                    messages.append({"role": role, "content": str(content)})
        elif prompt is not None:
            messages = [{"role": "user", "content": str(prompt)}]

        if not messages:
            raise KeyError("Missing prompt for DAPO-Math row")

        reward_model = row.get("reward_model") or {}
        ground_truth = None
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth")
        if ground_truth is None:
            ground_truth = row.get("answer") or row.get("Answer") or row.get("target")
        if ground_truth is None:
            raise KeyError("Missing ground_truth for DAPO-Math row")

        messages.append({"role": "assistant", "content": str(ground_truth)})
        return {"messages": messages, "target": str(ground_truth)}

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
