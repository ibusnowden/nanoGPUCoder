"""
OpenThoughts dataset loader (OpenThoughts3-1.2M on HF).
Supports either reasoning traces (CoT) or answer-only formatting.
"""

import os
import re

from tasks.common import Task, load_dataset

_DEFAULT_DATASET = "open-thoughts/OpenThoughts3-1.2M"
_PROMPT_FIELDS = ("question", "prompt", "instruction", "input", "problem", "query")
_REASON_FIELDS = ("reasoning", "analysis", "cot", "chain_of_thought", "thoughts", "rationale")
_ANSWER_FIELDS = ("answer", "final", "output", "response", "completion")
_MESSAGES_FIELDS = ("messages", "conversation", "conversations")
_ROLE_KEYS = ("role", "from", "speaker", "author", "type")
_CONTENT_KEYS = ("content", "value", "text", "message", "response", "output", "completion")


def _pick_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _normalize_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def _extract_final_answer(text):
    if text is None:
        return ""
    # Try <final>...</final> tags (common in CoT datasets)
    match = re.search(r"<final>(.*?)</final>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try "Final Answer:" or similar labels
    for tag in ("Final Answer:", "Final:", "Answer:", "####"):
        idx = text.rfind(tag)
        if idx != -1:
            return text[idx + len(tag):].strip()
    return text.strip()


def _normalize_role(value):
    if value is None:
        return None
    role = str(value).strip().lower()
    if role in {"system", "sys"}:
        return "system"
    if role in {"user", "human", "prompt", "question", "instruction", "input"}:
        return "user"
    if role in {"assistant", "model", "bot", "gpt", "response"}:
        return "assistant"
    return None


class OpenThoughts(Task):
    """
    OpenThoughts dataset. Mode controls how the assistant message is formed:
    - mode="cot": include reasoning traces (preferred for RS-CoT)
    - mode="answer_only": keep only the final answer (preferred for Chat-SFT)
    """

    def __init__(
        self,
        split="train",
        dataset_id=None,
        mode="cot",
        prompt_field=None,
        reason_field=None,
        answer_field=None,
        response_field=None,
        messages_field=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert mode in {"cot", "answer_only"}, "mode must be cot|answer_only"
        self.mode = mode
        self.dataset_id = dataset_id or _DEFAULT_DATASET
        self.split = split
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.ds = load_dataset(self.dataset_id, split=split, token=hf_token).shuffle(seed=42)

        columns = set(self.ds.column_names)
        self.messages_field = messages_field or _pick_column(columns, _MESSAGES_FIELDS)
        self.prompt_field = prompt_field or _pick_column(columns, _PROMPT_FIELDS)
        self.reason_field = reason_field or _pick_column(columns, _REASON_FIELDS)
        self.answer_field = answer_field or _pick_column(columns, _ANSWER_FIELDS)
        self.response_field = response_field or _pick_column(columns, _ANSWER_FIELDS)

        if self.messages_field is None and self.prompt_field is None:
            raise ValueError(
                "OpenThoughts loader could not find a prompt field. "
                "Set prompt_field/messages_field explicitly."
            )

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def _from_messages(self, row):
        messages = row[self.messages_field]
        if not isinstance(messages, list):
            raise ValueError("OpenThoughts messages field must be a list.")
        normalized = []
        fallback_role = "user"
        for message in messages:
            if isinstance(message, dict):
                role_value = None
                for key in _ROLE_KEYS:
                    if key in message:
                        role_value = message.get(key)
                        if role_value is not None:
                            break
                role = _normalize_role(role_value)
                content_value = None
                for key in _CONTENT_KEYS:
                    if key in message:
                        content_value = message.get(key)
                        if content_value is not None:
                            break
                content = _normalize_text(content_value)
            else:
                role = None
                content = _normalize_text(message)

            if not content:
                continue
            if role is None:
                role = fallback_role
            normalized.append({"role": role, "content": content})

            if role == "system":
                fallback_role = "user"
            else:
                fallback_role = "assistant" if role == "user" else "user"

        if not normalized:
            raise ValueError("OpenThoughts messages field has no usable content.")

        if self.mode == "answer_only":
            last = normalized[-1]
            if last["role"] == "assistant":
                last["content"] = _extract_final_answer(last["content"])
                normalized[-1] = last
        return {"messages": normalized}

    def _from_fields(self, row):
        prompt = _normalize_text(row.get(self.prompt_field))
        if prompt is None:
            raise ValueError("OpenThoughts row missing prompt content.")

        reasoning = _normalize_text(row.get(self.reason_field)) if self.reason_field else None
        answer = _normalize_text(row.get(self.answer_field)) if self.answer_field else None
        response = _normalize_text(row.get(self.response_field)) if self.response_field else None

        if self.mode == "cot":
            if response:
                # Response field typically already has <think> tags (e.g., OpenThoughts3)
                assistant = response
            elif reasoning and answer:
                # Wrap reasoning in <think> tags if not already present
                if not reasoning.strip().startswith("<think>"):
                    reasoning = f"<think>{reasoning.strip()}</think>"
                assistant = f"{reasoning}\n{answer}"
            elif reasoning:
                if not reasoning.strip().startswith("<think>"):
                    reasoning = f"<think>{reasoning.strip()}</think>"
                assistant = reasoning
            else:
                assistant = answer or response or ""
        else:
            if answer:
                assistant = answer
            elif response:
                assistant = _extract_final_answer(response)
            elif reasoning:
                assistant = _extract_final_answer(reasoning)
            else:
                assistant = ""

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ]
        return {"messages": messages}

    def get_example(self, index):
        row = self.ds[index]
        if self.messages_field:
            return self._from_messages(row)
        return self._from_fields(row)
