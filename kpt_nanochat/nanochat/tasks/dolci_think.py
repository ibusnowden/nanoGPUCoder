"""
Dolci-Think dataset loader for nanochat SFT.

Supports:
- Direct HuggingFace loading (allenai/Dolci-Think-SFT-7B)
- Local JSONL loading (preprocessed by prepare_dolci_think.py)
- Mode: cot (full reasoning) or answer_only (strip <think> blocks)
"""

import json
import os
import re

from datasets import load_dataset as hf_load_dataset

from tasks.common import Task, ddp_barrier, ddp_rank, load_dataset


_DEFAULT_DATASET = "allenai/Dolci-Think-SFT-7B"
_ROLE_KEYS = ("role", "from", "speaker", "author", "type")
_CONTENT_KEYS = ("content", "value", "text", "message", "response", "output", "completion")
_PROMPT_FIELDS = (
    "prompt",
    "question",
    "question_text",
    "instruction",
    "input",
    "inputs",
    "input_text",
    "problem",
    "problem_text",
    "query",
    "task",
    "statement",
    "prompt_text",
)
_RESPONSE_FIELDS = (
    "ground_truth",
    "ground_truths",
    "target",
    "expected",
    "gold",
    "reference",
    "label",
    "response",
    "completion",
    "answer",
    "answers",
    "final_answer",
    "output",
    "final",
    "solution",
    "assistant",
    "reply",
    "chosen",
)
_MESSAGES_FIELDS = ("messages", "conversation", "conversations", "dialogue", "chat")


def _extract_answer_from_think(text: str) -> str:
    """
    Remove <think>...</think> block and return only the final answer.
    """
    # Remove <think> blocks
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()

def _extract_final_answer(text: str) -> str:
    if text is None:
        return ""
    text = _extract_answer_from_think(text)
    for tag in ("Final Answer:", "Final:", "Answer:", "####"):
        idx = text.rfind(tag)
        if idx != -1:
            return text[idx + len(tag):].strip()
    return text.strip()

def _normalize_answer(text: str) -> str:
    return _extract_final_answer(text).strip()


def _normalize_role(role: str) -> str:
    """Normalize role names to user/assistant/system."""
    role = str(role).strip().lower()
    if role in {"system", "sys"}:
        return "system"
    if role in {"user", "human", "prompt", "question", "instruction", "input"}:
        return "user"
    if role in {"assistant", "model", "bot", "gpt", "response"}:
        return "assistant"
    return "user"  # fallback


def _normalize_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if "content" in item:
                    parts.append(str(item.get("content")))
                elif "text" in item:
                    parts.append(str(item.get("text")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "content" in value:
            return str(value.get("content"))
        if "text" in value:
            return str(value.get("text"))
    return str(value)

def _collapse_singleton(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value

def _pick_field(row, candidates):
    for name in candidates:
        if name in row:
            value = row.get(name)
            if value is not None:
                return value
    return None

def _normalize_messages(messages, mode):
    if not isinstance(messages, list):
        return []
    normalized = []
    fallback_role = "user"
    for msg in messages:
        if isinstance(msg, dict):
            role_value = None
            for key in _ROLE_KEYS:
                if key in msg:
                    role_value = msg.get(key)
                    if role_value is not None:
                        break
            role = _normalize_role(role_value) if role_value is not None else None
            content_value = None
            for key in _CONTENT_KEYS:
                if key in msg:
                    content_value = msg.get(key)
                    if content_value is not None:
                        break
            content = _normalize_text(content_value)
        else:
            role = None
            content = _normalize_text(msg)

        if not content:
            continue
        if role is None:
            role = fallback_role

        if mode == "answer_only" and role == "assistant":
            content = _extract_answer_from_think(content)
        if content:
            normalized.append({"role": role, "content": content})

        if role == "system":
            fallback_role = "user"
        else:
            fallback_role = "assistant" if role == "user" else "user"
    return normalized

def _messages_from_row(row, mode):
    messages = _pick_field(row, _MESSAGES_FIELDS)
    if messages:
        normalized = _normalize_messages(messages, mode)
        if normalized:
            return normalized

    prompt = _collapse_singleton(_pick_field(row, _PROMPT_FIELDS))
    response = _collapse_singleton(_pick_field(row, _RESPONSE_FIELDS))
    if prompt is None or response is None:
        return []

    prompt_text = _normalize_text(prompt)
    response_text = _normalize_text(response)
    if not prompt_text or not response_text:
        return []
    if mode == "answer_only":
        response_text = _extract_answer_from_think(response_text)
    return [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": response_text},
    ]


class DolciThink(Task):
    """
    Dolci-Think-SFT-7B dataset loader.

    Args:
        split: Dataset split (default: "train")
        dataset_id: HuggingFace dataset ID or None for local JSONL
        local_path: Path to local JSONL (from prepare_dolci_think.py)
        streaming: Stream from HF and cache only up to stop examples
        stream_cache_path: JSONL cache path for streamed subset
        mode: "cot" (keep <think> blocks) or "answer_only" (strip them)
        category_filter: List of categories to include, or None for all
        stop: Max examples to load

    Categories (from prepare_dolci_think.py):
        - reasoning_math, reasoning_science, reasoning_logic, reasoning_general
        - coding, chat, safety, short_reasoning
    """

    def __init__(
        self,
        split: str = "train",
        dataset_id: str = None,
        local_path: str = None,
        streaming: bool = False,
        stream_cache_path: str = None,
        mode: str = "cot",
        category_filter: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert mode in {"cot", "answer_only"}, "mode must be cot|answer_only"
        self.mode = mode
        self.category_filter = set(category_filter) if category_filter else None

        if local_path:
            # Load from local JSONL (preprocessed)
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"DolciThink JSONL not found: {local_path}")
            self.ds = load_dataset("json", data_files={split: local_path}, split=split)
        else:
            # Load from HuggingFace
            self.dataset_id = dataset_id or _DEFAULT_DATASET
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if streaming:
                cache_path = stream_cache_path or os.path.expanduser(
                    "~/.cache/nanochat/dolci_think_streamed.jsonl"
                )
                if ddp_rank() == 0 and not os.path.exists(cache_path):
                    self._stream_to_cache(cache_path, split, hf_token)
                ddp_barrier()
                self.ds = load_dataset("json", data_files={split: cache_path}, split=split)
            else:
                self.ds = load_dataset(self.dataset_id, split=split, token=hf_token)

        # Apply category filter if specified
        if self.category_filter and "category" in self.ds.column_names:
            self.ds = self.ds.filter(lambda x: x.get("category") in self.category_filter)

        # Drop rows that cannot be normalized into messages to prevent runtime failures.
        def _has_messages(row):
            messages = _messages_from_row(row, self.mode)
            return bool(messages)

        self.ds = self.ds.filter(_has_messages)

        # Shuffle
        self.ds = self.ds.shuffle(seed=42)

    def _stream_to_cache(self, cache_path, split, hf_token):
        stream_ds = hf_load_dataset(
            self.dataset_id,
            split=split,
            token=hf_token,
            streaming=True,
        )
        if self.category_filter:
            stream_ds = stream_ds.filter(
                lambda x: x.get("category") in self.category_filter
            )

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        count = 0
        with open(cache_path, "w") as f:
            for row in stream_ds:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                count += 1
                if self.stop is not None and count >= self.stop:
                    break
        if count == 0:
            raise ValueError("DolciThink streaming produced no rows.")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = _messages_from_row(row, self.mode)

        if not messages:
            raise ValueError("DolciThink row has no messages.")

        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        if not isinstance(assistant_response, str):
            return False
        messages = conversation.get("messages") if isinstance(conversation, dict) else None
        if not messages:
            return False
        ref_answer = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                ref_answer = msg.get("content")
                break
        if ref_answer is None:
            return False
        return _normalize_answer(assistant_response) == _normalize_answer(ref_answer)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))
