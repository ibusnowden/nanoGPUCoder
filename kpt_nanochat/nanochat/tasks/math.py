"""
Math multiple-choice evaluation using the AQuA-RAT dataset.
https://huggingface.co/datasets/aqua_rat
"""

import re

from tasks.common import Task, render_mc, extract_choice_letter, load_dataset


_OPTION_PREFIX_RE = re.compile(r"^[A-Za-z][\)\.]\s*")
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")


def _strip_option(choice):
    if not isinstance(choice, str):
        choice = str(choice)
    return _OPTION_PREFIX_RE.sub("", choice).strip()


def _normalize(text):
    if text is None:
        return ""
    return _NON_ALNUM_RE.sub("", str(text)).lower()


def _parse_options(raw_options):
    if isinstance(raw_options, (list, tuple)):
        return [_strip_option(opt) for opt in raw_options]
    if isinstance(raw_options, str):
        parts = re.split(r"[A-Ea-e][\)\.]\s*", raw_options)
        parts = [part.strip(" ,;") for part in parts if part.strip()]
        if parts:
            return [_strip_option(part) for part in parts]
        return [_strip_option(raw_options)]
    return []


def _pick_answer(raw_answer, choices, letters):
    if raw_answer is None:
        return None
    if isinstance(raw_answer, (int, float)):
        idx = int(raw_answer)
        if 0 <= idx < len(letters):
            return letters[idx]
        if 1 <= idx <= len(letters):
            return letters[idx - 1]
    answer_str = str(raw_answer).strip()
    if not answer_str:
        return None
    upper = answer_str.upper()
    if upper in letters:
        return upper
    if upper.isdigit():
        idx = int(upper)
        if 0 <= idx < len(letters):
            return letters[idx]
        if 1 <= idx <= len(letters):
            return letters[idx - 1]
    match = re.search(r"[A-Za-z]", answer_str)
    if match:
        letter = match.group(0).upper()
        if letter in letters:
            return letter
    norm_answer = _normalize(answer_str)
    for idx, choice in enumerate(choices):
        if _normalize(choice) == norm_answer:
            return letters[idx]
    return None


class MATH(Task):
    """
    AQuA-RAT multiple-choice math problems.
    subset: only "all" is supported for compatibility with callers.
    split: "train" or "test".
    """

    def __init__(self, subset: str = "all", split: str = "test", **kwargs):
        super().__init__(**kwargs)
        assert subset == "all", "AQuA-RAT only supports subset=all"
        assert split in ["train", "test"], "split must be train|test"
        self.ds = load_dataset("aqua_rat", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row.get("question") or row.get("Question") or row.get("problem") or row.get("Problem")
        if question is None:
            raise KeyError("Missing question field for AQuA-RAT row")
        raw_options = row.get("options") or row.get("choices") or row.get("Options")
        choices = _parse_options(raw_options)
        if not choices:
            raise ValueError("Missing options for AQuA-RAT row")
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        raw_answer = row.get("correct") or row.get("answer") or row.get("label") or row.get("Answer")
        answer = _pick_answer(raw_answer, choices, letters)
        if answer is None:
            raise ValueError("Missing or invalid answer for AQuA-RAT row")
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
        conversation = {
            "messages": messages,
            "letters": letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation["letters"], (
            f"AQuA answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        assistant_message = conversation["messages"][-1]["content"]
        return assistant_response == assistant_message

    def reward(self, conversation, assistant_response):
        choice = extract_choice_letter(assistant_response, conversation["letters"])
        if choice is None:
            return 0.0
        assistant_message = conversation["messages"][-1]["content"]
        return float(choice == assistant_message)
