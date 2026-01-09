"""
Hendrycks MATH (competition math) evaluation.
https://huggingface.co/datasets/hendrycks/competition_math
"""

import re
from fractions import Fraction

from tasks.common import Task, load_dataset


_DEFAULT_DATASET = "hendrycks/competition_math"
_ANSWER_PREFIX_RE = re.compile(r"^(answer|final answer)\s*[:\-]\s*", re.IGNORECASE)
_BOXED_PATTERN = re.compile(r"\\boxed\s*\{")
_FBOX_PATTERN = re.compile(r"\\fbox\s*\{")
_FINAL_TAG_RE = re.compile(r"<final>(.*?)</final>", re.DOTALL | re.IGNORECASE)
_ANSWER_INLINE_RE = re.compile(r"(answer|final answer|final)\s*(is|:)?\s*(.+)", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
_WRAP_COMMANDS = ("text", "mathrm", "mathbf", "mathbb", "mathcal", "mathit", "displaystyle")
_FRAC_COMMANDS = ("\\frac", "\\dfrac", "\\tfrac")


def _find_matching_brace(text, start_idx):
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _extract_braced(text, start_idx):
    if start_idx >= len(text) or text[start_idx] != "{":
        return None, None
    end_idx = _find_matching_brace(text, start_idx)
    if end_idx is None:
        return None, None
    return text[start_idx + 1:end_idx], end_idx + 1


def _extract_last_boxed(text):
    last = None
    for match in _BOXED_PATTERN.finditer(text):
        content, _ = _extract_braced(text, match.end() - 1)
        if content is not None:
            last = content
    for match in _FBOX_PATTERN.finditer(text):
        content, _ = _extract_braced(text, match.end() - 1)
        if content is not None:
            last = content
    return last


def _extract_last_number(text):
    if not text:
        return None
    matches = _NUMERIC_RE.findall(text)
    return matches[-1] if matches else None


def _unwrap_outer_commands(text):
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for cmd in _WRAP_COMMANDS:
            prefix = "\\" + cmd
            if not text.startswith(prefix):
                continue
            idx = len(prefix)
            while idx < len(text) and text[idx].isspace():
                idx += 1
            content, end_idx = _extract_braced(text, idx)
            if content is not None and end_idx == len(text):
                text = content.strip()
                changed = True
                break
    return text


def _replace_frac(text):
    i = 0
    out = []
    while i < len(text):
        matched = False
        for cmd in _FRAC_COMMANDS:
            if not text.startswith(cmd, i):
                continue
            j = i + len(cmd)
            while j < len(text) and text[j].isspace():
                j += 1
            num, next_j = _extract_braced(text, j) if j < len(text) and text[j] == "{" else (None, None)
            if num is None:
                continue
            k = next_j
            while k < len(text) and text[k].isspace():
                k += 1
            den, next_k = _extract_braced(text, k) if k < len(text) and text[k] == "{" else (None, None)
            if den is None:
                continue
            out.append(f"({num})/({den})")
            i = next_k
            matched = True
            break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def _extract_answer(text):
    if not text:
        return ""
    match = _FINAL_TAG_RE.search(text)
    if match:
        return match.group(1).strip()
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        return boxed
    if "####" in text:
        return text.split("####")[-1].strip()
    inline_match = None
    for match in _ANSWER_INLINE_RE.finditer(text):
        inline_match = match
    if inline_match:
        candidate = inline_match.group(3).strip()
        numeric = _extract_last_number(candidate)
        return numeric if numeric is not None else candidate
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.strip()
    last = lines[-1]
    return _ANSWER_PREFIX_RE.sub("", last).strip()


def _normalize_answer(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = _ANSWER_PREFIX_RE.sub("", text)
    boxed = _extract_last_boxed(text)
    if boxed is not None:
        text = boxed
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\\\", " ")
    text = text.replace("$", "")
    text = _unwrap_outer_commands(text)
    text = _replace_frac(text)
    text = text.strip().strip(".,;")
    text = re.sub(r"\s+", "", text)
    return text


def _as_number(text):
    if not text:
        return None
    if re.fullmatch(r"-?\d+/\d+", text):
        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        try:
            return float(text)
        except ValueError:
            return None
    return None


class MATH(Task):
    """
    Hendrycks MATH competition problems.
    subset: only "all" is supported for compatibility with callers.
    split: "train" or "test".
    """

    def __init__(self, subset: str = "all", split: str = "test", dataset_id: str = _DEFAULT_DATASET, **kwargs):
        super().__init__(**kwargs)
        assert subset == "all", "MATH only supports subset=all"
        assert split in ["train", "test"], "split must be train|test"
        self.ds = load_dataset(dataset_id, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row.get("problem") or row.get("question") or row.get("Problem")
        if problem is None:
            raise KeyError("Missing problem field for MATH row")
        answer = row.get("answer") or row.get("Answer")
        if answer is None:
            answer = _extract_last_boxed(row.get("solution", ""))
        if answer is None:
            raise KeyError("Missing answer field for MATH row")
        messages = [
            {"role": "user", "content": problem},
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
