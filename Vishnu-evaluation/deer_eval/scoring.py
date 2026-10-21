"""Answer extraction and grading.

Uses DEER's own grader (vendored in deer_eval/grading) so our numbers are
comparable with the paper's. Both GSM8K and MATH answers are read from the
last \\boxed{...} in the response, which is what DEER's prompt asks for.
"""
import re

from .grading import check_is_correct, extract_answer

THINK_END = "</think>"


def split_response(text):
    """Split a response into (reasoning, answer) at the last </think>.

    If the model never closed its thinking (hit the length limit), the whole
    text counts as reasoning and the answer part is empty.
    """
    if THINK_END in text:
        head, _, tail = text.rpartition(THINK_END)
        return head, tail
    return text, ""


def extract_pred(text):
    """Final predicted answer: the last \\boxed{} in the response.

    Prefers the part after </think>; falls back to the whole text, matching
    DEER's check.py (which searches the full response).
    """
    _, answer_part = split_response(text)
    pred = extract_answer(answer_part) if answer_part else ""
    if not pred:
        pred = extract_answer(text)
    return pred


def gsm8k_gold_from_solution(solution):
    """Gold answer from a raw GSM8K solution ('... #### 1,234')."""
    m = re.search(r"####\s*(.+)$", solution.strip())
    if not m:
        return None
    return m.group(1).strip().replace(",", "")


def is_correct(pred, gold):
    if pred is None or pred == "":
        return False
    try:
        return bool(check_is_correct(str(pred), str(gold)))
    except Exception:
        return False
