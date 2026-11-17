"""Answer extraction and grading (strict last-\\boxed{} rule, as in DEER's evaluation)."""
from __future__ import annotations

from typing import Optional

from .policy import normalize_answer


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (balanced braces), or None."""
    i = text.rfind("\\boxed")
    while i != -1:
        j = i + len("\\boxed")
        while j < len(text) and text[j] == " ":
            j += 1
        if j < len(text) and text[j] == "{":
            depth, k = 0, j
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
                k += 1
        i = text.rfind("\\boxed", 0, i)
    return None


def brace_answer(text: str) -> tuple[Optional[str], bool]:
    """For a probe continuation that starts right after '\\boxed': (answer, closed)."""
    s = text.lstrip()
    if not s.startswith("{"):
        return None, False
    depth = 0
    for k, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[1:k].strip(), True
    return None, False


def is_correct(pred: Optional[str], gold: str) -> bool:
    if pred is None or pred == "":
        return False
    a, b = normalize_answer(pred), normalize_answer(gold)
    if a is not None and a == b:
        return True
    try:
        from math_verify import parse, verify
        g = parse(f"${gold}$")
        p = parse(f"${pred}$")
        return bool(g) and bool(p) and bool(verify(g, p))
    except Exception:
        return False
