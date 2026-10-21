import pytest

from deer_eval.io import load_benchmark
from deer_eval.scoring import extract_pred, gsm8k_gold_from_solution, is_correct, split_response


def test_split_response():
    assert split_response("think a</think>ans") == ("think a", "ans")
    assert split_response("no end") == ("no end", "")
    # last </think> wins
    assert split_response("a</think>b</think>c") == ("a</think>b", "c")


@pytest.mark.parametrize("text,expected", [
    ("so ... </think>\n\nThe answer is \\boxed{18}.", "18"),
    ("\\boxed{3} then later </think> final \\boxed{4}", "4"),
    ("</think>\\boxed{\\frac{3}{4}}", "\\frac{3}{4}"),
    ("</think>\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    # no box after </think>: fall back to the whole text, like DEER's check.py
    ("thinking \\boxed{7}</think>The answer is seven.", "7"),
    ("no answer at all", ""),
])
def test_extract_pred(text, expected):
    assert extract_pred(text) == expected


@pytest.mark.parametrize("pred,gold,ok", [
    ("18", "18", True),
    ("18.0", "18", True),
    ("\\$18", "18", True),
    ("1,000", "1000", True),
    ("\\frac{3}{4}", "0.75", True),
    ("\\dfrac{3}{4}", "\\frac{3}{4}", True),
    ("\\left( 3, \\frac{\\pi}{2} \\right)", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
    ("2\\sqrt{2}", "\\sqrt{8}", True),
    ("17", "18", False),
    ("", "18", False),
])
def test_is_correct(pred, gold, ok):
    assert is_correct(pred, gold) is ok


def test_gsm8k_gold_from_solution():
    assert gsm8k_gold_from_solution("blah\n#### 1,234") == "1234"
    assert gsm8k_gold_from_solution("no marker") is None


def test_math_gold_solutions_grade_as_correct():
    """The boxed answer in each reference MATH solution should match its gold
    answer. Catches grader/environment problems (e.g. a broken latex parser)."""
    rows = load_benchmark("math")[:60]
    bad = [r["answer"] for r in rows if not is_correct(extract_pred(r["solution"]), r["answer"])]
    assert len(bad) <= 1, bad
