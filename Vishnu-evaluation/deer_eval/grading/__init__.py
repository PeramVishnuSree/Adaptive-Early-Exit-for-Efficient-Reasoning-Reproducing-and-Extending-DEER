"""Answer extraction and grading, vendored from the official DEER repo (MIT).

Source: https://github.com/iie-ycx/DEER (utils/parser.py, grader.py,
math_normalization.py). Changes: import fixes only (see README).
"""
from .parser import extract_answer, strip_string  # noqa: F401
from .grader import check_is_correct, math_equal  # noqa: F401
