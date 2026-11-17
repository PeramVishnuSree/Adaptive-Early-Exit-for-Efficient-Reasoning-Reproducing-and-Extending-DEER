"""State-dependent stopping threshold for AdaptiveDEER.

A fixed confidence cutoff stops on the first sharp spike, including a wrong
trial answer at the first checkpoint, and it waits through a repeated answer
that sits just under the cutoff. The threshold here moves with that state:

- the first checkpoint uses a higher bar, so one early spike is not enough;
- a repeated trial answer uses a lower bar;
- a very weak token inside the trial answer vetoes the exit;
- an unfinished trial answer (unbalanced brackets, an open environment, or a
  trailing backslash) is not an exit, however high its confidence is.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence


def answer_is_complete(answer) -> bool:
    """True when the trial answer can stand alone as a finished expression.

    High confidence on a cut-off formula is a common premature exit: the
    decoder is sure about the tokens it has produced so far, and those tokens
    are not an answer yet.
    """
    if answer is None:
        return False
    text = str(answer).strip()
    if not text:
        return False
    if text.count("{") != text.count("}"):
        return False
    if text.count("(") != text.count(")"):
        return False
    if text.endswith("\\"):
        return False
    if text.count("\\begin") != text.count("\\end"):
        return False
    return True


def _min_prob(checkpoint: dict) -> float:
    probs = checkpoint.get("token_probs") or []
    return float(min(probs)) if probs else 0.0


def _repeated(cps: Sequence[dict]) -> bool:
    if len(cps) < 2:
        return False
    current = cps[-1].get("trial_answer", "")
    previous = cps[-2].get("trial_answer", "")
    return current != "" and current == previous


def confidence_adaptive_policy(
    tau_first: float = 0.999,
    tau_base: float = 0.97,
    tau_stable: float = 0.90,
    min_prob_floor: float = 0.0,
    require_complete: bool = False,
) -> Callable:
    """Stop when confidence clears a threshold that depends on the trajectory.

    Compatible with ``deer_eval.simulate.simulate_policy``: ``stop(cps, rec)``.
    ``require_complete`` refuses an exit whose trial answer is unfinished.
    """

    def stop(cps: Sequence[dict], rec: Optional[dict] = None) -> bool:
        if not cps:
            return False
        current = cps[-1]
        if require_complete and not answer_is_complete(current.get("trial_answer")):
            return False
        if _min_prob(current) < min_prob_floor:
            return False
        confidence = float(current.get("confidence", 0.0))
        if len(cps) == 1:
            return confidence > tau_first
        if _repeated(cps):
            return confidence > tau_stable
        return confidence > tau_base

    return stop


def probability_adaptive_policy(
    probability,
    tau_first: float = 0.9,
    tau_base: float = 0.7,
    tau_stable: float = 0.55,
    min_prob_floor: float = 0.0,
    require_complete: bool = False,
) -> Callable:
    """Same state-dependent rule, applied to P(trial answer is correct).

    ``probability(cps, rec)`` returns the controller score for the current
    checkpoint. The caller usually closes over a precomputed lookup.
    """

    def stop(cps: Sequence[dict], rec: Optional[dict] = None) -> bool:
        if not cps:
            return False
        current = cps[-1]
        if require_complete and not answer_is_complete(current.get("trial_answer")):
            return False
        if _min_prob(current) < min_prob_floor:
            return False
        score = float(probability(cps, rec))
        if len(cps) == 1:
            return score > tau_first
        if _repeated(cps):
            return score > tau_stable
        return score > tau_base

    return stop
