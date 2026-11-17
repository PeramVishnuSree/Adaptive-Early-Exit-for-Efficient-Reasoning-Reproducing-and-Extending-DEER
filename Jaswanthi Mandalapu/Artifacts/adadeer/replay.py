"""Offline replay of stopping policies on logged never-exit trajectories.

This is the same protocol as the Check-In 3 simulator (Section V-D): a policy
stops at the first checkpoint it accepts and the early exit is scored on that
checkpoint's trial answer ("keep the trial answer"). Problems where the policy
never exits (or that have no checkpoints) get the never-exit run's own outcome.

Canonical input: JSONL, one problem per line
{
  "id": "gsm8k-17", "dataset": "gsm8k", "level": null,
  "full_correct": true, "full_output_tokens": 912, "full_reasoning_tokens": 801,
  "checkpoints": [
     {"reasoning_tokens": 233, "answer": "18", "confidence": 0.997,
      "token_probs": [0.41, 0.998, 0.996], "correct": true, "probe_tokens": 5},
     ...
  ]
}
Alternative key names used by common loggers are accepted (see _ALIASES).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional

from .policy import Policy, confidence_from_probs

_ALIASES = {
    "id": ["id", "idx", "problem_id", "question_id", "qid"],
    "dataset": ["dataset", "data", "benchmark"],
    "level": ["level", "difficulty"],
    "full_correct": ["full_correct", "final_correct", "correct", "is_correct"],
    "full_output_tokens": ["full_output_tokens", "output_tokens", "total_tokens", "length"],
    "full_reasoning_tokens": ["full_reasoning_tokens", "reasoning_tokens", "think_tokens"],
    "checkpoints": ["checkpoints", "trials", "trajectory", "steps"],
    # checkpoint-level
    "c_tokens": ["reasoning_tokens", "tokens_so_far", "think_tokens", "position", "L"],
    "c_answer": ["answer", "trial_answer", "pred", "parsed_answer"],
    "c_conf": ["confidence", "conf", "c"],
    "c_probs": ["token_probs", "probs", "probabilities"],
    "c_correct": ["correct", "is_correct", "label"],
    "c_probe": ["probe_tokens", "trial_tokens", "answer_tokens"],
}


def _get(d: dict, key: str, default=None):
    for k in _ALIASES[key]:
        if k in d and d[k] is not None:
            return d[k]
    return default


@dataclass
class Checkpoint:
    reasoning_tokens: float
    answer: Optional[str]
    confidence: Optional[float]
    token_probs: Optional[list]
    correct: bool
    probe_tokens: float


@dataclass
class Problem:
    id: str
    dataset: str
    level: Optional[int]
    full_correct: bool
    full_output_tokens: float
    full_reasoning_tokens: float
    checkpoints: list


def load_jsonl(paths: Iterable[str]) -> list[Problem]:
    out = []
    for path in paths:
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                cps = []
                for c in _get(d, "checkpoints", []) or []:
                    cps.append(Checkpoint(
                        reasoning_tokens=float(_get(c, "c_tokens", 0)),
                        answer=_get(c, "c_answer"),
                        confidence=_get(c, "c_conf"),
                        token_probs=_get(c, "c_probs"),
                        correct=bool(_get(c, "c_correct", False)),
                        probe_tokens=float(_get(c, "c_probe", 6)),
                    ))
                cps.sort(key=lambda c: c.reasoning_tokens)
                lvl = _get(d, "level")
                out.append(Problem(
                    id=str(_get(d, "id", f"{path}:{i}")),
                    dataset=str(_get(d, "dataset", "unknown")).lower(),
                    level=int(lvl) if lvl not in (None, "") else None,
                    full_correct=bool(_get(d, "full_correct", False)),
                    full_output_tokens=float(_get(d, "full_output_tokens", 0)),
                    full_reasoning_tokens=float(_get(d, "full_reasoning_tokens", 0)),
                    checkpoints=cps,
                ))
    return out


@dataclass
class Outcome:
    correct: bool
    output_tokens: float
    compute_tokens: float
    exited: bool
    exit_k: Optional[int]


@dataclass
class ReplayConfig:
    max_checkpoints: int = 10        # DEER default (max_judge_steps)
    think_budget: float = 4800       # 0.6 * 8000 in Check-In 3; 0.6*16384 for the paper
    conf_mode: str = "logged"        # "logged" | "avg1" | "gm" | "min"


def checkpoint_conf(c: Checkpoint, mode: str) -> float:
    if mode == "logged" or not c.token_probs:
        if c.confidence is None:
            return confidence_from_probs(c.token_probs or [], "avg1")
        return float(c.confidence)
    return confidence_from_probs(c.token_probs, mode)


def replay_one(p: Problem, policy: Policy, rc: ReplayConfig) -> Outcome:
    policy.reset(rc.think_budget)
    probe_cost = 0.0
    for k, c in enumerate(p.checkpoints[: rc.max_checkpoints], start=1):
        probe_cost += c.probe_tokens
        conf = checkpoint_conf(c, rc.conf_mode)
        if policy.should_exit(c.answer, conf, c.reasoning_tokens):
            out_tok = c.reasoning_tokens + c.probe_tokens   # kept trial answer
            return Outcome(c.correct, out_tok, out_tok + probe_cost - c.probe_tokens, True, k)
    return Outcome(p.full_correct, p.full_output_tokens, p.full_output_tokens + probe_cost, False, None)


def replay(problems: list[Problem], policy: Policy, rc: ReplayConfig) -> list[Outcome]:
    return [replay_one(p, policy, rc) for p in problems]


def summarize(outs: list[Outcome]) -> dict:
    n = len(outs)
    if n == 0:
        return {"n": 0, "acc": float("nan"), "tokens": float("nan"),
                "compute": float("nan"), "exits": 0, "exit_acc": float("nan")}
    ex = [o for o in outs if o.exited]
    return {
        "n": n,
        "acc": 100.0 * sum(o.correct for o in outs) / n,
        "tokens": sum(o.output_tokens for o in outs) / n,
        "compute": sum(o.compute_tokens for o in outs) / n,
        "exits": len(ex),
        "exit_acc": 100.0 * sum(o.correct for o in ex) / len(ex) if ex else float("nan"),
    }
