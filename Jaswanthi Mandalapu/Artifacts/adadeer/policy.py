"""Stopping policies for early exit in reasoning models.

Two policies share one interface so they can be replayed offline on logged
never-exit trajectories *and* dropped into the live DEER generation loop:

  * FixedThresholdDEER  -- the original DEER rule: exit when c_k > tau.
  * AdaDEER             -- adaptive threshold tau_k computed per question and
                           per checkpoint from the trajectory seen so far.

Interface (per problem):
    policy.reset(think_budget)
    for each checkpoint k = 1, 2, ...:
        if policy.should_exit(answer=..., confidence=..., reasoning_tokens=...):
            stop and keep / finalize the answer
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence


# --------------------------------------------------------------------------
# Confidence from trial-answer token probabilities
# --------------------------------------------------------------------------
def confidence_from_probs(probs: Sequence[float], mode: str = "avg1") -> float:
    """DEER confidence of a trial answer from its max-token probabilities.

    avg1 : arithmetic mean from the 2nd token on (official code for R1-Distill)
    gm   : geometric mean over all tokens (Eq. 4 of the DEER paper)
    min  : minimum token probability
    """
    p = [float(x) for x in probs if x is not None]
    if not p:
        return 0.0
    if mode == "avg1":
        q = p[1:] if len(p) > 1 else p
        return sum(q) / len(q)
    if mode == "gm":
        return math.exp(sum(math.log(max(x, 1e-12)) for x in p) / len(p))
    if mode == "min":
        return min(p)
    raise ValueError(f"unknown confidence mode {mode!r}")


# --------------------------------------------------------------------------
# Answer normalisation (only used to detect *repeated* trial answers;
# correctness labels always come from the official grader)
# --------------------------------------------------------------------------
_STRIP = [r"\left", r"\right", r"\!", r"\,", r"\;", r"\:", r"\ ", "$", "^\\circ", "^{\\circ}"]


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    s = str(ans).strip()
    if not s:
        return None
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    for t in _STRIP:
        s = s.replace(t, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace(" ", "").rstrip(".").lower()
    s = re.sub(r"(units?|pounds?|dollars?|cents?|degrees?|meters?|cm|km|hours?|minutes?)$", "", s)
    # numeric canonical form: "18.0" == "18" == "18.00", "1,000" == "1000"
    num = s.replace(",", "")
    try:
        v = float(num)
        if math.isfinite(v):
            return repr(round(v, 9))
    except ValueError:
        pass
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m and int(m.group(2)) != 0:
        return repr(round(int(m.group(1)) / int(m.group(2)), 9))
    return s


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
class Policy:
    name = "policy"

    def reset(self, think_budget: Optional[float] = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def should_exit(self, answer, confidence: float, reasoning_tokens: float) -> bool:  # pragma: no cover
        raise NotImplementedError


@dataclass
class FixedThresholdDEER(Policy):
    """Original DEER: d_k = 1[c_k > tau]."""
    tau: float = 0.95

    @property
    def name(self) -> str:
        return f"DEER tau={self.tau:g}"

    def reset(self, think_budget=None):
        pass

    def should_exit(self, answer, confidence, reasoning_tokens):
        return confidence > self.tau


@dataclass
class AdaDEERConfig:
    """Hyper-parameters of the adaptive threshold.

    tau_k = clip( tau0
                  - alpha * min(s_k - 1, s_max)      # answer stability bonus
                  + gamma * v_k                       # confidence volatility penalty
                  + beta  * u_k                       # answer-switching (difficulty) penalty
                  - eta   * b_k ,                     # thinking-budget pressure
                  tau_min, tau_max )
    """
    tau0: float = 0.99      # threshold for an answer seen for the first time
    alpha: float = 0.06     # threshold drop per extra consecutive repeat
    s_max: int = 3          # cap on the stability bonus
    gamma: float = 0.5      # weight on confidence volatility
    vol_window: int = 3     # checkpoints used for volatility
    beta: float = 0.05      # weight on answer-switching rate
    eta: float = 0.10       # weight on budget pressure
    budget_start: float = 0.5   # fraction of the thinking budget where pressure starts
    tau_min: float = 0.85
    tau_max: float = 0.999


class AdaDEER(Policy):
    """Adaptive-threshold DEER (per-question, per-checkpoint threshold)."""

    def __init__(self, cfg: Optional[AdaDEERConfig] = None, **kw):
        self.cfg = cfg or AdaDEERConfig(**kw)
        self.reset()

    @property
    def name(self) -> str:
        c = self.cfg
        return (f"AdaDEER tau0={c.tau0:g} a={c.alpha:g} g={c.gamma:g} "
                f"b={c.beta:g} e={c.eta:g}")

    # -- state ---------------------------------------------------------------
    def reset(self, think_budget: Optional[float] = None) -> None:
        self.think_budget = think_budget
        self.k = 0
        self.prev_answer = None
        self.streak = 0
        self.switches = 0
        self.confs: list[float] = []
        self.last_tau = None

    # -- components ------------------------------------------------------------
    def _volatility(self) -> float:
        c = self.confs
        if len(c) < 2:
            return 0.0
        w = c[-(self.cfg.vol_window + 1):]
        diffs = [abs(b - a) for a, b in zip(w[:-1], w[1:])]
        return sum(diffs) / len(diffs)

    def _switch_rate(self) -> float:
        return self.switches / (self.k - 1) if self.k > 1 else 0.0

    def _budget_pressure(self, reasoning_tokens: float) -> float:
        if not self.think_budget:
            return 0.0
        f = reasoning_tokens / self.think_budget
        s = self.cfg.budget_start
        return min(1.0, max(0.0, (f - s) / (1.0 - s)))

    def threshold(self, reasoning_tokens: float) -> float:
        c = self.cfg
        tau = (c.tau0
               - c.alpha * min(self.streak - 1, c.s_max)
               + c.gamma * self._volatility()
               + c.beta * self._switch_rate()
               - c.eta * self._budget_pressure(reasoning_tokens))
        return min(c.tau_max, max(c.tau_min, tau))

    # -- decision --------------------------------------------------------------
    def should_exit(self, answer, confidence, reasoning_tokens):
        a = normalize_answer(answer)
        self.k += 1
        if a is not None and a == self.prev_answer:
            self.streak += 1
        else:
            if self.k > 1:
                self.switches += 1
            self.streak = 1
        self.prev_answer = a
        self.confs.append(float(confidence))
        self.last_tau = self.threshold(reasoning_tokens)
        # an unparseable trial answer (no \boxed content) is never accepted
        return a is not None and confidence > self.last_tau


class EvidenceDEER(Policy):
    """Sequential evidence accumulation on the log-odds scale.

    E_k = logit(c_k) + rho * E_{k-1}   if the trial answer repeats, else logit(c_k)
    exit when E_k > theta.   rho = 0 recovers DEER with tau = sigmoid(theta).
    """

    def __init__(self, theta: float = 4.6, rho: float = 0.5, eps: float = 1e-4):
        self.theta, self.rho, self.eps = theta, rho, eps
        self.reset()

    @property
    def name(self):
        return f"EvidenceDEER theta={self.theta:g} rho={self.rho:g}"

    def reset(self, think_budget=None):
        self.prev, self.E = None, 0.0

    def should_exit(self, answer, confidence, reasoning_tokens):
        a = normalize_answer(answer)
        c = min(max(float(confidence), self.eps), 1 - self.eps)
        z = math.log(c / (1 - c))
        self.E = z + self.rho * self.E if (a is not None and a == self.prev) else z
        self.prev = a
        return a is not None and self.E > self.theta


def make_policy(spec: dict) -> Policy:
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "deer":
        return FixedThresholdDEER(**spec)
    if kind == "adadeer":
        return AdaDEER(AdaDEERConfig(**spec))
    raise ValueError(kind)


__all__ = ["confidence_from_probs", "normalize_answer", "FixedThresholdDEER",
           "AdaDEER", "AdaDEERConfig", "EvidenceDEER", "make_policy", "asdict"]
