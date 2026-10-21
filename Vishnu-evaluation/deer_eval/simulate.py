"""Offline early-exit simulation from per-checkpoint logs.

Run DEER once with exiting disabled (threshold > 1, our logged script),
which records a trial answer + confidence at every checkpoint. Any stopping
policy can then be replayed here without touching the GPU: a fixed
threshold, an answer-stability rule, or a learned controller.

Caveat: when real DEER exits, it writes a fresh answer after </think>. Here
the exit answer is the checkpoint's *trial* answer, which is a close but not
exact proxy. Check agreement with a real DEER run at one threshold before
trusting a sweep (see scripts/compare_runs.py).
"""
import numpy as np

from .scoring import is_correct


def score_checkpoints(annotated):
    """Add 'trial_correct' to each checkpoint in place."""
    for r in annotated:
        for c in r.get("checkpoints") or []:
            c["trial_correct"] = is_correct(c.get("trial_answer", ""), r["gold_answer"])
    return annotated


def simulate_policy(annotated, should_stop):
    """Replay a stopping policy.

    should_stop(cps, record) -> bool is called with the checkpoints seen so
    far (cps[-1] is the current one). Records whose run exited early on its
    own ('confident') are rejected, since their later checkpoints are missing.
    Returns per-question results with the same keys metrics.summarize needs.
    """
    results = []
    for r in annotated:
        if r["exit_reason"] == "confident":
            raise ValueError("simulate_policy needs a never-exit run (threshold > 1).")
        cps = r.get("checkpoints") or []
        stopped = None
        for k in range(len(cps)):
            if should_stop(cps[: k + 1], r):
                stopped = k
                break
        if stopped is None:
            res = {
                "correct": r["correct"],
                "reasoning_tokens": r["reasoning_tokens"],
                "output_tokens": r["output_tokens"],
                "compute_tokens": r["compute_tokens"],
                "n_checkpoints": len(cps),
                "exit_reason": r["exit_reason"],
            }
        else:
            c = cps[stopped]
            trial = sum(x.get("n_trial_tokens", 0) for x in cps[: stopped + 1])
            res = {
                "correct": bool(c["trial_correct"]),
                "reasoning_tokens": c["thinking_tokens"],
                "output_tokens": c["thinking_tokens"] + c.get("n_trial_tokens", 0),
                "compute_tokens": c["thinking_tokens"] + trial,
                "n_checkpoints": stopped + 1,
                "exit_reason": "confident",
            }
        for key in ("idx", "level", "subject"):
            if key in r:
                res[key] = r[key]
        results.append(res)
    return results


def threshold_policy(tau):
    return lambda cps, rec: cps[-1]["confidence"] > tau


def stability_policy(m):
    """Stop when the trial answer has been the same for m checkpoints in a row."""
    def stop(cps, rec):
        if len(cps) < m:
            return False
        last = [c.get("trial_answer", "") for c in cps[-m:]]
        return last[0] != "" and all(a == last[0] for a in last)
    return stop


def sweep_thresholds(annotated, taus=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)):
    from .metrics import summarize
    rows = []
    for tau in taus:
        s = summarize(simulate_policy(annotated, threshold_policy(tau)))
        s["tau"] = float(tau)
        rows.append(s)
    return rows


def checkpoint_table(annotated):
    """Flat rows (one per checkpoint) for analysis or controller training.

    Features follow the Check-In 1 plan: confidence c_k, delta c_k, tokens
    L_k, position k. 'label' (trial answer correct) is the target only:
    it cannot be an input feature, since it is unknown at test time.
    """
    rows = []
    for r in annotated:
        prev = None
        for k, c in enumerate(r.get("checkpoints") or []):
            conf = c["confidence"]
            rows.append({
                "idx": r["idx"],
                "k": k,
                "confidence": conf,
                "delta_confidence": 0.0 if prev is None else conf - prev,
                "thinking_tokens": c["thinking_tokens"],
                "answer_changed": k > 0 and c.get("trial_answer") != r["checkpoints"][k - 1].get("trial_answer"),
                "min_token_prob": float(np.min(c["token_probs"])) if c.get("token_probs") else np.nan,
                "label": bool(c.get("trial_correct", False)),
                "final_correct": bool(r["correct"]),
                "level": r.get("level"),
            })
            prev = conf
    return rows
