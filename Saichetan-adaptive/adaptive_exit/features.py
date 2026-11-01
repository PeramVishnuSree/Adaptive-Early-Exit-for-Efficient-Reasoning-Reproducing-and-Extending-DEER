"""12-D causal trajectory features (Check-In 2 / Experiment2 notebook)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

FEATURE_COLS: List[str] = [
    "trial_confidence",
    "delta_c1",
    "delta_c2",
    "ema_conf",
    "conf_drawdown",
    "token_entropy",
    "dispersion_min_prob",
    "conf_entropy_ratio",
    "ans_match_prev1",
    "ans_stability_streak",
    "norm_token_length",
    "norm_checkpoint_pos",
]


def engineer_trajectory_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add the 12 feature columns used by AdaptiveDEER-MLP.

    Required columns: problem_id, checkpoint_idx, trial_confidence,
    token_entropy, dispersion_min_prob, trial_answer_str,
    reasoning_tokens_so_far, is_trial_correct (for labels).
    """
    need = {
        "problem_id",
        "checkpoint_idx",
        "trial_confidence",
        "token_entropy",
        "dispersion_min_prob",
        "trial_answer_str",
        "reasoning_tokens_so_far",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    df = df.sort_values(["problem_id", "checkpoint_idx"]).copy()
    g = df.groupby("problem_id", sort=False)

    df["conf_entropy_ratio"] = df["trial_confidence"] / (df["token_entropy"] + 0.10)

    prev_c1 = g["trial_confidence"].shift(1).fillna(df["trial_confidence"] * 0.75)
    prev_c2 = g["trial_confidence"].shift(2).fillna(prev_c1)
    df["delta_c1"] = df["trial_confidence"] - prev_c1
    df["delta_c2"] = df["delta_c1"] - (prev_c1 - prev_c2)
    df["ema_conf"] = g["trial_confidence"].transform(lambda s: s.ewm(alpha=0.55).mean())
    df["max_conf_so_far"] = g["trial_confidence"].cummax()
    df["conf_drawdown"] = df["max_conf_so_far"] - df["trial_confidence"]

    prev_ans1 = g["trial_answer_str"].shift(1).fillna("__NONE__")
    df["ans_match_prev1"] = (df["trial_answer_str"] == prev_ans1).astype(float)

    streaks: List[int] = []
    for _, p_df in g:
        cur_streak = 0
        prev_a = None
        for a in p_df["trial_answer_str"].tolist():
            cur_streak = (cur_streak + 1) if (a == prev_a) else 0
            streaks.append(cur_streak)
            prev_a = a
    df["ans_stability_streak"] = streaks

    df["norm_token_length"] = np.clip(df["reasoning_tokens_so_far"] / 1500.0, 0.0, 1.5)
    df["norm_checkpoint_pos"] = np.clip(df["checkpoint_idx"] / 15.0, 0.0, 1.5)

    if "is_trial_correct" in df.columns:
        df["y_correct"] = df["is_trial_correct"].astype(int)

    return df, list(FEATURE_COLS)


def features_from_deer_checkpoints(checkpoints: list, problem_id: str = "q0") -> pd.DataFrame:
    """Convert Vishnu-style checkpoint dicts into a raw frame for engineer_*.

    Expected keys per checkpoint: thinking_tokens, confidence, trial_answer,
    optional token_probs / token_entropy / dispersion_min_prob / trial_correct.
    """
    rows = []
    for k, c in enumerate(checkpoints):
        probs = c.get("token_probs") or []
        entropy = c.get("token_entropy")
        if entropy is None and probs:
            p = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
            entropy = float(-(p * np.log(p)).sum())
        if entropy is None:
            entropy = 0.5
        disp = c.get("dispersion_min_prob")
        if disp is None:
            disp = float(min(probs)) if probs else 0.5
        rows.append(
            {
                "problem_id": problem_id,
                "checkpoint_idx": k + 1,
                "trial_confidence": float(c.get("confidence", c.get("pred_prob", 0.0))),
                "token_entropy": float(entropy),
                "dispersion_min_prob": float(disp),
                "trial_answer_str": str(c.get("trial_answer", "")),
                "reasoning_tokens_so_far": int(c.get("thinking_tokens", 0)),
                "is_trial_correct": int(bool(c.get("trial_correct", False))),
            }
        )
    return pd.DataFrame(rows)
