"""Offline sequential stopping on a flat engineered feature table."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def simulate_sequential_stopping(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    name: str = "policy",
) -> Dict[str, float]:
    """First checkpoint where score_col >= threshold; else last checkpoint.

    Expects columns: problem_id, checkpoint_idx, reasoning_tokens_so_far,
    y_correct (or is_trial_correct), full_cot_tokens (optional).
    """
    label_col = "y_correct" if "y_correct" in df.columns else "is_trial_correct"
    rows = []
    for pid, g in df.sort_values("checkpoint_idx").groupby("problem_id", sort=False):
        g = g.reset_index(drop=True)
        stopped = None
        for i, r in g.iterrows():
            if float(r[score_col]) >= threshold:
                stopped = i
                break
        if stopped is None:
            stopped = len(g) - 1
        r = g.iloc[stopped]
        full = float(r.get("full_cot_tokens", g["reasoning_tokens_so_far"].max() * 1.2))
        used = float(r["reasoning_tokens_so_far"])
        rows.append(
            {
                "correct": int(r[label_col]),
                "tokens": used,
                "full": full,
                "premature": int(r[label_col] == 0),
            }
        )
    n = max(len(rows), 1)
    acc = 100.0 * sum(r["correct"] for r in rows) / n
    avg_tok = sum(r["tokens"] for r in rows) / n
    avg_full = sum(r["full"] for r in rows) / n
    save = 100.0 * (1.0 - avg_tok / max(avg_full, 1e-6))
    prem = 100.0 * sum(r["premature"] for r in rows) / n
    return {
        "Controller": name,
        "Accuracy (%)": round(acc, 2),
        "Net Compute Savings (%)": round(save, 2),
        "Premature Exit Error Rate (%)": round(prem, 2),
        "Avg Reasoning Tokens": round(avg_tok, 1),
    }
