"""Small synthetic trajectory corpus for local tests (mirrors notebook generator)."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def make_synthetic_trajectories(n_problems: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for i in range(n_problems):
        dataset = "GSM8K" if i % 2 == 0 else "MATH-500"
        n_cp = int(rng.integers(3, 8))
        correct_at = int(rng.integers(1, n_cp + 1))
        answer = f"ans_{i}"
        wrong = f"wrong_{i}"
        conf = 0.45
        for k in range(1, n_cp + 1):
            conf = float(np.clip(conf + rng.normal(0.08, 0.04), 0.35, 0.99))
            is_correct = int(k >= correct_at)
            trial = answer if is_correct else wrong
            rows.append(
                {
                    "model_scale": "1.5B",
                    "problem_id": f"syn_{i:04d}",
                    "dataset": dataset,
                    "difficulty_level": int(rng.integers(1, 4)),
                    "subject": "Synthetic",
                    "checkpoint_idx": k,
                    "total_checkpoints": n_cp,
                    "reasoning_tokens_so_far": int(80 * k + rng.integers(0, 40)),
                    "full_cot_tokens": int(80 * n_cp + 200),
                    "probe_overhead_tokens": 4,
                    "trial_confidence": conf,
                    "token_entropy": float(max(0.05, 1.2 - conf)),
                    "dispersion_min_prob": float(max(0.05, conf - 0.15)),
                    "trial_answer_str": trial,
                    "is_trial_correct": is_correct,
                    "full_cot_correct": 1,
                    "step_latency_ms": float(20 * k),
                }
            )
    return pd.DataFrame(rows)
