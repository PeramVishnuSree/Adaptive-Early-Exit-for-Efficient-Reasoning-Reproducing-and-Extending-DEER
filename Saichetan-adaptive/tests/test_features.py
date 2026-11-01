import pandas as pd

from adaptive_exit.features import FEATURE_COLS, engineer_trajectory_features, features_from_deer_checkpoints
from adaptive_exit.synthetic import make_synthetic_trajectories


def test_engineer_produces_12_features():
    raw = make_synthetic_trajectories(n_problems=8, seed=0)
    feat, cols = engineer_trajectory_features(raw)
    assert cols == FEATURE_COLS
    assert len(cols) == 12
    assert set(cols).issubset(feat.columns)
    assert feat["delta_c1"].notna().all()


def test_features_from_deer_checkpoints():
    cps = [
        {"thinking_tokens": 100, "confidence": 0.5, "trial_answer": "1", "token_probs": [0.6, 0.7]},
        {"thinking_tokens": 200, "confidence": 0.9, "trial_answer": "1", "token_probs": [0.9, 0.95], "trial_correct": True},
    ]
    raw = features_from_deer_checkpoints(cps, problem_id="p0")
    feat, cols = engineer_trajectory_features(raw)
    assert len(feat) == 2
    assert feat.iloc[1]["ans_match_prev1"] == 1.0
    assert feat.iloc[1]["ans_stability_streak"] >= 1
