from adaptive_exit.features import FEATURE_COLS, engineer_trajectory_features
from adaptive_exit.simulate import simulate_sequential_stopping
from adaptive_exit.synthetic import make_synthetic_trajectories


def test_simulate_static_vs_oracle_shape():
    raw = make_synthetic_trajectories(n_problems=12, seed=3)
    feat, _ = engineer_trajectory_features(raw)
    static = simulate_sequential_stopping(feat, "trial_confidence", 0.8, "static")
    assert "Accuracy (%)" in static
    assert "Net Compute Savings (%)" in static
    # Score with gold label as an oracle upper bound on stop precision path
    oracle = simulate_sequential_stopping(feat, "y_correct", 0.5, "oracle")
    assert oracle["Premature Exit Error Rate (%)"] == 0.0
