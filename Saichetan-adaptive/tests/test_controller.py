from pathlib import Path

from adaptive_exit.controller import StoppingController
from adaptive_exit.features import FEATURE_COLS, engineer_trajectory_features
from adaptive_exit.synthetic import make_synthetic_trajectories


def test_fit_predict_save_load(tmp_path: Path):
    raw = make_synthetic_trajectories(n_problems=30, seed=1)
    feat, _ = engineer_trajectory_features(raw)
    X = feat[FEATURE_COLS].values
    y = feat["y_correct"].values
    ctrl = StoppingController(threshold=0.5)
    losses = ctrl.fit(X, y, epochs=3, batch_size=64)
    assert len(losses) == 3
    proba = ctrl.predict_proba(X[:5])
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()

    ckpt = tmp_path / "ctrl.pt"
    ctrl.save(ckpt)
    ctrl2 = StoppingController().load(ckpt)
    p2 = ctrl2.predict_proba(X[:5])
    assert abs(float(proba[0]) - float(p2[0])) < 1e-5


def test_should_stop_on_checkpoints():
    raw = make_synthetic_trajectories(n_problems=20, seed=2)
    feat, _ = engineer_trajectory_features(raw)
    ctrl = StoppingController(threshold=0.01)
    ctrl.fit(feat[FEATURE_COLS].values, feat["y_correct"].values, epochs=2, batch_size=64)
    cps = [
        {"thinking_tokens": 50, "confidence": 0.4, "trial_answer": "a", "token_probs": [0.4, 0.5]},
        {"thinking_tokens": 120, "confidence": 0.95, "trial_answer": "a", "token_probs": [0.9, 0.95]},
    ]
    assert isinstance(ctrl.should_stop(cps, {"idx": 0}), bool)
