from adaptive_exit.adaptive_threshold import (
    answer_is_complete,
    confidence_adaptive_policy,
)


def _cp(confidence, answer, probs=None):
    return {"confidence": confidence, "trial_answer": answer, "token_probs": probs or [0.9]}


def test_unfinished_formula_is_not_complete():
    assert answer_is_complete("78")
    assert answer_is_complete(r"\dfrac{3\sqrt{3}}{2}")
    assert not answer_is_complete(r"\dfrac{3\sqrt{3")
    assert not answer_is_complete(r"\begin{pmatrix} 1 \\")
    assert not answer_is_complete(r"80^{\circ")
    assert not answer_is_complete("")


def test_first_checkpoint_uses_the_higher_bar():
    policy = confidence_adaptive_policy(tau_first=0.995, tau_base=0.90, tau_stable=0.80)
    assert policy([_cp(0.99, "40")], {}) is False
    assert policy([_cp(0.997, "40")], {}) is True


def test_repeated_answer_uses_the_lower_bar():
    policy = confidence_adaptive_policy(tau_first=0.995, tau_base=0.97, tau_stable=0.85)
    trajectory = [_cp(0.94, "45"), _cp(0.88, "45")]
    assert policy(trajectory[:1], {}) is False
    assert policy(trajectory, {}) is True


def test_complete_gate_refuses_a_confident_fragment():
    policy = confidence_adaptive_policy(tau_first=0.90, require_complete=True)
    fragment = _cp(0.999, r"\dfrac{3\sqrt{3")
    assert policy([fragment], {}) is False
    assert policy([_cp(0.999, "78")], {}) is True
