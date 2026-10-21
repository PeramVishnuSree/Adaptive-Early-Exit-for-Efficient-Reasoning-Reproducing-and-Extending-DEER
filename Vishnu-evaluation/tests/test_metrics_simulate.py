import pytest

from deer_eval.metrics import annotate, group_summary, relative_to, summarize
from deer_eval.simulate import (checkpoint_table, score_checkpoints, simulate_policy,
                                stability_policy, sweep_thresholds, threshold_policy)


def rec(idx, answer, gold, think_tok, ans_tok, cps=(), exit_reason="natural"):
    return {
        "idx": idx, "question": f"q{idx}", "gold_answer": gold,
        "generated_responses": [f"reasoning</think>\\boxed{{{answer}}}"],
        "num_thinking_tokens": think_tok, "num_answer_tokens": ans_tok,
        "checkpoints": list(cps), "exit_reason": exit_reason,
    }


def cp(tokens, answer, conf, n=3):
    return {"thinking_tokens": tokens, "trial_answer": answer, "confidence": conf,
            "n_trial_tokens": n, "token_probs": [conf] * n}


@pytest.fixture
def never_exit_run():
    return score_checkpoints(annotate([
        # right early with high confidence
        rec(0, "5", "5", 1000, 50, [cp(200, "5", 0.97), cp(500, "5", 0.99)]),
        # wrong early (confident!), right at the end
        rec(1, "8", "8", 2000, 50, [cp(300, "7", 0.96), cp(900, "8", 0.80), cp(1500, "8", 0.93)]),
        # never confident, wrong
        rec(2, "1", "2", 3000, 50, [cp(400, "1", 0.30)]),
    ]))


def test_annotate_and_summarize(never_exit_run):
    s = summarize(never_exit_run, wall_time_s=30)
    assert s["n"] == 3
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["avg_reasoning_tokens"] == pytest.approx(2000)
    assert s["avg_output_tokens"] == pytest.approx(2050)
    # trial tokens: 6 + 9 + 3 = 18 over 3 questions
    assert s["avg_compute_tokens"] == pytest.approx(2050 + 6)
    assert s["wall_time_per_question_s"] == pytest.approx(10)


def test_annotate_needs_tokenizer_without_counts():
    r = rec(0, "5", "5", 1, 1)
    del r["num_thinking_tokens"]
    with pytest.raises(ValueError, match="tokenizer"):
        annotate([r])


def test_annotate_official_flags():
    r = rec(0, "5", "5", 1, 1)
    del r["exit_reason"]
    r["high_prob"] = 1
    assert annotate([r])[0]["exit_reason"] == "confident"


def test_annotate_rejects_misaligned_benchmark():
    with pytest.raises(ValueError, match="does not match"):
        annotate([rec(0, "5", "5", 1, 1)], benchmark_rows=[{"problem": "other", "answer": "5"}])


def test_relative_to():
    base = {"accuracy": 0.8, "avg_output_tokens": 1000, "avg_reasoning_tokens": 900, "avg_compute_tokens": 1000}
    run = {"accuracy": 0.82, "avg_output_tokens": 600, "avg_reasoning_tokens": 500, "avg_compute_tokens": 700}
    rel = relative_to(run, base)
    assert rel["token_reduction_ratio"] == pytest.approx(0.4)
    assert rel["accuracy_delta"] == pytest.approx(0.02)


def test_threshold_simulation(never_exit_run):
    res = simulate_policy(never_exit_run, threshold_policy(0.95))
    # q0 exits at cp0 (right), q1 exits at cp0 (wrong!), q2 never exits (wrong)
    assert [r["correct"] for r in res] == [True, False, False]
    assert [r["reasoning_tokens"] for r in res] == [200, 300, 3000]
    assert [r["exit_reason"] for r in res] == ["confident", "confident", "natural"]

    # tau above every confidence == full run
    full = simulate_policy(never_exit_run, threshold_policy(1.01))
    assert [r["reasoning_tokens"] for r in full] == [1000, 2000, 3000]


def test_stability_policy(never_exit_run):
    res = simulate_policy(never_exit_run, stability_policy(2))
    # q0 stable at cp1, q1 stable at cp2 ("8","8"), q2 has one checkpoint
    assert [r["reasoning_tokens"] for r in res] == [500, 1500, 3000]
    assert [r["correct"] for r in res] == [True, True, False]


def test_simulation_rejects_early_exit_runs():
    run = score_checkpoints(annotate([rec(0, "5", "5", 100, 5, [cp(100, "5", 0.99)], "confident")]))
    with pytest.raises(ValueError):
        simulate_policy(run, threshold_policy(0.9))


def test_sweep_monotone_tokens(never_exit_run):
    rows = sweep_thresholds(never_exit_run, taus=(0.5, 0.95, 0.999))
    toks = [r["avg_reasoning_tokens"] for r in rows]
    assert toks == sorted(toks)


def test_checkpoint_table(never_exit_run):
    rows = checkpoint_table(never_exit_run)
    assert len(rows) == 6
    q1 = [r for r in rows if r["idx"] == 1]
    assert q1[1]["delta_confidence"] == pytest.approx(0.80 - 0.96)
    assert q1[1]["answer_changed"] is True
    assert [r["label"] for r in q1] == [False, True, True]


def test_group_summary():
    run = annotate([rec(0, "5", "5", 10, 1), rec(1, "5", "6", 30, 1)])
    run[0]["level"], run[1]["level"] = 1, 5
    g = group_summary(run, "level")
    assert g[1]["accuracy"] == 1.0 and g[5]["accuracy"] == 0.0
