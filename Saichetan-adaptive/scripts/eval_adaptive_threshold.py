#!/usr/bin/env python3
"""Held-out test of a state-dependent AdaptiveDEER threshold.

The 1,000/500 training trajectories used by compare_stopping_methods.py are
not in this checkout, so this uses the checkpoint-3 never-exit logs (first
100 GSM8K and 100 MATH-500). Each seed holds out 30% of questions. The
threshold is chosen only on the other 70%, then frozen. Early exits are
graded on the trial answer, with DEER's cap of 10 checks, matching
Vishnu's offline simulator.

Two selection rules, both applied on the training questions only:

- equal cost: highest training accuracy whose output tokens do not exceed
  DEER at tau=0.95;
- gap closing: highest training accuracy, then fewer tokens. This is allowed
  to spend more tokens than DEER when that buys accuracy toward the
  perfect-stopping bound.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VISHNU = ROOT.parent / "Vishnu-evaluation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VISHNU))

from adaptive_exit.adaptive_threshold import (  # noqa: E402
    answer_is_complete,
    confidence_adaptive_policy,
    probability_adaptive_policy,
)
from adaptive_exit.controller import StoppingController  # noqa: E402
from adaptive_exit.features import (  # noqa: E402
    FEATURE_COLS,
    engineer_trajectory_features,
    features_from_deer_checkpoints,
)
from deer_eval.io import read_jsonl  # noqa: E402
from deer_eval.metrics import summarize  # noqa: E402
from deer_eval.simulate import simulate_policy, stability_policy  # noqa: E402

DATA = {
    "gsm8k": VISHNU / "results/checkpoint3/never_exit/never_gsm8k.scored.jsonl",
    "math": VISHNU / "results/checkpoint3/never_exit/never_math.scored.jsonl",
}
MAX_CHECKS = 10
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
TEST_FRACTION = 0.30
MLP_EPOCHS = 12

# Small grid. The first-checkpoint bar is the lever that refuses a single spike.
CONF_GRID = [
    {"tau_first": tf, "tau_base": tb, "tau_stable": ts, "min_prob_floor": mf}
    for tf in (0.999, 1.1)
    for tb in (0.95, 0.97, 0.99)
    for ts in (0.85, 0.90, 0.94)
    for mf in (0.0, 0.50)
]
MLP_GLOBAL = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MLP_ADAPTIVE = [
    {"tau_first": tf, "tau_base": tb, "tau_stable": ts, "min_prob_floor": 0.0}
    for tf in (0.80, 0.92)
    for tb in (0.60, 0.75)
    for ts in (0.45, 0.60)
]
# Closing family: unfinished answers never exit. The first checkpoint and a
# one-off answer keep a high bar; a repeated answer may exit lower.
CLOSING_GRID = [
    {
        "tau_first": tf,
        "tau_base": tb,
        "tau_stable": ts,
        "min_prob_floor": 0.0,
        "require_complete": True,
    }
    for tf in (0.995, 1.1)
    for tb in (0.95, 0.97, 1.1)
    for ts in (0.85, 0.88, 0.93)
]


def capped(policy):
    return lambda cps, rec: len(cps) <= MAX_CHECKS and policy(cps, rec)


def deer(tau, require_complete=False):
    def stop(cps, rec):
        if require_complete and not answer_is_complete(cps[-1].get("trial_answer")):
            return False
        return cps[-1]["confidence"] > tau

    return stop


def run(recs, policy):
    return simulate_policy(recs, capped(policy))


def metrics(results):
    summary = summarize(results)
    return {
        "accuracy": summary["accuracy"],
        "output_tokens": summary["avg_output_tokens"],
        "reasoning_tokens": summary["avg_reasoning_tokens"],
    }


def paired_diff(candidate, reference):
    acc = np.mean([c["correct"] for c in candidate]) - np.mean([r["correct"] for r in reference])
    tok = np.mean([c["output_tokens"] for c in candidate]) - np.mean([r["output_tokens"] for r in reference])
    return float(acc), float(tok)


def split_questions(recs, seed):
    """70/30 by question, stratified on whether a checkpoint exists."""
    rng = np.random.default_rng(seed)
    with_cp = [r for r in recs if r.get("checkpoints")]
    without = [r for r in recs if not r.get("checkpoints")]
    train, test = [], []
    for group in (with_cp, without):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n_test = max(1, int(round(len(group) * TEST_FRACTION))) if group else 0
        test_ids = set(idx[:n_test])
        for i, rec in enumerate(group):
            (test if i in test_ids else train).append(rec)
    return train, test


def feature_frame(recs):
    frames = []
    for rec in recs:
        if rec.get("checkpoints"):
            frames.append(features_from_deer_checkpoints(rec["checkpoints"], problem_id=str(rec["idx"])))
    if not frames:
        return None
    import pandas as pd

    feats, _ = engineer_trajectory_features(pd.concat(frames, ignore_index=True))
    return feats


def fit_mlp(train_recs, seed):
    feats = feature_frame(train_recs)
    if feats is None or feats["y_correct"].nunique() < 2:
        return None
    rng_state = np.random.seed(seed)
    del rng_state
    import torch

    torch.manual_seed(seed)
    controller = StoppingController(device="cpu")
    controller.fit(feats[FEATURE_COLS].values, feats["y_correct"].values, epochs=MLP_EPOCHS)
    return controller


def probability_lookup(controller, recs):
    """Map (question idx, 1-based checkpoint count) -> P(correct)."""
    lookup = {}
    if controller is None:
        return lookup
    for rec in recs:
        cps = rec.get("checkpoints") or []
        if not cps:
            continue
        pid = str(rec["idx"])
        raw = features_from_deer_checkpoints(cps, problem_id=pid)
        feats, cols = engineer_trajectory_features(raw)
        probs = controller.predict_proba(feats[cols].values)
        for k, p in enumerate(probs, start=1):
            lookup[(rec["idx"], k)] = float(p)
    return lookup


def score_fn(lookup):
    def probability(cps, rec):
        return lookup.get((rec["idx"], len(cps)), 0.0)

    return probability


def choose(candidates, budget):
    """candidates: list of (name, params, results). Prefer accuracy, then fewer tokens."""
    eligible = [c for c in candidates if metrics(c[2])["output_tokens"] <= budget + 1e-6]
    pool = eligible or candidates
    pool.sort(key=lambda c: (metrics(c[2])["accuracy"], -metrics(c[2])["output_tokens"]), reverse=True)
    return pool[0]


def evaluate_family(name, param_results, train_deer_tokens, train_full_tokens):
    equal = choose(param_results, train_deer_tokens)
    closing = choose(param_results, train_full_tokens)
    return {"equal_cost": equal, "gap_close": closing, "family": name}


def conf_candidates(recs):
    out = []
    for params in CONF_GRID:
        out.append((json.dumps(params), params, run(recs, confidence_adaptive_policy(**params))))
    return out


def closing_candidates(recs):
    out = []
    for params in CLOSING_GRID:
        out.append((json.dumps(params), params, run(recs, confidence_adaptive_policy(**params))))
    return out


def mlp_candidates(recs, lookup):
    prob = score_fn(lookup)
    out = []
    for tau in MLP_GLOBAL:
        params = {"tau": tau}
        policy = lambda cps, rec, tau=tau, prob=prob: prob(cps, rec) > tau
        out.append((f"global {tau}", params, run(recs, policy)))
    for params in MLP_ADAPTIVE:
        out.append(
            (
                json.dumps(params),
                params,
                run(recs, probability_adaptive_policy(prob, **params)),
            )
        )
    return out


def summarize_seed(seed_rows):
    acc = np.array([r["accuracy"] for r in seed_rows])
    tok = np.array([r["output_tokens"] for r in seed_rows])
    dacc = np.array([r["delta_acc"] for r in seed_rows])
    dtok = np.array([r["delta_tok"] for r in seed_rows])
    return {
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "tokens_mean": float(tok.mean()),
        "tokens_std": float(tok.std(ddof=1)),
        "delta_acc_mean": float(dacc.mean()),
        "delta_acc_std": float(dacc.std(ddof=1)),
        "delta_tok_mean": float(dtok.mean()),
        "delta_tok_std": float(dtok.std(ddof=1)),
        "seeds_better_acc": int((dacc > 0).sum()),
        "n_seeds": len(seed_rows),
        "params": dict(Counter(json.dumps(r["params"], sort_keys=True) for r in seed_rows if "params" in r)),
    }


def one_dataset(name, recs):
    families = (
        "DEER τ=0.95",
        "DEER τ=0.95 + complete",
        "Consistency k=2",
        "Confidence adaptive",
        "Closing adaptive",
        "MLP global",
        "MLP adaptive",
        "Perfect stopping",
    )
    buckets = {fam: {"equal_cost": [], "gap_close": []} for fam in families}
    for seed in SEEDS:
        train, test = split_questions(recs, seed)
        deer_train = run(train, deer(0.95))
        full_train = run(train, lambda cps, rec: False)
        deer_test = run(test, deer(0.95))
        deer_m = metrics(deer_test)
        budget = metrics(deer_train)["output_tokens"]
        full_budget = metrics(full_train)["output_tokens"]

        fixed = {
            "DEER τ=0.95": deer_test,
            "DEER τ=0.95 + complete": run(test, deer(0.95, require_complete=True)),
            "Consistency k=2": run(test, stability_policy(2)),
            "Perfect stopping": run(test, lambda cps, rec: bool(cps[-1].get("trial_correct"))),
        }
        for fam, res in fixed.items():
            dacc, dtok = paired_diff(res, deer_test)
            row = {**metrics(res), "delta_acc": dacc, "delta_tok": dtok, "seed": seed}
            buckets[fam]["equal_cost"].append(row)
            buckets[fam]["gap_close"].append(row)

        controller = fit_mlp(train, seed)
        lookup_train = probability_lookup(controller, train)
        lookup_test = probability_lookup(controller, test)

        train_conf = conf_candidates(train)
        train_closing = closing_candidates(train)
        conf_choice = evaluate_family("conf", train_conf, budget, full_budget)
        closing_choice = evaluate_family("closing", train_closing, budget, full_budget)
        mlp_train = mlp_candidates(train, lookup_train)
        # Split MLP candidates into global vs adaptive by the stored params.
        mlp_global_train = [c for c in mlp_train if "tau" in c[1]]
        mlp_adapt_train = [c for c in mlp_train if "tau_base" in c[1]]
        mlp_global_choice = evaluate_family("mlp", mlp_global_train, budget, full_budget)
        mlp_adapt_choice = evaluate_family("mlp-adapt", mlp_adapt_train, budget, full_budget)

        def apply_conf(params):
            return run(test, confidence_adaptive_policy(**params))

        def apply_mlp_global(params):
            prob = score_fn(lookup_test)
            return run(test, lambda cps, rec, tau=params["tau"], prob=prob: prob(cps, rec) > tau)

        def apply_mlp_adapt(params):
            return run(test, probability_adaptive_policy(score_fn(lookup_test), **params))

        chosen = {
            "Confidence adaptive": (conf_choice, apply_conf),
            "Closing adaptive": (closing_choice, apply_conf),
            "MLP global": (mlp_global_choice, apply_mlp_global),
            "MLP adaptive": (mlp_adapt_choice, apply_mlp_adapt),
        }
        for fam, (choice, apply) in chosen.items():
            for rule in ("equal_cost", "gap_close"):
                _label, params, _train_res = choice[rule]
                res = apply(params)
                dacc, dtok = paired_diff(res, deer_test)
                buckets[fam][rule].append(
                    {
                        **metrics(res),
                        "delta_acc": dacc,
                        "delta_tok": dtok,
                        "seed": seed,
                        "params": params,
                        "deer_accuracy": deer_m["accuracy"],
                    }
                )
        print(f"  {name} seed {seed}: test n={len(test)} deer={deer_m['accuracy']:.3f}", flush=True)
    return {fam: {rule: summarize_seed(rows) for rule, rows in rules.items()} for fam, rules in buckets.items()}


def fmt(summary):
    return (
        f"{100 * summary['accuracy_mean']:.1f} ± {100 * summary['accuracy_std']:.1f}",
        f"{summary['tokens_mean']:.0f} ± {summary['tokens_std']:.0f}",
        f"{100 * summary['delta_acc_mean']:+.1f} ± {100 * summary['delta_acc_std']:.1f}",
        f"{summary['delta_tok_mean']:+.0f} ± {summary['delta_tok_std']:.0f}",
        f"{summary['seeds_better_acc']}/{summary['n_seeds']}",
    )


def main():
    out_dir = ROOT / "results" / "adaptive_threshold_heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    lines = [
        "# Adaptive threshold, held-out questions",
        "",
        "Checkpoint-3 never-exit logs (100 GSM8K, 100 MATH-500). These are not",
        "the 1,000/500 training trajectories used by compare_stopping_methods.py.",
        "Eight seeds, 30% of questions held out. Numeric thresholds are chosen",
        "on the complementary 70% only. The policy family itself (a higher bar",
        "on the first checkpoint, a lower bar once an answer repeats, and a",
        "refusal to exit on an unfinished formula) was chosen after reading",
        "DEER's misses on these same logs, so the held-out gap is slightly",
        "optimistic. Δ is versus DEER τ=0.95 on the same held-out questions.",
        "Mean ± standard deviation across seeds.",
        "",
    ]
    for name, path in DATA.items():
        print(f"dataset {name}", flush=True)
        recs = read_jsonl(path)
        report[name] = one_dataset(name, recs)
        for rule, title in (
            ("equal_cost", "Equal token cost (train tokens ≤ DEER τ=0.95)"),
            ("gap_close", "Gap closing (highest train accuracy, then fewer tokens)"),
        ):
            lines += [
                f"## {name}: {title}",
                "",
                "| Method | Accuracy (%) | Output tokens | Δ acc vs DEER (points) | Δ tokens | Seeds with higher acc |",
                "|---|---|---|---|---|---|",
            ]
            for fam, rules in report[name].items():
                acc, tok, dacc, dtok, wins = fmt(rules[rule])
                lines.append(f"| {fam} | {acc} | {tok} | {dacc} | {dtok} | {wins} |")
            chosen = report[name]["Closing adaptive"][rule]["params"]
            if chosen:
                modal = ", ".join(f"{params} ×{count}" for params, count in chosen.items())
                lines.append("")
                lines.append(f"Closing adaptive parameters selected on train: {modal}.")
            lines.append("")
    text = "\n".join(lines)
    (out_dir / "results.md").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(text)
    print(f"Wrote {out_dir}/results.md")


if __name__ == "__main__":
    main()
