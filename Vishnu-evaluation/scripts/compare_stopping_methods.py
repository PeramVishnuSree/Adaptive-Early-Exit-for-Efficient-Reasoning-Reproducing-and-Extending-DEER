"""Compare early-exit stopping methods offline: train on training trajectories,
test on held-out test trajectories.

Inputs are scored never-exit runs (from score_run.py): every checkpoint has a
trial answer, its confidence and whether it is correct, so any stopping rule can
be replayed without the GPU (deer_eval.simulate). An early exit is scored on the
trial answer it stopped on, i.e. DEER with --final_answer trial.

Methods
  Full reasoning           never stop early
  DEER tau                 stop when confidence > tau (the fixed-threshold baseline)
  Consistency k            stop when the trial answer is the same k checkpoints in a row
  Consistency + conf t     same answer as the previous checkpoint and confidence > t
  LogReg / GBDT / MLP      learned p(trial answer correct) from 12 trajectory features
                           (Saichetan-adaptive/adaptive_exit); stop when p >= threshold
  GBDT ablations           confidence features only; + answer stability; all features
  Perfect stopping         stops at the first correct trial answer (upper bound)

Every method is a curve (one point per threshold). To compare methods at equal
cost, each method's threshold is chosen on the TRAINING set: the most accurate
threshold whose average output tokens do not exceed DEER tau=0.95's on the
training set. That threshold is then applied to the test set and compared with
DEER tau=0.95 by a paired bootstrap over test problems.

Usage (from Vishnu-evaluation/; needs torch and scikit-learn):
    python scripts/compare_stopping_methods.py \
        --train gsm8k=outputs/today/never_gsm8k_train.scored.jsonl \
                math=outputs/today/never_math_train.scored.jsonl \
        --test  gsm8k=results/checkpoint3/never_exit/never_gsm8k.scored.jsonl \
                math=results/checkpoint3/never_exit/never_math.scored.jsonl \
        --out_dir results/methods_offline
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ADAPTIVE = ROOT.parent / "Saichetan-adaptive"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ADAPTIVE))

import pandas as pd  # noqa: E402

from deer_eval.io import read_jsonl, write_json  # noqa: E402
from deer_eval.metrics import summarize  # noqa: E402
from deer_eval.simulate import simulate_policy, stability_policy  # noqa: E402

DEER_TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
P_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
CONS_CONF = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]
REFERENCE = "DEER tau=0.95"

CONF_FEATURES = ["trial_confidence", "delta_c1", "delta_c2", "ema_conf", "conf_drawdown",
                 "token_entropy", "dispersion_min_prob", "conf_entropy_ratio"]
STABILITY_FEATURES = ["ans_match_prev1", "ans_stability_streak"]


def parse_pairs(items):
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        out[name] = read_jsonl(path)
    return out


def qid(ds, rec):
    return f"{ds}:{rec['idx']}"


def feature_table(runs):
    """One row per checkpoint, 12 features + label, keyed by (question id, checkpoint number)."""
    from adaptive_exit.features import engineer_trajectory_features, features_from_deer_checkpoints
    frames = []
    for ds, recs in runs.items():
        for r in recs:
            if r.get("checkpoints"):
                frames.append(features_from_deer_checkpoints(r["checkpoints"], problem_id=qid(ds, r)))
    feats, cols = engineer_trajectory_features(pd.concat(frames, ignore_index=True))
    return feats, cols


def capped(policy, max_checks):
    """DEER checks at most max_checks times; later checkpoints never stop."""
    if not max_checks:
        return policy
    return lambda cps, rec: len(cps) <= max_checks and policy(cps, rec)


def run(recs, ds, policy, max_checks):
    held = [dict(r, _qid=qid(ds, r)) for r in recs]
    return simulate_policy(held, capped(policy, max_checks))


def threshold(tau):
    return lambda cps, rec: cps[-1]["confidence"] > tau


def cons_conf(t):
    def stop(cps, rec):
        if len(cps) < 2:
            return False
        a, b = cps[-2]["trial_answer"], cps[-1]["trial_answer"]
        return b != "" and a == b and cps[-1]["confidence"] > t
    return stop


def oracle(cps, rec):
    return bool(cps[-1]["trial_correct"])


def prob_policy(lookup, t):
    return lambda cps, rec: lookup.get((rec["_qid"], len(cps)), 0.0) >= t


# ---------------------------------------------------------------- learned scorers
def fit_models(train_feats, cols, seed):
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from adaptive_exit.controller import StoppingController

    X, y = train_feats[cols].values, train_feats["y_correct"].values
    models = {}
    models["LogReg"] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y), cols)
    models["GBDT"] = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed).fit(X, y), cols)
    torch.manual_seed(seed)
    mlp = StoppingController(device="cpu")
    mlp.fit(X, y, epochs=15)
    models["MLP"] = (mlp, cols)
    for name, sub in [("GBDT conf-only", CONF_FEATURES), ("GBDT conf+stability", CONF_FEATURES + STABILITY_FEATURES)]:
        models[name] = (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
                        .fit(train_feats[sub].values, y), sub)
    return models


def predict(model, feats):
    m, cols = model
    X = feats[cols].values
    p = m.predict_proba(X)
    return p[:, 1] if np.ndim(p) == 2 else p


def lookup_table(feats, probs):
    return {(pid, int(k)): float(p) for pid, k, p in zip(feats["problem_id"], feats["checkpoint_idx"], probs)}


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores)).rank().values
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------- evaluation
def curves_for(runs, feats_by_split, models, max_checks):
    """{dataset: {method family: [(param, per-question results), ...]}} for one split."""
    split_feats = feats_by_split
    out = {}
    for ds, recs in runs.items():
        fam = {"Full reasoning": [(None, run(recs, ds, lambda c, r: False, max_checks))],
               "Perfect stopping": [(None, run(recs, ds, oracle, max_checks))],
               "DEER": [(t, run(recs, ds, threshold(t), max_checks)) for t in DEER_TAUS],
               "Consistency k": [(k, run(recs, ds, stability_policy(k), max_checks)) for k in (2, 3)],
               "Consistency + conf": [(t, run(recs, ds, cons_conf(t), max_checks)) for t in CONS_CONF]}
        for name, model in models.items():
            lk = lookup_table(split_feats, predict(model, split_feats))
            fam[name] = [(t, run(recs, ds, prob_policy(lk, t), max_checks)) for t in P_THRESHOLDS]
        out[ds] = fam
    return out


def point(results):
    s = summarize(results)
    return {"accuracy": s["accuracy"], "output_tokens": s["avg_output_tokens"],
            "reasoning_tokens": s["avg_reasoning_tokens"], "compute_tokens": s["avg_compute_tokens"]}


def paired_bootstrap(a, b, n_boot=4000, seed=0):
    """CI of mean(a) - mean(b) over paired per-question values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def select_on_train(train_curve, budget):
    """Most accurate parameter on train whose tokens <= budget (ties: fewer tokens)."""
    best = None
    for param, res in train_curve:
        p = point(res)
        if p["output_tokens"] <= budget + 1e-9:
            key = (p["accuracy"], -p["output_tokens"])
            if best is None or key > best[0]:
                best = (key, param)
    return None if best is None else best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="dataset=scored never-exit jsonl")
    ap.add_argument("--test", nargs="+", required=True, help="dataset=scored never-exit jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_checks", type=int, default=10, help="DEER's max_judge_steps (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test = parse_pairs(args.train), parse_pairs(args.test)
    train_feats, cols = feature_table(train)
    test_feats, _ = feature_table(test)
    print(f"train: {len(train_feats)} checkpoints from {train_feats['problem_id'].nunique()} questions; "
          f"test: {len(test_feats)} checkpoints from {test_feats['problem_id'].nunique()} questions")

    models = fit_models(train_feats, cols, args.seed)
    aucs = {"DEER confidence": auc(test_feats["trial_confidence"], test_feats["y_correct"])}
    for name, m in models.items():
        aucs[name] = auc(predict(m, test_feats), test_feats["y_correct"])

    train_curves = curves_for(train, train_feats, models, args.max_checks)
    test_curves = curves_for(test, test_feats, models, args.max_checks)

    summary = {"auc_test_checkpoints": aucs, "max_checks": args.max_checks, "datasets": {}}
    lines = ["# Stopping methods: trained on training trajectories, evaluated on test", "",
             "Early exits are scored on the trial answer they stop on. Operating points are chosen on the "
             f"training set at <= the tokens of {REFERENCE}; differences vs. {REFERENCE} are paired "
             "bootstrap 95% CIs over test problems.", "",
             "Checkpoint-level ROC-AUC on test (does the score predict a correct trial answer?): "
             + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()), ""]

    for ds in test:
        tc, trc = test_curves[ds], train_curves.get(ds)
        ref_test = dict(tc["DEER"])[0.95]
        ref_train_tokens = point(dict(trc["DEER"])[0.95])["output_tokens"] if trc else None
        full = point(tc["Full reasoning"][0][1])
        orc = point(tc["Perfect stopping"][0][1])
        ref = point(ref_test)
        rows = []
        for fam, curve in tc.items():
            over = False
            if fam in ("Full reasoning", "Perfect stopping"):
                param = None
            elif fam == "DEER":
                param = 0.95
            else:
                param = select_on_train(trc[fam], ref_train_tokens) if trc else None
                over = param is None
                if over:  # no setting fits the budget on train: report the cheapest one
                    src = trc[fam] if trc else curve
                    param = min(src, key=lambda pr: point(pr[1])["output_tokens"])[0]
            res = dict(curve)[param]
            p = point(res)
            label = fam if param is None else f"{fam} ({param}{', over budget' if over else ''})"
            acc_d = paired_bootstrap([r["correct"] for r in res], [r["correct"] for r in ref_test])
            tok_d = paired_bootstrap([r["output_tokens"] for r in res], [r["output_tokens"] for r in ref_test])
            gap = (orc["accuracy"] - ref["accuracy"])
            rows.append({"method": label, "family": fam, "param": param, "over_budget": over,
                         **p, "acc_diff_vs_deer": acc_d, "token_diff_vs_deer": tok_d,
                         "gap_closed": (p["accuracy"] - ref["accuracy"]) / gap if gap > 0 else float("nan")})
        summary["datasets"][ds] = {
            "n_test": len(test[ds]), "operating_points": rows,
            "curves": {fam: [{"param": prm, **point(res)} for prm, res in curve] for fam, curve in tc.items()}}

        lines += [f"## {ds} ({len(test[ds])} test problems)", "",
                  "| Method (param chosen on train) | Acc. (%) | Output tok | Δ acc vs DEER (95% CI) | Δ tok vs DEER (95% CI) | Gap to perfect closed |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            a, t = r["acc_diff_vs_deer"], r["token_diff_vs_deer"]
            gap = "" if np.isnan(r["gap_closed"]) else "%.0f%%" % (100 * r["gap_closed"])
            lines.append(f"| {r['method']} | {100 * r['accuracy']:.1f} | {r['output_tokens']:.0f} | "
                         f"{100 * a[0]:+.1f} ({100 * a[1]:+.1f}, {100 * a[2]:+.1f}) | "
                         f"{t[0]:+.0f} ({t[1]:+.0f}, {t[2]:+.0f}) | {gap} |")
        lines.append("")
        plot(ds, summary["datasets"][ds]["curves"], full, orc, out_dir)

    write_json(summary, out_dir / "summary.json")
    (out_dir / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/results.md, summary.json, tradeoff_<dataset>.png")


def plot(ds, curves, full, orc, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"DEER": ("#2a78d6", "o-"), "LogReg": ("#1baf7a", "s--"), "GBDT": ("#eb6834", "^-"),
             "MLP": ("#e87ba4", "v--"), "Consistency + conf": ("#6b6b6b", "D:")}
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for fam, (color, fmt) in style.items():
        pts = sorted((c["output_tokens"], 100 * c["accuracy"]) for c in curves[fam])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], fmt, color=color, ms=4, lw=1.5, label=fam)
    for c in curves["Consistency k"]:
        ax.plot(c["output_tokens"], 100 * c["accuracy"], "P", color="#008300", ms=8)
        ax.annotate(f"k={c['param']}", (c["output_tokens"], 100 * c["accuracy"]), xytext=(4, -10),
                    textcoords="offset points", fontsize=7)
    ax.plot(full["output_tokens"], 100 * full["accuracy"], "s", color="black", ms=8, label="Full reasoning")
    ax.plot(orc["output_tokens"], 100 * orc["accuracy"], "*", color="#eda100", ms=13, label="Perfect stopping (bound)")
    deer95 = next(c for c in curves["DEER"] if c["param"] == 0.95)
    ax.annotate("τ=0.95", (deer95["output_tokens"], 100 * deer95["accuracy"]), xytext=(4, 6),
                textcoords="offset points", fontsize=7, color="#2a78d6")
    ax.set_xlabel("Average output tokens (test)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{ds}: stopping methods (trained on train, tested on test)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"tradeoff_{ds}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
