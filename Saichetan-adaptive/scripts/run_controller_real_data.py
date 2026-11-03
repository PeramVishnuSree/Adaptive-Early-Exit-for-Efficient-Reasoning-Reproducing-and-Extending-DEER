#!/usr/bin/env python3
"""Train and evaluate the AdaptiveDEER-MLP controller on real DEER trajectories.

Data: the never-exit runs in Vishnu-evaluation/results/checkpoint3/never_exit/
(R1-Distill-Qwen-1.5B, first 100 GSM8K + first 100 MATH-500 questions, a trial
answer and confidence logged at every "Wait" checkpoint).

For each random seed:
  1. Split the questions 70/30 (by question, so no question is in both sets).
  2. Train the MLP on the checkpoints of the training questions (both datasets pooled).
  3. Replay stopping policies on the held-out questions only:
       - full reasoning (never stop early)
       - DEER: stop when confidence > tau
       - answer consistency: same trial answer at two checkpoints in a row
       - consistency + momentum: same answer and confidence went up
       - AdaptiveDEER-MLP at several probability thresholds
Results are averaged over seeds and written to --out-dir.

Usage (from Saichetan-adaptive/):
    python scripts/run_controller_real_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ADAPTIVE_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptive-dir", default=str(ADAPTIVE_DIR), help="Saichetan-adaptive folder")
    ap.add_argument("--data-dir", default=None,
                    help="Folder with never_<ds>.scored.jsonl (default: Vishnu's checkpoint 3 runs)")
    ap.add_argument("--out-dir", default=None, help="Default: <adaptive-dir>/results/checkpoint3_real")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=15)
    return ap.parse_args()


args = parse_args()
ROOT = Path(args.adaptive_dir).resolve()
VISHNU = ROOT.parent / "Vishnu-evaluation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VISHNU))

from adaptive_exit.controller import StoppingController  # noqa: E402
from adaptive_exit.features import (FEATURE_COLS, engineer_trajectory_features,  # noqa: E402
                                    features_from_deer_checkpoints)
from deer_eval.io import read_jsonl  # noqa: E402
from deer_eval.metrics import summarize  # noqa: E402
from deer_eval.simulate import simulate_policy, stability_policy, threshold_policy  # noqa: E402

DATASETS = ["gsm8k", "math"]
DEER_TAUS = [0.8, 0.9, 0.95, 0.99]
MLP_THRESHOLDS = [0.3, 0.4, 0.5, 0.62, 0.7, 0.8, 0.9]


def qid(ds, rec):
    return f"{ds}_{rec['idx']}"


def load_data(data_dir):
    """Records per dataset, plus one feature row per checkpoint (all questions)."""
    records, frames = {}, []
    for ds in DATASETS:
        recs = read_jsonl(data_dir / f"never_{ds}.scored.jsonl")
        records[ds] = recs
        for r in recs:
            if r.get("checkpoints"):
                frames.append(features_from_deer_checkpoints(r["checkpoints"], problem_id=qid(ds, r)))
    raw = pd.concat(frames, ignore_index=True)
    # All features are causal (they use only checkpoints 1..k), so row k of the full
    # trajectory equals what the controller would see after k checkpoints.
    feats, _ = engineer_trajectory_features(raw)
    return records, feats


def auc(scores, labels):
    """ROC-AUC via ranks (ties averaged)."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().values
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def momentum_policy(cps, rec):
    if len(cps) < 2:
        return False
    a, b = cps[-2], cps[-1]
    return b.get("trial_answer", "") != "" and b["trial_answer"] == a["trial_answer"] \
        and b["confidence"] > a["confidence"]


def run_seed(seed, records, feats):
    rng = np.random.default_rng(seed)
    all_ids = sorted(qid(ds, r) for ds in DATASETS for r in records[ds])
    rng.shuffle(all_ids)
    test_ids = set(all_ids[: int(round(len(all_ids) * args.test_size))])

    train = feats[~feats["problem_id"].isin(test_ids)]
    test = feats[feats["problem_id"].isin(test_ids)].copy()

    torch.manual_seed(seed)
    ctrl = StoppingController()
    ctrl.fit(train[FEATURE_COLS].values, train["y_correct"].values, epochs=args.epochs)
    test["p_mlp"] = ctrl.predict_proba(test[FEATURE_COLS].values)

    # probability per (question, checkpoint number)
    p_lookup = {(r.problem_id, int(r.checkpoint_idx)): float(r.p_mlp) for r in test.itertuples()}

    def mlp_policy(t):
        return lambda cps, rec: p_lookup[(rec["_qid"], len(cps))] >= t

    policies = {"Full reasoning": lambda cps, rec: False}
    policies.update({f"DEER tau={t}": threshold_policy(t) for t in DEER_TAUS})
    policies["Consistency (2 in a row)"] = stability_policy(2)
    policies["Consistency + momentum"] = momentum_policy
    policies.update({f"MLP p>={t}": mlp_policy(t) for t in MLP_THRESHOLDS})

    out = {"auc": {"MLP": auc(test["p_mlp"], test["y_correct"]),
                   "DEER confidence": auc(test["trial_confidence"], test["y_correct"])},
           "n_train_checkpoints": int(len(train)), "n_test_checkpoints": int(len(test))}
    for ds in DATASETS:
        held = [dict(r, _qid=qid(ds, r)) for r in records[ds] if qid(ds, r) in test_ids]
        out[ds] = {"n_questions": len(held)}
        for name, pol in policies.items():
            s = summarize(simulate_policy(held, pol))
            out[ds][name] = {"accuracy": s["accuracy"], "output_tokens": s["avg_output_tokens"]}
    return out


def mean_std(values):
    v = np.array(values, dtype=float)
    return float(np.nanmean(v)), float(np.nanstd(v))


def main():
    data_dir = Path(args.data_dir) if args.data_dir else VISHNU / "results/checkpoint3/never_exit"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results/checkpoint3_real"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, feats = load_data(data_dir)
    print(f"{len(feats)} checkpoints from {feats['problem_id'].nunique()} questions; "
          f"{sum(len(v) for v in records.values())} questions in total")

    runs = [run_seed(s, records, feats) for s in range(args.seeds)]
    policy_names = [k for k in runs[0]["gsm8k"] if k != "n_questions"]

    summary = {"seeds": args.seeds, "test_size": args.test_size, "epochs": args.epochs,
               "auc": {k: mean_std([r["auc"][k] for r in runs]) for k in runs[0]["auc"]}}
    lines = ["# AdaptiveDEER-MLP on real trajectories (held-out questions)", "",
             f"Mean ± std over {args.seeds} random 70/30 question splits. Accuracy uses DEER's grader; "
             "tokens are average output tokens per question. Early exits are scored on the trial answer.", "",
             f"Checkpoint-level ROC-AUC (predicting whether the trial answer is correct): "
             f"MLP {summary['auc']['MLP'][0]:.3f} ± {summary['auc']['MLP'][1]:.3f}, "
             f"DEER confidence {summary['auc']['DEER confidence'][0]:.3f} ± {summary['auc']['DEER confidence'][1]:.3f}", ""]

    for ds in DATASETS:
        summary[ds] = {}
        n_q = int(np.mean([r[ds]["n_questions"] for r in runs]))
        lines += [f"## {ds.upper()} (~{n_q} held-out questions per split)", "",
                  "| Policy | Accuracy (%) | Output tokens | Token reduction vs. full |", "|---|---|---|---|"]
        full_tok = np.mean([r[ds]["Full reasoning"]["output_tokens"] for r in runs])
        for name in policy_names:
            acc = mean_std([100 * r[ds][name]["accuracy"] for r in runs])
            tok = mean_std([r[ds][name]["output_tokens"] for r in runs])
            summary[ds][name] = {"accuracy_pct": acc, "output_tokens": tok}
            lines.append(f"| {name} | {acc[0]:.1f} ± {acc[1]:.1f} | {tok[0]:.0f} ± {tok[1]:.0f} | "
                         f"{100 * (1 - tok[0] / full_tok):.0f}% |")
        lines.append("")

    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "per_seed": runs}, indent=2))
    (out_dir / "results.md").write_text("\n".join(lines))
    plot(summary, policy_names, out_dir)
    print("\n".join(lines))
    print(f"\nWrote {out_dir}/results.md, summary.json, tradeoff_<dataset>.png")


def plot(summary, names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ds in DATASETS:
        s = summary[ds]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for prefix, color, label in [("DEER tau=", "tab:blue", "DEER (tau)"), ("MLP p>=", "tab:red", "AdaptiveDEER-MLP (p)")]:
            pts = [(n, s[n]) for n in names if n.startswith(prefix)]
            x = [v["output_tokens"][0] for _, v in pts]
            y = [v["accuracy_pct"][0] for _, v in pts]
            ax.plot(x, y, "o-", color=color, label=label)
            for (n, _), xi, yi in zip(pts, x, y):
                ax.annotate(n[len(prefix):], (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
        for n, marker, color in [("Full reasoning", "s", "black"), ("Consistency (2 in a row)", "^", "tab:green"),
                                 ("Consistency + momentum", "v", "tab:purple")]:
            ax.plot(s[n]["output_tokens"][0], s[n]["accuracy_pct"][0], marker, color=color, ms=8, label=n)
        ax.set_xlabel("Average output tokens (held-out questions)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{ds.upper()}: early-exit policies, mean of {summary['seeds']} splits")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"tradeoff_{ds}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
