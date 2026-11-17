"""Held-out comparison of DEER (fixed threshold) and AdaDEER (adaptive threshold).

Reproduces the Check-In 3 protocol (Table III): problems are split 70/30 by
problem (per dataset), repeated over 5 seeds, every policy is replayed on the
held-out problems with the offline simulator.  Hyper-parameters (DEER's tau and
AdaDEER's coefficients) are chosen on the *training* problems only, with the
same objective for both, so the comparison is tuned-vs-tuned.

Usage
  python -m adadeer.experiments --data gsm8k_never_exit.jsonl math500_never_exit.jsonl \
         --out results/ --think-budget 4800 --max-checkpoints 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics as st
from dataclasses import asdict

from .policy import AdaDEER, AdaDEERConfig, EvidenceDEER, FixedThresholdDEER, Policy, normalize_answer
from .replay import Outcome, Problem, ReplayConfig, load_jsonl, replay, summarize

DEER_TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]

ADA_GRID = dict(
    tau0=[0.95, 0.97, 0.99, 0.995, 0.999],
    alpha=[0.0, 0.03, 0.06, 0.10],
    gamma=[0.0, 0.5, 1.0],
    beta=[0.0, 0.05, 0.10],
    eta=[0.0, 0.10, 0.20],
    tau_min=[0.80, 0.90],
)
EVID_GRID = dict(theta=[3.0, 3.5, 4.0, 4.6, 5.0, 5.5, 6.0, 6.9], rho=[0.0, 0.25, 0.5, 0.75, 1.0])
EVID_FROZEN = dict(theta=4.6, rho=1.0)     # fixed on the dev set before seeing eval problems
EVID_NAME = "AdaDEER-E frozen (theta=4.6, rho=1)"


# ---------------------------------------------------------------- baselines
class NeverExit(Policy):
    name = "Full reasoning (never-exit)"
    def reset(self, think_budget=None): pass
    def should_exit(self, answer, confidence, reasoning_tokens): return False


class Consistency(Policy):
    """Exit when the trial answer repeats (optionally with rising confidence)."""
    def __init__(self, momentum=False):
        self.momentum = momentum
        self.name = "Consistency + momentum" if momentum else "Consistency (2 in a row)"
    def reset(self, think_budget=None):
        self.prev, self.prev_c = None, None
    def should_exit(self, answer, confidence, reasoning_tokens):
        a = normalize_answer(answer)
        ok = a is not None and a == self.prev
        if self.momentum:
            ok = ok and self.prev_c is not None and confidence > self.prev_c
        self.prev, self.prev_c = a, confidence
        return ok


# ---------------------------------------------------------------- splitting
def make_splits(problems, protocol, n_splits, seed, test_frac=0.3):
    by_ds = {}
    for i, p in enumerate(problems):
        by_ds.setdefault(p.dataset, []).append(i)
    splits = []
    if protocol == "random":
        for s in range(n_splits):
            rng = random.Random(seed + s)
            test = []
            for ds, idx in sorted(by_ds.items()):
                idx = idx[:]
                rng.shuffle(idx)
                test += idx[: round(test_frac * len(idx))]
            splits.append(sorted(test))
    else:  # kfold: every problem is in exactly one test fold
        rng = random.Random(seed)
        folds = [[] for _ in range(n_splits)]
        for ds, idx in sorted(by_ds.items()):
            idx = idx[:]
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                folds[j % n_splits].append(i)
        splits = [sorted(f) for f in folds]
    return splits


# ---------------------------------------------------------------- tuning
def _score(summ, ref, objective, lam, full_tok):
    if objective == "match-deer":   # most accurate policy that is no more costly than DEER(0.95)
        if summ["tokens"] > ref["tokens"] * 1.0001:
            return -1e9 - summ["tokens"]
        return summ["acc"] - 1e-6 * summ["tokens"]
    return summ["acc"] - lam * 100.0 * summ["tokens"] / full_tok


def tune(train, rc, objective, lam, kind):
    full_tok = summarize(replay(train, NeverExit(), rc))["tokens"]
    ref = summarize(replay(train, FixedThresholdDEER(0.95), rc))
    best, best_s = None, -math.inf
    if kind == "deer":
        cands = [FixedThresholdDEER(t) for t in DEER_TAUS]
    elif kind == "evidence":
        cands = [EvidenceDEER(th, r) for th in EVID_GRID["theta"] for r in EVID_GRID["rho"]]
    else:
        keys = list(ADA_GRID)
        cands = [AdaDEER(AdaDEERConfig(**dict(zip(keys, v))))
                 for v in itertools.product(*ADA_GRID.values())]
    for pol in cands:
        s = _score(summarize(replay(train, pol, rc)), ref, objective, lam, full_tok)
        if s > best_s:
            best, best_s = pol, s
    return best


# ---------------------------------------------------------------- iso-token comparison
def interp_curve(points, x):
    """Accuracy of a token->accuracy curve (Pareto upper envelope) at x tokens."""
    pts = sorted(points)
    env, best = [], -1
    for t, a in pts:
        best = max(best, a)
        env.append((t, best))
    if x <= env[0][0]:
        return env[0][1]
    if x >= env[-1][0]:
        return env[-1][1]
    for (t0, a0), (t1, a1) in zip(env, env[1:]):
        if t0 <= x <= t1:
            return a0 if t1 == t0 else a0 + (a1 - a0) * (x - t0) / (t1 - t0)
    return env[-1][1]


# ---------------------------------------------------------------- stats
def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(a_out, b_out, iters=5000, seed=0):
    rng = random.Random(seed)
    n = len(a_out)
    dacc, dtok = [], []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        dacc.append(100 * sum(a_out[i].correct - b_out[i].correct for i in idx) / n)
        dtok.append(sum(a_out[i].output_tokens - b_out[i].output_tokens for i in idx) / n)
    q = lambda v: (sorted(v)[int(0.025 * iters)], sorted(v)[int(0.975 * iters)])
    return q(dacc), q(dtok)


# ---------------------------------------------------------------- main
def run(args):
    problems = load_jsonl(args.data)
    rc = ReplayConfig(args.max_checkpoints, args.think_budget, args.conf_mode)
    datasets = sorted({p.dataset for p in problems})
    splits = make_splits(problems, args.protocol, args.n_splits, args.seed)
    os.makedirs(args.out, exist_ok=True)

    default_ada = AdaDEERConfig()
    ablations = {
        "AdaDEER defaults": default_ada,
        "  - no stability (alpha=0)": AdaDEERConfig(**{**asdict(default_ada), "alpha": 0.0}),
        "  - no volatility (gamma=0)": AdaDEERConfig(**{**asdict(default_ada), "gamma": 0.0}),
        "  - no switching (beta=0)": AdaDEERConfig(**{**asdict(default_ada), "beta": 0.0}),
        "  - no budget (eta=0)": AdaDEERConfig(**{**asdict(default_ada), "eta": 0.0}),
    }

    rows = {}            # policy name -> dataset -> list of per-split summaries
    oof = {}             # policy name -> {problem index: Outcome} (k-fold only)
    chosen = []          # tuned hyper-parameters per split
    iso = {ds: [] for ds in datasets}

    def record(name, test_idx, outs):
        for ds in datasets + ["all"]:
            sel = [o for i, o in zip(test_idx, outs) if ds == "all" or problems[i].dataset == ds]
            rows.setdefault(name, {}).setdefault(ds, []).append(summarize(sel))
        for i, o in zip(test_idx, outs):
            oof.setdefault(name, {})[i] = o

    for s, test_idx in enumerate(splits):
        test_set = set(test_idx)
        train = [p for i, p in enumerate(problems) if i not in test_set]
        test = [problems[i] for i in test_idx]

        deer_t = tune(train, rc, args.objective, args.lam, "deer")
        ada_t = tune(train, rc, args.objective, args.lam, "adadeer")
        ev_t = tune(train, rc, args.objective, args.lam, "evidence")
        chosen.append({"split": s, "deer_tau": deer_t.tau, "adadeer": asdict(ada_t.cfg),
                       "evidence": {"theta": ev_t.theta, "rho": ev_t.rho}})

        fixed = [NeverExit(), FixedThresholdDEER(0.90), FixedThresholdDEER(0.95),
                 FixedThresholdDEER(0.99), Consistency(False), Consistency(True)]
        for pol in fixed:
            record(pol.name, test_idx, replay(test, pol, rc))
        record("DEER tuned tau (train)", test_idx, replay(test, deer_t, rc))
        for name, cfg in ablations.items():
            record(name, test_idx, replay(test, AdaDEER(cfg), rc))
        record("AdaDEER tuned (train)", test_idx, replay(test, ada_t, rc))
        record(EVID_NAME, test_idx, replay(test, EvidenceDEER(**EVID_FROZEN), rc))
        record("AdaDEER-E tuned (train)", test_idx, replay(test, ev_t, rc))

        # iso-token accuracy gain over the whole DEER tau curve, per dataset
        for ds in datasets:
            sub = [p for p in test if p.dataset == ds]
            curve = [(sm["tokens"], sm["acc"]) for sm in
                     (summarize(replay(sub, FixedThresholdDEER(t), rc)) for t in DEER_TAUS)]
            for name, pol in [("AdaDEER defaults", AdaDEER(default_ada)), ("AdaDEER tuned", ada_t),
                              (EVID_NAME, EvidenceDEER(**EVID_FROZEN)), ("AdaDEER-E tuned", ev_t)]:
                sm = summarize(replay(sub, pol, rc))
                iso[ds].append({"split": s, "policy": name, "tokens": sm["tokens"], "acc": sm["acc"],
                                "deer_acc_at_same_tokens": interp_curve(curve, sm["tokens"]),
                                "gain": sm["acc"] - interp_curve(curve, sm["tokens"])})

    # ---------------- tables
    def ms(v):
        v = [x for x in v if not math.isnan(x)]
        if not v:
            return "–"
        return f"{st.mean(v):.1f} ± {st.pstdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"

    lines = [f"# Held-out results ({args.protocol}, {len(splits)} splits; mean ± std over splits)",
             f"think budget {args.think_budget}, max checkpoints {args.max_checkpoints}, "
             f"confidence {args.conf_mode}, tuning objective {args.objective}", ""]
    hdr = "| Policy | " + " | ".join(f"{d} Acc (%) | {d} Tokens | {d} Exits" for d in datasets) + " |"
    lines += [hdr, "|" + "---|" * (1 + 3 * len(datasets))]
    for name, per in rows.items():
        cells = []
        for d in datasets:
            r = per[d]
            cells += [ms([x["acc"] for x in r]), f"{st.mean(x['tokens'] for x in r):,.0f}",
                      f"{st.mean(x['exits'] for x in r):.1f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Iso-token accuracy gain vs. the full DEER tau curve (test problems)", "",
              "| Dataset | Policy | Acc | Tokens | DEER acc @ same tokens | Gain (pts) |", "|---|---|---|---|---|---|"]
    for ds in datasets:
        for name in ["AdaDEER defaults", "AdaDEER tuned", EVID_NAME, "AdaDEER-E tuned"]:
            r = [x for x in iso[ds] if x["policy"] == name]
            lines.append(f"| {ds} | {name} | {ms([x['acc'] for x in r])} | "
                         f"{st.mean(x['tokens'] for x in r):,.0f} | "
                         f"{ms([x['deer_acc_at_same_tokens'] for x in r])} | {ms([x['gain'] for x in r])} |")

    if args.protocol == "kfold":
        lines += ["", "## Paired test over all problems (out-of-fold), AdaDEER vs DEER tau=0.95", "",
                  "| Dataset | Policy | dAcc 95% CI | dTokens 95% CI | AdaDEER-only correct | DEER-only correct | McNemar p |",
                  "|---|---|---|---|---|---|---|"]
        for ds in datasets + ["all"]:
            idx = [i for i in range(len(problems)) if ds == "all" or problems[i].dataset == ds]
            base = [oof["DEER tau=0.95"][i] for i in idx]
            for name in ["AdaDEER defaults", "AdaDEER tuned (train)", EVID_NAME, "AdaDEER-E tuned (train)"]:
                new = [oof[name][i] for i in idx]
                b = sum(n.correct and not o.correct for n, o in zip(new, base))
                c = sum(o.correct and not n.correct for n, o in zip(new, base))
                (a_lo, a_hi), (t_lo, t_hi) = paired_bootstrap(new, base)
                lines.append(f"| {ds} | {name} | [{a_lo:+.1f}, {a_hi:+.1f}] | [{t_lo:+.0f}, {t_hi:+.0f}] | "
                             f"{b} | {c} | {mcnemar_exact(b, c):.3f} |")

        levels = sorted({p.level for p in problems if p.level is not None})
        if levels:
            lines += ["", "## MATH-500 by level (out-of-fold)", "",
                      "| Level | n | DEER 0.95 acc | DEER 0.95 tok | AdaDEER acc | AdaDEER tok |", "|---|---|---|---|---|---|"]
            for L in levels:
                idx = [i for i, p in enumerate(problems) if p.level == L]
                d = summarize([oof["DEER tau=0.95"][i] for i in idx])
                a = summarize([oof[EVID_NAME][i] for i in idx])
                lines.append(f"| {L} | {len(idx)} | {d['acc']:.1f} | {d['tokens']:,.0f} | {a['acc']:.1f} | {a['tokens']:,.0f} |")

    lines += ["", "## Hyper-parameters chosen on the training problems", "", "```",
              *[json.dumps(c) for c in chosen], "```"]
    table = "\n".join(lines)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"rows": rows, "iso": iso, "chosen": chosen, "args": vars(args)}, f, indent=1, default=str)
    print(table)

    if not args.no_plot:
        plot_tradeoff(problems, splits, rc, datasets, args.out)


def plot_tradeoff(problems, splits, rc, datasets, out):
    """Accuracy vs tokens on test problems: DEER tau sweep vs AdaDEER tau0 sweep (Fig. 3 style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tau0s = [0.9, 0.93, 0.95, 0.97, 0.99, 0.995, 0.999]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for label, mk, pols, color in [
            ("DEER (τ)", "o", [FixedThresholdDEER(t) for t in DEER_TAUS[3:]], "#1f77b4"),
            ("AdaDEER (τ0, default coefs)", "s",
             [AdaDEER(AdaDEERConfig(tau0=t)) for t in tau0s], "#ff7f0e"),
            ("AdaDEER-E (θ, ρ=1)", "^", [EvidenceDEER(th, 1.0) for th in EVID_GRID["theta"]], "#d62728")]:
            xs, ys = [], []
            for pol in pols:
                acc, tok = [], []
                for test_idx in splits:
                    sm = summarize(replay([problems[i] for i in test_idx if problems[i].dataset == ds], pol, rc))
                    acc.append(sm["acc"]); tok.append(sm["tokens"])
                xs.append(st.mean(tok)); ys.append(st.mean(acc))
            ax.plot(xs, ys, marker=mk, label=label, color=color)
        full = [summarize(replay([problems[i] for i in t if problems[i].dataset == ds], NeverExit(), rc)) for t in splits]
        ax.scatter([st.mean(f["tokens"] for f in full)], [st.mean(f["acc"] for f in full)],
                   marker="s", color="black", label="Full reasoning")
        ax.set_title(ds); ax.set_xlabel("avg output tokens (held-out)"); ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tradeoff.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="never-exit trajectory JSONL file(s)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--protocol", choices=["random", "kfold"], default="random",
                    help="random = Check-In 3 protocol (5 x 70/30); kfold = every problem tested once")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think-budget", type=float, default=4800)
    ap.add_argument("--max-checkpoints", type=int, default=10)
    ap.add_argument("--conf-mode", default="logged", choices=["logged", "avg1", "gm", "min"])
    ap.add_argument("--objective", default="match-deer", choices=["match-deer", "tradeoff"])
    ap.add_argument("--lam", type=float, default=0.05, help="accuracy points per 1%% of full tokens (tradeoff)")
    ap.add_argument("--no-plot", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
