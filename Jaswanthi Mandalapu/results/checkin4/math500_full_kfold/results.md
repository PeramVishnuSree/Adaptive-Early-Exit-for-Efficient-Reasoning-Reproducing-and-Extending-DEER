# Held-out results (kfold, 5 splits; mean ± std over splits)
think budget 4800, max checkpoints 10, confidence logged, tuning objective match-deer

| Policy | math500 Acc (%) | math500 Tokens | math500 Exits |
|---|---|---|---|
| Full reasoning (never-exit) | 79.0 ± 3.2 | 3,209 | 0.0 |
| DEER tau=0.9 | 75.0 ± 3.2 | 1,438 | 87.0 |
| DEER tau=0.95 | 76.6 ± 2.7 | 1,623 | 83.2 |
| DEER tau=0.99 | 79.0 ± 2.8 | 1,892 | 75.8 |
| Consistency (2 in a row) | 74.8 ± 2.5 | 1,250 | 74.6 |
| Consistency + momentum | 76.4 ± 3.6 | 1,405 | 71.4 |
| DEER tuned tau (train) | 76.6 ± 2.7 | 1,623 | 83.2 |
| AdaDEER defaults | 80.0 ± 1.8 | 1,694 | 80.2 |
|   - no stability (alpha=0) | 80.0 ± 2.8 | 2,079 | 67.8 |
|   - no volatility (gamma=0) | 80.2 ± 2.0 | 1,628 | 82.2 |
|   - no switching (beta=0) | 80.2 ± 2.0 | 1,675 | 80.6 |
|   - no budget (eta=0) | 80.0 ± 1.8 | 1,704 | 79.4 |
| AdaDEER tuned (train) | 80.2 ± 2.1 | 1,559 | 84.2 |
| AdaDEER-E frozen (theta=4.6, rho=1) | 79.6 ± 2.7 | 1,488 | 85.2 |
| AdaDEER-E tuned (train) | 79.4 ± 2.4 | 1,545 | 84.0 |

## Iso-token accuracy gain vs. the full DEER tau curve (test problems)

| Dataset | Policy | Acc | Tokens | DEER acc @ same tokens | Gain (pts) |
|---|---|---|---|---|---|
| math500 | AdaDEER defaults | 80.0 ± 1.8 | 1,694 | 77.9 ± 2.6 | 2.1 ± 1.4 |
| math500 | AdaDEER tuned | 80.2 ± 2.1 | 1,559 | 77.0 ± 2.9 | 3.2 ± 1.7 |
| math500 | AdaDEER-E frozen (theta=4.6, rho=1) | 79.6 ± 2.7 | 1,488 | 75.9 ± 3.9 | 3.7 ± 2.5 |
| math500 | AdaDEER-E tuned | 79.4 ± 2.4 | 1,545 | 76.4 ± 4.0 | 3.0 ± 2.6 |

## Paired test over all problems (out-of-fold), AdaDEER vs DEER tau=0.95

| Dataset | Policy | dAcc 95% CI | dTokens 95% CI | AdaDEER-only correct | DEER-only correct | McNemar p |
|---|---|---|---|---|---|---|
| math500 | AdaDEER defaults | [+1.8, +5.2] | [-11, +151] | 18 | 1 | 0.000 |
| math500 | AdaDEER tuned (train) | [+2.0, +5.4] | [-154, +15] | 19 | 1 | 0.000 |
| math500 | AdaDEER-E frozen (theta=4.6, rho=1) | [+1.2, +5.0] | [-236, -44] | 19 | 4 | 0.003 |
| math500 | AdaDEER-E tuned (train) | [+1.0, +4.6] | [-170, +6] | 17 | 3 | 0.003 |
| all | AdaDEER defaults | [+1.8, +5.2] | [-11, +151] | 18 | 1 | 0.000 |
| all | AdaDEER tuned (train) | [+2.0, +5.4] | [-154, +15] | 19 | 1 | 0.000 |
| all | AdaDEER-E frozen (theta=4.6, rho=1) | [+1.2, +5.0] | [-236, -44] | 19 | 4 | 0.003 |
| all | AdaDEER-E tuned (train) | [+1.0, +4.6] | [-170, +6] | 17 | 3 | 0.003 |

## MATH-500 by level (out-of-fold)

| Level | n | DEER 0.95 acc | DEER 0.95 tok | AdaDEER acc | AdaDEER tok |
|---|---|---|---|---|---|
| 1 | 43 | 90.7 | 769 | 93.0 | 632 |
| 2 | 90 | 90.0 | 883 | 92.2 | 869 |
| 3 | 105 | 85.7 | 1,194 | 90.5 | 1,147 |
| 4 | 128 | 75.8 | 1,596 | 78.9 | 1,587 |
| 5 | 134 | 56.7 | 2,756 | 59.0 | 2,352 |

## Hyper-parameters chosen on the training problems

```
{"split": 0, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.0, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 0.75}}
{"split": 1, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.0, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 2, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.0, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 3, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.9, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 0.75}}
{"split": 4, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.1, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 0.75}}
```
