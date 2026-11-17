# Held-out results (kfold, 5 splits; mean ± std over splits)
think budget 4800, max checkpoints 10, confidence logged, tuning objective match-deer

| Policy | gsm8k Acc (%) | gsm8k Tokens | gsm8k Exits | math500 Acc (%) | math500 Tokens | math500 Exits |
|---|---|---|---|---|---|---|
| Full reasoning (never-exit) | 81.0 ± 5.8 | 2,017 | 0.0 | 81.0 ± 7.3 | 2,856 | 0.0 |
| DEER tau=0.9 | 84.0 ± 3.7 | 616 | 19.0 | 76.0 ± 6.6 | 1,271 | 18.0 |
| DEER tau=0.95 | 84.0 ± 3.7 | 683 | 18.6 | 79.0 ± 4.9 | 1,409 | 17.2 |
| DEER tau=0.99 | 85.0 ± 4.5 | 846 | 16.6 | 79.0 ± 5.8 | 1,670 | 15.0 |
| Consistency (2 in a row) | 87.0 ± 2.4 | 820 | 11.2 | 79.0 ± 10.7 | 1,256 | 14.2 |
| Consistency + momentum | 88.0 ± 2.4 | 911 | 10.8 | 81.0 ± 7.3 | 1,322 | 14.0 |
| DEER tuned tau (train) | 84.0 ± 3.7 | 683 | 18.6 | 79.0 ± 4.9 | 1,409 | 17.2 |
| AdaDEER defaults | 85.0 ± 4.5 | 824 | 15.6 | 81.0 ± 5.8 | 1,493 | 16.4 |
|   - no stability (alpha=0) | 85.0 ± 4.5 | 997 | 14.4 | 80.0 ± 8.4 | 1,841 | 13.4 |
|   - no volatility (gamma=0) | 85.0 ± 4.5 | 789 | 16.4 | 82.0 ± 5.1 | 1,384 | 17.0 |
|   - no switching (beta=0) | 85.0 ± 4.5 | 820 | 15.6 | 82.0 ± 5.1 | 1,425 | 16.6 |
|   - no budget (eta=0) | 85.0 ± 4.5 | 824 | 15.6 | 81.0 ± 5.8 | 1,498 | 16.2 |
| AdaDEER tuned (train) | 84.0 ± 3.7 | 744 | 17.8 | 81.0 ± 5.8 | 1,328 | 17.2 |
| AdaDEER-E frozen (theta=4.6, rho=1) | 85.0 ± 4.5 | 667 | 18.0 | 81.0 ± 5.8 | 1,345 | 17.2 |
| AdaDEER-E tuned (train) | 83.0 ± 4.0 | 684 | 18.2 | 81.0 ± 5.8 | 1,309 | 17.4 |

## Iso-token accuracy gain vs. the full DEER tau curve (test problems)

| Dataset | Policy | Acc | Tokens | DEER acc @ same tokens | Gain (pts) |
|---|---|---|---|---|---|
| gsm8k | AdaDEER defaults | 85.0 ± 4.5 | 824 | 85.0 ± 3.2 | 0.0 ± 3.2 |
| gsm8k | AdaDEER tuned | 84.0 ± 3.7 | 744 | 85.0 ± 3.2 | -1.0 ± 2.0 |
| gsm8k | AdaDEER-E frozen (theta=4.6, rho=1) | 85.0 ± 4.5 | 667 | 84.0 ± 3.7 | 1.0 ± 3.7 |
| gsm8k | AdaDEER-E tuned | 83.0 ± 4.0 | 684 | 84.3 ± 3.5 | -1.3 ± 1.9 |
| math500 | AdaDEER defaults | 81.0 ± 5.8 | 1,493 | 80.0 ± 4.5 | 1.0 ± 2.0 |
| math500 | AdaDEER tuned | 81.0 ± 5.8 | 1,328 | 78.0 ± 7.5 | 3.0 ± 4.0 |
| math500 | AdaDEER-E frozen (theta=4.6, rho=1) | 81.0 ± 5.8 | 1,345 | 78.0 ± 7.5 | 3.0 ± 4.0 |
| math500 | AdaDEER-E tuned | 81.0 ± 5.8 | 1,309 | 78.0 ± 7.5 | 3.0 ± 4.0 |

## Paired test over all problems (out-of-fold), AdaDEER vs DEER tau=0.95

| Dataset | Policy | dAcc 95% CI | dTokens 95% CI | AdaDEER-only correct | DEER-only correct | McNemar p |
|---|---|---|---|---|---|---|
| gsm8k | AdaDEER defaults | [+0.0, +3.0] | [+98, +186] | 1 | 0 | 1.000 |
| gsm8k | AdaDEER tuned (train) | [+0.0, +0.0] | [+32, +94] | 0 | 0 | 1.000 |
| gsm8k | AdaDEER-E frozen (theta=4.6, rho=1) | [+0.0, +3.0] | [-137, +73] | 1 | 0 | 1.000 |
| gsm8k | AdaDEER-E tuned (train) | [-3.0, +0.0] | [-90, +58] | 0 | 1 | 1.000 |
| math500 | AdaDEER defaults | [+0.0, +5.0] | [-53, +210] | 2 | 0 | 0.500 |
| math500 | AdaDEER tuned (train) | [-2.0, +6.0] | [-300, +84] | 3 | 1 | 0.625 |
| math500 | AdaDEER-E frozen (theta=4.6, rho=1) | [-2.0, +6.0] | [-273, +105] | 3 | 1 | 0.625 |
| math500 | AdaDEER-E tuned (train) | [-2.0, +6.0] | [-303, +56] | 3 | 1 | 0.625 |
| all | AdaDEER defaults | [+0.0, +3.5] | [+40, +179] | 3 | 0 | 0.250 |
| all | AdaDEER tuned (train) | [-1.0, +3.0] | [-118, +76] | 3 | 1 | 0.625 |
| all | AdaDEER-E frozen (theta=4.6, rho=1) | [-0.5, +4.0] | [-154, +61] | 4 | 1 | 0.375 |
| all | AdaDEER-E tuned (train) | [-1.5, +3.0] | [-159, +40] | 3 | 2 | 1.000 |

## MATH-500 by level (out-of-fold)

| Level | n | DEER 0.95 acc | DEER 0.95 tok | AdaDEER acc | AdaDEER tok |
|---|---|---|---|---|---|
| 1 | 11 | 81.8 | 1,329 | 81.8 | 641 |
| 2 | 25 | 92.0 | 762 | 96.0 | 853 |
| 3 | 19 | 89.5 | 788 | 94.7 | 884 |
| 4 | 22 | 86.4 | 1,273 | 86.4 | 1,400 |
| 5 | 23 | 47.8 | 2,793 | 47.8 | 2,545 |

## Hyper-parameters chosen on the training problems

```
{"split": 0, "deer_tau": 0.95, "adadeer": {"tau0": 0.97, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 3.0, "rho": 1.0}}
{"split": 1, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.1, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.05, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 2, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 3, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.1, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.05, "eta": 0.0, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 4, "deer_tau": 0.95, "adadeer": {"tau0": 0.95, "alpha": 0.1, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.1, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 3.0, "rho": 0.75}}
```
