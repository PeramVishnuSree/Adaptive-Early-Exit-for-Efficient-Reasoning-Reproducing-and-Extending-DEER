# Held-out results (random, 5 splits; mean ± std over splits)
think budget 4800, max checkpoints 10, confidence logged, tuning objective match-deer

| Policy | math500 Acc (%) | math500 Tokens | math500 Exits |
|---|---|---|---|
| Full reasoning (never-exit) | 78.7 ± 2.4 | 3,306 | 0.0 |
| DEER tau=0.9 | 74.1 ± 2.2 | 1,507 | 129.6 |
| DEER tau=0.95 | 75.1 ± 1.7 | 1,671 | 125.2 |
| DEER tau=0.99 | 78.0 ± 1.8 | 1,962 | 113.2 |
| Consistency (2 in a row) | 74.3 ± 3.1 | 1,277 | 112.4 |
| Consistency + momentum | 75.3 ± 2.3 | 1,468 | 107.2 |
| DEER tuned tau (train) | 75.1 ± 1.7 | 1,671 | 125.2 |
| AdaDEER defaults | 79.5 ± 2.0 | 1,783 | 119.4 |
|   - no stability (alpha=0) | 79.5 ± 2.5 | 2,179 | 99.6 |
|   - no volatility (gamma=0) | 79.5 ± 2.0 | 1,695 | 122.6 |
|   - no switching (beta=0) | 79.5 ± 2.0 | 1,777 | 119.8 |
|   - no budget (eta=0) | 79.5 ± 2.0 | 1,792 | 118.0 |
| AdaDEER tuned (train) | 79.3 ± 1.8 | 1,697 | 122.8 |
| AdaDEER-E frozen (theta=4.6, rho=1) | 79.1 ± 2.3 | 1,547 | 127.6 |
| AdaDEER-E tuned (train) | 78.9 ± 1.9 | 1,644 | 123.6 |

## Iso-token accuracy gain vs. the full DEER tau curve (test problems)

| Dataset | Policy | Acc | Tokens | DEER acc @ same tokens | Gain (pts) |
|---|---|---|---|---|---|
| math500 | AdaDEER defaults | 79.5 ± 2.0 | 1,783 | 76.4 ± 1.7 | 3.1 ± 0.6 |
| math500 | AdaDEER tuned | 79.3 ± 1.8 | 1,697 | 75.7 ± 1.8 | 3.6 ± 0.8 |
| math500 | AdaDEER-E frozen (theta=4.6, rho=1) | 79.1 ± 2.3 | 1,547 | 74.4 ± 2.0 | 4.7 ± 1.1 |
| math500 | AdaDEER-E tuned | 78.9 ± 1.9 | 1,644 | 75.0 ± 1.5 | 3.9 ± 0.7 |

## Hyper-parameters chosen on the training problems

```
{"split": 0, "deer_tau": 0.95, "adadeer": {"tau0": 0.995, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.1, "eta": 0.1, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 0.75}}
{"split": 1, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.9, "tau_max": 0.999}, "evidence": {"theta": 5.5, "rho": 1.0}}
{"split": 2, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.9, "tau_max": 0.999}, "evidence": {"theta": 5.0, "rho": 1.0}}
{"split": 3, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.0, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 0.75}}
{"split": 4, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.06, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.9, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 0.75}}
```
