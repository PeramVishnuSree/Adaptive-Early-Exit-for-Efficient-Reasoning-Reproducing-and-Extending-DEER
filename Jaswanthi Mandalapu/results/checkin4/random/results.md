# Held-out results (random, 5 splits; mean ± std over splits)
think budget 4800, max checkpoints 10, confidence logged, tuning objective match-deer

| Policy | gsm8k Acc (%) | gsm8k Tokens | gsm8k Exits | math500 Acc (%) | math500 Tokens | math500 Exits |
|---|---|---|---|---|---|---|
| Full reasoning (never-exit) | 84.0 ± 1.3 | 1,968 | 0.0 | 78.0 ± 3.4 | 2,723 | 0.0 |
| DEER tau=0.9 | 86.0 ± 5.7 | 601 | 28.6 | 75.3 ± 1.6 | 1,262 | 26.6 |
| DEER tau=0.95 | 86.0 ± 5.7 | 663 | 27.8 | 76.7 ± 2.1 | 1,365 | 25.6 |
| DEER tau=0.99 | 86.7 ± 4.7 | 839 | 24.6 | 76.7 ± 3.0 | 1,615 | 22.0 |
| Consistency (2 in a row) | 88.7 ± 4.5 | 834 | 16.2 | 75.3 ± 3.4 | 1,195 | 20.8 |
| Consistency + momentum | 90.0 ± 4.7 | 909 | 15.8 | 77.3 ± 3.9 | 1,244 | 20.4 |
| DEER tuned tau (train) | 86.0 ± 5.7 | 663 | 27.8 | 76.7 ± 2.1 | 1,365 | 25.6 |
| AdaDEER defaults | 86.7 ± 4.7 | 780 | 24.2 | 78.0 ± 3.4 | 1,406 | 24.6 |
|   - no stability (alpha=0) | 86.7 ± 4.7 | 946 | 22.4 | 77.3 ± 3.9 | 1,767 | 19.8 |
|   - no volatility (gamma=0) | 86.7 ± 4.7 | 753 | 25.2 | 78.7 ± 4.5 | 1,320 | 25.2 |
|   - no switching (beta=0) | 86.7 ± 4.7 | 776 | 24.2 | 78.7 ± 4.5 | 1,361 | 24.8 |
|   - no budget (eta=0) | 86.7 ± 4.7 | 780 | 24.2 | 78.0 ± 3.4 | 1,410 | 24.4 |
| AdaDEER tuned (train) | 86.0 ± 5.7 | 693 | 27.0 | 77.3 ± 3.3 | 1,301 | 25.6 |
| AdaDEER-E frozen (theta=4.6, rho=1) | 86.7 ± 4.7 | 643 | 26.8 | 77.3 ± 3.3 | 1,275 | 25.6 |
| AdaDEER-E tuned (train) | 86.0 ± 5.7 | 618 | 27.4 | 77.3 ± 3.3 | 1,270 | 25.6 |

## Iso-token accuracy gain vs. the full DEER tau curve (test problems)

| Dataset | Policy | Acc | Tokens | DEER acc @ same tokens | Gain (pts) |
|---|---|---|---|---|---|
| gsm8k | AdaDEER defaults | 86.7 ± 4.7 | 780 | 87.1 ± 5.3 | -0.4 ± 1.6 |
| gsm8k | AdaDEER tuned | 86.0 ± 5.7 | 693 | 86.7 ± 6.0 | -0.7 ± 1.3 |
| gsm8k | AdaDEER-E frozen (theta=4.6, rho=1) | 86.7 ± 4.7 | 643 | 86.7 ± 6.0 | 0.0 ± 2.1 |
| gsm8k | AdaDEER-E tuned | 86.0 ± 5.7 | 618 | 86.7 ± 6.0 | -0.7 ± 1.3 |
| math500 | AdaDEER defaults | 78.0 ± 3.4 | 1,406 | 77.1 ± 2.8 | 0.9 ± 2.5 |
| math500 | AdaDEER tuned | 77.3 ± 3.3 | 1,301 | 76.3 ± 2.7 | 1.0 ± 3.3 |
| math500 | AdaDEER-E frozen (theta=4.6, rho=1) | 77.3 ± 3.3 | 1,275 | 76.3 ± 2.7 | 1.0 ± 3.3 |
| math500 | AdaDEER-E tuned | 77.3 ± 3.3 | 1,270 | 76.3 ± 2.7 | 1.0 ± 3.3 |

## Hyper-parameters chosen on the training problems

```
{"split": 0, "deer_tau": 0.95, "adadeer": {"tau0": 0.95, "alpha": 0.03, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.1, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 3.5, "rho": 1.0}}
{"split": 1, "deer_tau": 0.95, "adadeer": {"tau0": 0.95, "alpha": 0.03, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.1, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 2, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.1, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 3, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.1, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.05, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
{"split": 4, "deer_tau": 0.95, "adadeer": {"tau0": 0.99, "alpha": 0.1, "s_max": 3, "gamma": 0.0, "vol_window": 3, "beta": 0.0, "eta": 0.2, "budget_start": 0.5, "tau_min": 0.8, "tau_max": 0.999}, "evidence": {"theta": 4.6, "rho": 1.0}}
```
