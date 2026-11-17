# Stopping methods: trained on training trajectories, evaluated on test

Early exits are scored on the trial answer they stop on. Operating points are chosen on the training set at <= the tokens of DEER tau=0.95; differences vs. DEER tau=0.95 are paired bootstrap 95% CIs over test problems.

Checkpoint-level ROC-AUC on test (does the score predict a correct trial answer?): DEER confidence 0.779, LogReg 0.914, GBDT 0.810, MLP 0.912, GBDT conf-only 0.706, GBDT conf+stability 0.758

## gsm8k (100 test problems)

| Method (param chosen on train) | Acc. (%) | Output tok | Δ acc vs DEER (95% CI) | Δ tok vs DEER (95% CI) | Gap to perfect closed |
|---|---|---|---|---|---|
| Full reasoning | 76.0 | 1123 | -6.0 (-11.0, -2.0) | +495 (+280, +744) | -200% |
| Perfect stopping | 85.0 | 659 | +3.0 (+0.0, +7.0) | +31 (-159, +224) | 100% |
| DEER (0.95) | 82.0 | 628 | +0.0 (+0.0, +0.0) | +0 (+0, +0) | 0% |
| Consistency k (2, over budget) | 82.0 | 694 | +0.0 (-3.0, +3.0) | +66 (-95, +225) | 0% |
| Consistency + conf (0.0, over budget) | 82.0 | 694 | +0.0 (-3.0, +3.0) | +66 (-95, +225) | 0% |
| LogReg (0.5) | 82.0 | 573 | +0.0 (-4.0, +4.0) | -55 (-139, -8) | 0% |
| GBDT (0.4) | 83.0 | 535 | +1.0 (-3.0, +5.0) | -93 (-212, -11) | 33% |
| MLP (0.4) | 82.0 | 596 | +0.0 (-4.0, +4.0) | -31 (-191, +125) | 0% |
| GBDT conf-only (0.4) | 84.0 | 537 | +2.0 (-2.0, +6.0) | -91 (-210, -9) | 67% |
| GBDT conf+stability (0.5) | 85.0 | 552 | +3.0 (+0.0, +7.0) | -76 (-195, +5) | 100% |
