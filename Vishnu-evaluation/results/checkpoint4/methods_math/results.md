# Stopping methods: trained on training trajectories, evaluated on test

Early exits are scored on the trial answer they stop on. Operating points are chosen on the training set at <= the tokens of DEER tau=0.95; differences vs. DEER tau=0.95 are paired bootstrap 95% CIs over test problems.

Checkpoint-level ROC-AUC on test (does the score predict a correct trial answer?): DEER confidence 0.921, LogReg 0.943, GBDT 0.934, MLP 0.936, GBDT conf-only 0.918, GBDT conf+stability 0.929

## math (100 test problems)

| Method (param chosen on train) | Acc. (%) | Output tok | Δ acc vs DEER (95% CI) | Δ tok vs DEER (95% CI) | Gap to perfect closed |
|---|---|---|---|---|---|
| Full reasoning | 81.0 | 2937 | +2.0 (-3.0, +7.0) | +1405 (+1132, +1692) | 33% |
| Perfect stopping | 85.0 | 1431 | +6.0 (+2.0, +11.0) | -100 (-270, +42) | 100% |
| DEER (0.95) | 79.0 | 1531 | +0.0 (+0.0, +0.0) | +0 (+0, +0) | 0% |
| Consistency k (2, over budget) | 81.0 | 1571 | +2.0 (-2.0, +6.0) | +40 (-220, +285) | 33% |
| Consistency + conf (0.0, over budget) | 81.0 | 1571 | +2.0 (-2.0, +6.0) | +40 (-220, +285) | 33% |
| LogReg (0.7) | 79.0 | 1524 | +0.0 (+0.0, +0.0) | -7 (-34, +18) | 0% |
| GBDT (0.6) | 75.0 | 1419 | -4.0 (-9.0, +0.0) | -112 (-242, -10) | -67% |
| MLP (0.4) | 76.0 | 1387 | -3.0 (-8.0, +1.0) | -144 (-261, -54) | -50% |
| GBDT conf-only (0.6) | 75.0 | 1431 | -4.0 (-9.0, +0.0) | -100 (-213, -6) | -67% |
| GBDT conf+stability (0.6) | 77.0 | 1443 | -2.0 (-6.0, +2.0) | -89 (-218, +12) | -33% |
