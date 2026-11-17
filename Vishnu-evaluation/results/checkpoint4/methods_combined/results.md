# Stopping methods: trained on training trajectories, evaluated on test

Early exits are scored on the trial answer they stop on. Operating points are chosen on the training set at <= the tokens of DEER tau=0.95; differences vs. DEER tau=0.95 are paired bootstrap 95% CIs over test problems.

Checkpoint-level ROC-AUC on test (does the score predict a correct trial answer?): DEER confidence 0.896, LogReg 0.930, GBDT 0.926, MLP 0.932, GBDT conf-only 0.893, GBDT conf+stability 0.917

## gsm8k (100 test problems)

| Method (param chosen on train) | Acc. (%) | Output tok | Δ acc vs DEER (95% CI) | Δ tok vs DEER (95% CI) | Gap to perfect closed |
|---|---|---|---|---|---|
| Full reasoning | 76.0 | 1123 | -6.0 (-11.0, -2.0) | +495 (+280, +744) | -200% |
| Perfect stopping | 85.0 | 659 | +3.0 (+0.0, +7.0) | +31 (-159, +224) | 100% |
| DEER (0.95) | 82.0 | 628 | +0.0 (+0.0, +0.0) | +0 (+0, +0) | 0% |
| Consistency k (2, over budget) | 82.0 | 694 | +0.0 (-3.0, +3.0) | +66 (-95, +225) | 0% |
| Consistency + conf (0.0, over budget) | 82.0 | 694 | +0.0 (-3.0, +3.0) | +66 (-95, +225) | 0% |
| LogReg (0.6) | 83.0 | 577 | +1.0 (-2.0, +4.0) | -51 (-129, -6) | 33% |
| GBDT (0.6) | 83.0 | 587 | +1.0 (-2.0, +4.0) | -41 (-120, +3) | 33% |
| MLP (0.4) | 82.0 | 624 | +0.0 (-4.0, +4.0) | -4 (-135, +147) | 0% |
| GBDT conf-only (0.6) | 82.0 | 616 | +0.0 (-3.0, +3.0) | -12 (-24, -3) | 0% |
| GBDT conf+stability (0.5) | 83.0 | 580 | +1.0 (-2.0, +4.0) | -48 (-129, -5) | 33% |

## math (100 test problems)

| Method (param chosen on train) | Acc. (%) | Output tok | Δ acc vs DEER (95% CI) | Δ tok vs DEER (95% CI) | Gap to perfect closed |
|---|---|---|---|---|---|
| Full reasoning | 81.0 | 2937 | +2.0 (-3.0, +7.0) | +1405 (+1132, +1692) | 33% |
| Perfect stopping | 85.0 | 1431 | +6.0 (+2.0, +11.0) | -100 (-270, +42) | 100% |
| DEER (0.95) | 79.0 | 1531 | +0.0 (+0.0, +0.0) | +0 (+0, +0) | 0% |
| Consistency k (2, over budget) | 81.0 | 1571 | +2.0 (-2.0, +6.0) | +40 (-220, +285) | 33% |
| Consistency + conf (0.0, over budget) | 81.0 | 1571 | +2.0 (-2.0, +6.0) | +40 (-220, +285) | 33% |
| LogReg (0.6) | 78.0 | 1493 | -1.0 (-3.0, +0.0) | -38 (-89, +1) | -17% |
| GBDT (0.6) | 78.0 | 1535 | -1.0 (-4.0, +2.0) | +4 (-31, +39) | -17% |
| MLP (0.4) | 73.0 | 1295 | -6.0 (-12.0, -1.0) | -236 (-406, -93) | -100% |
| GBDT conf-only (0.6) | 79.0 | 1536 | +0.0 (+0.0, +0.0) | +5 (-21, +32) | 0% |
| GBDT conf+stability (0.5) | 76.0 | 1348 | -3.0 (-8.0, +1.0) | -183 (-326, -61) | -50% |
