# MATH-500 first 100, R1-Distill-1.5B, 8k budget

Questions compared: 100

| Method | Accuracy | Avg reasoning tok | Avg output tok | Avg compute tok | Token reduction | Acc. delta |
|---|---|---|---|---|---|---|
| Full CoT | 66.0 ± 9.3 | 3665 | 3894 | 3894 | 0.0% | +0.0 |
| DEER t=0.95 | 69.0 ± 9.1 | 1423 | 1747 | 1763 | 55.1% | +3.0 |
| Never-exit (forced answer) | 81.0 ± 7.7 | 2570 | 2937 | 2997 | 24.6% | +15.0 |

## Offline threshold sweep (trial answer used as exit answer)

| tau | Accuracy | Avg reasoning tok |
|---|---|---|
| 0.5 | 59.0 | 630 |
| 0.6 | 59.0 | 754 |
| 0.7 | 63.0 | 854 |
| 0.8 | 69.0 | 1027 |
| 0.9 | 79.0 | 1276 |
| 0.95 | 79.0 | 1389 |
| 0.99 | 80.0 | 1467 |
