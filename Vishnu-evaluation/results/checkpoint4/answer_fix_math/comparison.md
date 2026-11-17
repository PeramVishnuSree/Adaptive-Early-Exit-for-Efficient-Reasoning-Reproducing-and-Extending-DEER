# MATH-500 first 100: DEER answer step variants

Questions compared: 100

| Method | Accuracy | Avg reasoning tok | Avg output tok | Avg compute tok | Token reduction | Acc. delta |
|---|---|---|---|---|---|---|
| Full CoT | 66.0 ± 9.3 | 3665 | 3894 | 3894 | 0.0% | +0.0 |
| DEER t=0.95 | 69.0 ± 9.1 | 1423 | 1747 | 1763 | 55.1% | +3.0 |
| DEER t=0.95 + trial answer | 79.0 ± 8.0 | 1333 | 1446 | 1462 | 62.9% | +13.0 |
| DEER t=0.95 + boxed answer | 80.0 ± 7.8 | 1361 | 1833 | 1849 | 52.9% | +14.0 |
| Never-exit | 81.0 ± 7.7 | 2570 | 2937 | 2997 | 24.6% | +15.0 |
| Never-exit + boxed answer | 77.0 ± 8.2 | 2598 | 2988 | 3021 | 23.3% | +11.0 |

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
