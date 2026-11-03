# GSM8K first 100, R1-Distill-1.5B, 8k budget

Questions compared: 100

| Method | Accuracy | Avg reasoning tok | Avg output tok | Avg compute tok | Token reduction | Acc. delta |
|---|---|---|---|---|---|---|
| Full CoT | 77.0 ± 8.2 | 1229 | 1452 | 1452 | 0.0% | +0.0 |
| DEER t=0.95 | 68.0 ± 9.1 | 442 | 773 | 778 | 46.8% | -9.0 |
| Never-exit (forced answer) | 76.0 ± 8.4 | 855 | 1123 | 1141 | 22.7% | -1.0 |

## Offline threshold sweep (trial answer used as exit answer)

| tau | Accuracy | Avg reasoning tok |
|---|---|---|
| 0.5 | 78.0 | 281 |
| 0.6 | 79.0 | 302 |
| 0.7 | 80.0 | 313 |
| 0.8 | 81.0 | 320 |
| 0.9 | 82.0 | 343 |
| 0.95 | 83.0 | 365 |
| 0.99 | 82.0 | 486 |
