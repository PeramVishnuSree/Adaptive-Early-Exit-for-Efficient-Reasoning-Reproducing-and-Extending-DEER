# AdaptiveDEER-MLP on real trajectories (held-out questions)

Mean ± std over 5 random 70/30 question splits. Accuracy uses DEER's grader; tokens are average output tokens per question. Early exits are scored on the trial answer.

Checkpoint-level ROC-AUC (predicting whether the trial answer is correct): MLP 0.920 ± 0.025, DEER confidence 0.929 ± 0.028

## GSM8K (~28 held-out questions per split)

| Policy | Accuracy (%) | Output tokens | Token reduction vs. full |
|---|---|---|---|
| Full reasoning | 79.3 ± 13.2 | 1092 ± 196 | 0% |
| DEER tau=0.8 | 83.9 ± 10.5 | 489 ± 23 | 55% |
| DEER tau=0.9 | 83.9 ± 10.5 | 498 ± 31 | 54% |
| DEER tau=0.95 | 84.5 ± 10.1 | 522 ± 42 | 52% |
| DEER tau=0.99 | 83.3 ± 11.5 | 671 ± 126 | 39% |
| Consistency (2 in a row) | 83.9 ± 10.5 | 647 ± 84 | 41% |
| Consistency + momentum | 83.9 ± 10.5 | 669 ± 86 | 39% |
| MLP p>=0.3 | 84.4 ± 8.6 | 486 ± 28 | 55% |
| MLP p>=0.4 | 85.1 ± 9.1 | 551 ± 71 | 49% |
| MLP p>=0.5 | 85.1 ± 9.1 | 561 ± 72 | 49% |
| MLP p>=0.62 | 84.5 ± 10.1 | 593 ± 86 | 46% |
| MLP p>=0.7 | 84.5 ± 10.1 | 642 ± 121 | 41% |
| MLP p>=0.8 | 84.5 ± 10.1 | 772 ± 164 | 29% |
| MLP p>=0.9 | 84.5 ± 10.1 | 851 ± 200 | 22% |

## MATH (~31 held-out questions per split)

| Policy | Accuracy (%) | Output tokens | Token reduction vs. full |
|---|---|---|---|
| Full reasoning | 83.4 ± 2.7 | 2969 ± 200 | 0% |
| DEER tau=0.8 | 70.2 ± 6.3 | 1102 ± 194 | 63% |
| DEER tau=0.9 | 82.1 ± 5.6 | 1353 ± 178 | 54% |
| DEER tau=0.95 | 82.5 ± 4.9 | 1454 ± 147 | 51% |
| DEER tau=0.99 | 83.1 ± 4.1 | 1520 ± 179 | 49% |
| Consistency (2 in a row) | 84.2 ± 3.8 | 1430 ± 174 | 52% |
| Consistency + momentum | 83.5 ± 4.0 | 1495 ± 174 | 50% |
| MLP p>=0.3 | 71.1 ± 4.5 | 992 ± 160 | 67% |
| MLP p>=0.4 | 76.4 ± 3.6 | 1207 ± 174 | 59% |
| MLP p>=0.5 | 81.1 ± 3.5 | 1425 ± 190 | 52% |
| MLP p>=0.62 | 83.8 ± 4.2 | 1623 ± 126 | 45% |
| MLP p>=0.7 | 85.0 ± 4.1 | 1804 ± 152 | 39% |
| MLP p>=0.8 | 85.0 ± 4.1 | 2011 ± 185 | 32% |
| MLP p>=0.9 | 85.0 ± 4.6 | 2290 ± 127 | 23% |
