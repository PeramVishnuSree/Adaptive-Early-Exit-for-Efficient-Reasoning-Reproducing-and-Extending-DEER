# Adaptive threshold, held-out questions

Checkpoint-3 never-exit logs (100 GSM8K, 100 MATH-500). These are not
the 1,000/500 training trajectories used by compare_stopping_methods.py.
Eight seeds, 30% of questions held out. Numeric thresholds are chosen
on the complementary 70% only. The policy family itself (a higher bar
on the first checkpoint, a lower bar once an answer repeats, and a
refusal to exit on an unfinished formula) was chosen after reading
DEER's misses on these same logs, so the held-out gap is slightly
optimistic. Δ is versus DEER τ=0.95 on the same held-out questions.
Mean ± standard deviation across seeds.

## gsm8k: Equal token cost (train tokens ≤ DEER τ=0.95)

| Method | Accuracy (%) | Output tokens | Δ acc vs DEER (points) | Δ tokens | Seeds with higher acc |
|---|---|---|---|---|---|
| DEER τ=0.95 | 84.2 ± 3.0 | 613 ± 95 | +0.0 ± 0.0 | +0 ± 0 | 0/8 |
| DEER τ=0.95 + complete | 84.2 ± 3.0 | 613 ± 95 | +0.0 ± 0.0 | +0 ± 0 | 0/8 |
| Consistency k=2 | 84.6 ± 2.5 | 606 ± 59 | +0.4 ± 2.1 | -7 ± 76 | 2/8 |
| Confidence adaptive | 84.6 ± 2.5 | 601 ± 82 | +0.4 ± 2.1 | -12 ± 63 | 2/8 |
| Closing adaptive | 85.4 ± 3.1 | 574 ± 81 | +1.2 ± 1.7 | -38 ± 62 | 3/8 |
| MLP global | 80.8 ± 8.3 | 514 ± 78 | -3.3 ± 6.7 | -99 ± 93 | 2/8 |
| MLP adaptive | 84.6 ± 2.5 | 622 ± 63 | +0.4 ± 2.1 | +10 ± 67 | 2/8 |
| Perfect stopping | 87.5 ± 4.6 | 545 ± 95 | +3.3 ± 2.5 | -68 ± 125 | 6/8 |

Closing adaptive parameters selected on train: {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 0.95, "tau_first": 0.995, "tau_stable": 0.85} ×8.

## gsm8k: Gap closing (highest train accuracy, then fewer tokens)

| Method | Accuracy (%) | Output tokens | Δ acc vs DEER (points) | Δ tokens | Seeds with higher acc |
|---|---|---|---|---|---|
| DEER τ=0.95 | 84.2 ± 3.0 | 613 ± 95 | +0.0 ± 0.0 | +0 ± 0 | 0/8 |
| DEER τ=0.95 + complete | 84.2 ± 3.0 | 613 ± 95 | +0.0 ± 0.0 | +0 ± 0 | 0/8 |
| Consistency k=2 | 84.6 ± 2.5 | 606 ± 59 | +0.4 ± 2.1 | -7 ± 76 | 2/8 |
| Confidence adaptive | 84.6 ± 2.5 | 601 ± 82 | +0.4 ± 2.1 | -12 ± 63 | 2/8 |
| Closing adaptive | 85.4 ± 3.1 | 574 ± 81 | +1.2 ± 1.7 | -38 ± 62 | 3/8 |
| MLP global | 82.9 ± 6.0 | 587 ± 103 | -1.3 ± 4.0 | -26 ± 62 | 2/8 |
| MLP adaptive | 84.6 ± 2.5 | 622 ± 63 | +0.4 ± 2.1 | +10 ± 67 | 2/8 |
| Perfect stopping | 87.5 ± 4.6 | 545 ± 95 | +3.3 ± 2.5 | -68 ± 125 | 6/8 |

Closing adaptive parameters selected on train: {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 0.95, "tau_first": 0.995, "tau_stable": 0.85} ×8.

## math: Equal token cost (train tokens ≤ DEER τ=0.95)

| Method | Accuracy (%) | Output tokens | Δ acc vs DEER (points) | Δ tokens | Seeds with higher acc |
|---|---|---|---|---|---|
| DEER τ=0.95 | 77.5 ± 7.5 | 1606 ± 350 | +0.0 ± 0.0 | +0 ± 0 | 0/8 |
| DEER τ=0.95 + complete | 78.8 ± 6.7 | 1634 ± 364 | +1.3 ± 1.7 | +27 ± 24 | 3/8 |
| Consistency k=2 | 79.2 ± 6.8 | 1518 ± 264 | +1.7 ± 3.6 | -89 ± 312 | 3/8 |
| Confidence adaptive | 78.8 ± 6.4 | 1542 ± 295 | +1.3 ± 5.0 | -65 ± 162 | 4/8 |
| Closing adaptive | 78.8 ± 6.4 | 1493 ± 318 | +1.3 ± 5.0 | -113 ± 121 | 4/8 |
| MLP global | 75.4 ± 5.0 | 1377 ± 366 | -2.1 ± 4.7 | -230 ± 310 | 0/8 |
| MLP adaptive | 78.8 ± 6.4 | 1698 ± 291 | +1.3 ± 3.5 | +92 ± 222 | 4/8 |
| Perfect stopping | 82.9 ± 6.3 | 1489 ± 314 | +5.4 ± 3.5 | -118 ± 148 | 7/8 |

Closing adaptive parameters selected on train: {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 0.95, "tau_first": 0.995, "tau_stable": 0.85} ×6, {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 0.95, "tau_first": 0.995, "tau_stable": 0.88} ×2.

## math: Gap closing (highest train accuracy, then fewer tokens)

| Method | Accuracy (%) | Output tokens | Δ acc vs DEER (points) | Δ tokens | Seeds with higher acc |
|---|---|---|---|---|---|
| DEER τ=0.95 | 77.5 ± 7.5 | 1606 ± 350 | +0.0 ± 0.0 | +0 ± 0 | 0/8 |
| DEER τ=0.95 + complete | 78.8 ± 6.7 | 1634 ± 364 | +1.3 ± 1.7 | +27 ± 24 | 3/8 |
| Consistency k=2 | 79.2 ± 6.8 | 1518 ± 264 | +1.7 ± 3.6 | -89 ± 312 | 3/8 |
| Confidence adaptive | 78.8 ± 6.4 | 1542 ± 295 | +1.3 ± 5.0 | -65 ± 162 | 4/8 |
| Closing adaptive | 77.9 ± 6.7 | 1560 ± 352 | +0.4 ± 6.0 | -47 ± 121 | 4/8 |
| MLP global | 79.2 ± 8.3 | 2038 ± 370 | +1.7 ± 2.5 | +432 ± 502 | 3/8 |
| MLP adaptive | 78.8 ± 6.4 | 1698 ± 291 | +1.3 ± 3.5 | +92 ± 222 | 4/8 |
| Perfect stopping | 82.9 ± 6.3 | 1489 ± 314 | +5.4 ± 3.5 | -118 ± 148 | 7/8 |

Closing adaptive parameters selected on train: {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 1.1, "tau_first": 0.995, "tau_stable": 0.85} ×2, {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 0.95, "tau_first": 0.995, "tau_stable": 0.85} ×1, {"min_prob_floor": 0.0, "require_complete": true, "tau_base": 0.95, "tau_first": 0.995, "tau_stable": 0.88} ×5.
