# Checkpoint 4 — state-dependent threshold (Sai Chetan Muppalla)

Author: Sai Chetan Muppalla (018310504), saichetan.muppalla@sjsu.edu.

This note is my Check-In 4 addition. It replays Vishnu's checkpoint-3 never-exit logs with a stopping rule that changes the confidence bar as the trial answer changes. No new GPU run. The numbers below are from `scripts/eval_adaptive_threshold.py`.

## What changed since Check-In 3

Check-In 3 (`results/checkpoint3_real/`) trained AdaptiveDEER-MLP on these same logs. The MLP did not beat raw confidence: checkpoint ROC-AUC 0.920 ± 0.025 versus 0.929 ± 0.028 for confidence. On MATH, at a similar token count, the MLP was below DEER τ=0.95.

Check-In 4 stops using the MLP as the improvement. The misses left between DEER τ=0.95 and perfect stopping are three kinds of checkpoint:

- a wrong trial answer on the first checkpoint with confidence near 1
- a repeated correct answer sitting just under 0.95
- a cut-off formula, such as an unclosed `\dfrac{...`, that the model is sure about

The rule in `adaptive_exit/adaptive_threshold.py` targets those three cases. The training fold of every GSM8K seed, and 6 of 8 MATH seeds, selected the same cutoffs:

| State of the trial answer | Exit if confidence is above |
|---|---|
| First checkpoint | 0.995 |
| A later answer that has not repeated | 0.95 |
| The same nonempty answer as the previous checkpoint | 0.85 |
| Unfinished answer (unbalanced brackets, open `\begin`, trailing `\`) | never |

Early exits are graded on the trial answer, with DEER's cap of 10 checks, through Vishnu's `deer_eval.simulate.simulate_policy`.

## Protocol

The 1,000 GSM8K and 500 MATH training trajectories used in `Vishnu-evaluation/scripts/compare_stopping_methods.py` are not in this checkout. This run uses the checkpoint-3 never-exit logs only: the first 100 GSM8K test problems and the first 100 MATH-500 problems.

Eight seeds. Each seed holds out 30% of questions, stratified on whether any checkpoint exists. Numeric thresholds are chosen on the other 70% only, then frozen. The equal-cost rule keeps the training candidate with the highest accuracy whose output tokens do not exceed DEER τ=0.95. That is the same token constraint Vishnu uses. A second rule, highest training accuracy with no token cap, is reported only as a check. On MATH it overfits the 70-question fold.

The shape of the rule was chosen after reading DEER's misses on these same 200 questions, so the held-out gain is a little optimistic. Compare each method with DEER on the same fold. The Check-In 3 MATH figure of 82.5% and the Check-In 4 figure of 77.5% are different held-out draws, not a change in DEER.

## Held-out results (equal token cost)

Mean ± standard deviation across 8 seeds. Delta is versus DEER τ=0.95 on the same held-out questions.

### GSM8K (30 held-out questions per seed)

| Method | Accuracy (%) | Output tokens | Δ acc (points) | Δ tokens | Seeds with higher acc |
|---|---|---|---|---|---|
| DEER τ=0.95 | 84.2 ± 3.0 | 613 ± 95 | 0 | 0 | 0/8 |
| Consistency, 2 in a row | 84.6 ± 2.5 | 606 ± 59 | +0.4 ± 2.1 | −7 ± 76 | 2/8 |
| Closing adaptive | 85.4 ± 3.1 | 574 ± 81 | +1.2 ± 1.7 | −38 ± 62 | 3/8 |
| MLP, one global threshold | 80.8 ± 8.3 | 514 ± 78 | −3.3 ± 6.7 | −99 ± 93 | 2/8 |
| Perfect stopping | 87.5 ± 4.6 | 545 ± 95 | +3.3 ± 2.5 | −68 ± 125 | 6/8 |

GSM8K seed by seed for the closing rule: +3.3 points on seeds 0, 1, and 7 (one question each, and about 110 fewer tokens), and a tie on the other five. It is never worse. That is 1.2 of the 3.3 points between DEER and perfect stopping.

### MATH-500 (30 held-out questions per seed)

| Method | Accuracy (%) | Output tokens | Δ acc (points) | Δ tokens | Seeds with higher acc |
|---|---|---|---|---|---|
| DEER τ=0.95 | 77.5 ± 7.5 | 1606 ± 350 | 0 | 0 | 0/8 |
| DEER τ=0.95, refuse an unfinished answer | 78.8 ± 6.7 | 1634 ± 364 | +1.3 ± 1.7 | +27 ± 24 | 3/8 |
| Consistency, 2 in a row | 79.2 ± 6.8 | 1518 ± 264 | +1.7 ± 3.6 | −89 ± 312 | 3/8 |
| Closing adaptive | 78.8 ± 6.4 | 1493 ± 318 | +1.3 ± 5.0 | −113 ± 121 | 4/8 |
| MLP, one global threshold | 75.4 ± 5.0 | 1377 ± 366 | −2.1 ± 4.7 | −230 ± 310 | 0/8 |
| Perfect stopping | 82.9 ± 6.3 | 1489 ± 314 | +5.4 ± 3.5 | −118 ± 148 | 7/8 |

MATH is not a stable accuracy win. Four seeds gain and two lose (seed 3 is −3.3 points, seed 7 is −6.7). The mean of +1.3 is about a quarter of the 5.4 point gap to perfect stopping, and the token count lands on the perfect-stopping budget. The completeness check alone accounts for DEER's own +1.3 points on MATH: high confidence on a cut-off formula was a premature exit.

Removing the token cap and taking the highest training accuracy makes MATH worse (+0.4 ± 6.0). Two seeds then pick a rule that never exits unless the answer repeats, and those seeds drop 3 and 10 points. The equal-cost constraint is the one to report.

Full tables, including the MLP with the same three bars, are in `results/adaptive_threshold_heldout/results.md`.

## Where this sits next to Vishnu's Check-In 4

Vishnu's live result is a different lever. Keeping the trial answer, instead of asking the model to rewrite it, is +14 points on GSM8K (p = 0.001) and +10 points on MATH (p = 0.041), with fewer tokens (`Vishnu-evaluation/results/checkpoint4/`). This note does not repeat that experiment.

Vishnu's stopping comparison, trained on separate problems and tested once on these 100, is the comparison for this rule. Learned scores match DEER and save tokens on GSM8K. On MATH they save tokens by losing accuracy, and "the same answer twice" is the only method above DEER's curve. My Check-In 4 takes that finding and adds the two cases that rule still misses: do not trust the first spike, and do not exit on a broken formula, while a repeated answer may exit below 0.95.

The evidence here is thinner than that comparison. Thresholds are frozen inside each fold, which matches the selection rule, but the family was designed on these logs, the separate training trajectories were not used, and there is no paired confidence interval. The supported claim is the GSM8K one: inside DEER's token budget the closing rule is never worse, it is higher on 3 of 8 seeds, and it uses fewer tokens. MATH still depends on which 30 questions are held out.

## Reproduce

From `Saichetan-adaptive`, with the project venv:

```bash
pytest tests/test_adaptive_threshold.py
python scripts/eval_adaptive_threshold.py
```

The script reads `Vishnu-evaluation/results/checkpoint3/never_exit/never_{gsm8k,math}.scored.jsonl` and rewrites `results/adaptive_threshold_heldout/`.
