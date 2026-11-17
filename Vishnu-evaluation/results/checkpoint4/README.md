# Checkpoint 4 results

Same model and settings as checkpoint 3: DeepSeek-R1-Distill-Qwen-1.5B, float16, vLLM 0.26 on a Colab T4,
8,000-token budget, `--think_ratio 0.6 --policy avg1`, DEER's strict `\boxed{}` grading. Test sets: first 100
problems of GSM8K test and MATH-500. Runs from `notebooks/04_training_trajectories_and_answer_fix.ipynb`
(raw outputs are in the team Drive, `deer_final/`).

## 1. Fixing DEER's answer step (live runs) — `answer_fix_<dataset>/`

After an early exit, official DEER appends `</think>` and lets the model write a new answer, which often has no
`\boxed{}` (checkpoint 3). Two fixes, run live with `--final_answer`:
- **trial**: keep the trial answer DEER stopped on (no extra generation)
- **boxed**: the final answer is prompted to start with `**Final Answer**\n\boxed{`

| | Full CoT | DEER τ=0.95 | DEER + trial | DEER + boxed | Never-exit | Never-exit + boxed |
|---|---|---|---|---|---|---|
| GSM8K acc / output tok | 77% / 1452 | 68% / 773 | **82% / 592** | 80% / 556 | 76% / 1123 | 81% / 1058 |
| MATH-500 acc / output tok | 66% / 3894 | 69% / 1747 | **79% / 1446** | 80% / 1833 | 81% / 2937 | 77% / 2988 |

Paired tests on the same 100 problems (McNemar exact test for accuracy, bootstrap 95% CI for tokens):

| Comparison | GSM8K | MATH-500 |
|---|---|---|
| DEER + trial vs. DEER | +14 pts (p = 0.001), −181 tok (−381, −13) | +10 pts (p = 0.041), −301 tok (−489, −116) |
| DEER + boxed vs. DEER | +12 pts (p = 0.023), −217 tok (−378, −94) | +11 pts (p = 0.043), +86 tok (−126, +332) |
| DEER + trial vs. Full CoT | +5 pts (p = 0.13), −860 tok (−1288, −475) | +13 pts (p = 0.004), −2448 tok (−2961, −1943) |
| DEER + trial vs. Never-exit | +6 pts (p = 0.07), −531 tok | −2 pts (p = 0.73), −1491 tok |

- Keeping the trial answer recovers DEER's lost accuracy on both datasets and saves tokens, since no new
  answer is generated. It matches full-reasoning accuracy with about half the tokens.
- The offline simulator predicted exactly these accuracies for "DEER + trial" (82% and 79%, checkpoint 3).
- **Run-to-run variation:** two runs with identical greedy settings (Never-exit vs. Never-exit + boxed) produced
  identical reasoning on only 82/100 GSM8K and 52/100 MATH problems (vLLM float16 batching is not bit-exact).
  Single-run accuracy differences of about ±4 points are within noise (e.g. Never-exit vs. Never-exit + boxed on
  MATH: 81% vs. 77%, p = 0.29).

## 2. Stopping methods trained on separate training data — `methods_gsm8k/`

Training data: never-exit trajectories on 1,000 GSM8K **train** problems (3,573 checkpoints from 368 problems
that reach at least one "Wait"). Test: the 100 GSM8K test problems from checkpoint 3 (356 checkpoints from
36 problems). Made with `scripts/compare_stopping_methods.py`; early exits are scored on the trial answer; each
method's threshold is chosen on the training set at ≤ the tokens of DEER τ=0.95 (10-check cap, as in DEER).

Checkpoint-level ROC-AUC on test (predicting whether a trial answer is correct): DEER confidence 0.779,
logistic regression 0.914, MLP 0.912, gradient boosting 0.810.

| Method (threshold chosen on train) | Acc. | Output tok | Δ acc vs DEER (95% CI) | Δ tok vs DEER (95% CI) |
|---|---|---|---|---|
| Full reasoning | 76% | 1123 | −6 (−11, −2) | +495 |
| **Perfect stopping (upper bound)** | 85% | 659 | +3 (0, +7) | +31 |
| DEER τ=0.95 | 82% | 628 | — | — |
| Consistency (2 in a row) | 82% | 694 | 0 (−3, +3) | +66 (−95, +225) |
| Logistic regression | 82% | 573 | 0 (−4, +4) | **−55 (−139, −8)** |
| MLP | 82% | 596 | 0 (−4, +4) | −31 (−191, +125) |
| Gradient boosting | 83% | 535 | +1 (−3, +5) | **−93 (−212, −11)** |
| Gradient boosting, confidence features only | 84% | 537 | +2 (−2, +6) | **−91 (−210, −9)** |
| Gradient boosting, confidence + answer stability | 85% | 552 | +3 (0, +7) | −76 (−195, +5) |

- Learned scores predict trial-answer correctness much better than DEER's raw confidence (AUC 0.91 vs. 0.78).
- At the policy level, the learned methods match DEER's accuracy with significantly fewer tokens (logistic
  regression, gradient boosting); accuracy gains are not significant.
- On GSM8K the room for any stopping rule is small (DEER 82% vs. the perfect-stopping bound 85%), and only 36 of
  the 100 test problems ever reach a checkpoint, so the test has little power. The full GSM8K test set is needed.

MATH-500 methods comparison: pending (MATH training run in progress).
