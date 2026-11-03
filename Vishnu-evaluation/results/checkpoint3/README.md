# Checkpoint 3 results

All runs: DeepSeek-R1-Distill-Qwen-1.5B, first 100 questions of GSM8K test and MATH-500, 8000-token budget,
float16 on a Colab T4, DEER's strict `\boxed{}` grading.

| Folder | What | Made by |
|---|---|---|
| `never_exit/` | Never-exit trajectories (trial answer + confidence at every checkpoint) | Vishnu, `notebooks/02_checkpoint3_never_exit.ipynb` |
| `comparison_gsm8k/`, `comparison_math/` | Live comparison tables, accuracy-vs-tokens plots (with offline τ sweep), MATH per-level plots | Vishnu, `scripts/compare_runs.py` |
| `../../../Saichetan-adaptive/checkpoint_runs/outputs/ckpt/` | Live Full CoT and DEER τ = 0.95 runs | Sai Chetan |
| `../../../Saichetan-adaptive/results/checkpoint3_real/` | AdaptiveDEER-MLP trained/tested on these trajectories | Sai Chetan |

## Main results (live runs)

| | Full CoT | DEER τ = 0.95 | Never-exit (full reasoning, forced answer) |
|---|---|---|---|
| GSM8K accuracy / output tokens | 77% / 1452 | 68% / 773 (−47%) | 76% / 1123 |
| MATH-500 accuracy / output tokens | 66% / 3894 | 69% / 1747 (−55%) | 81% / 2937 |

Offline sweep at τ = 0.95 (trial answer used as the exit answer): GSM8K 83% / 365 reasoning tokens,
MATH-500 79% / 1389 reasoning tokens.

### Why the numbers differ (checked question by question)

1. **Full CoT gets no answer when it hits the budget.** It ran out of tokens mid-thought on 11 GSM8K and
   32 MATH questions and got all of them wrong. DEER's script (and so the never-exit run) forces
   `</think>` after 4800 thinking tokens and still writes an answer, which saved 16 of the 30 MATH
   budget cases. On questions where neither run hit the budget, the two agree (GSM8K 76 vs. 74 of 88
   correct, MATH 62 vs. 62 of 64), so the pipelines are consistent.
2. **Live DEER loses accuracy after it exits, not when it decides to exit.** After an early exit DEER
   appends `</think>` and lets the model write a new final answer, which often has no `\boxed{}`:

   | | Early exits | Trial answer correct | Final answer correct | Final answer not boxed |
   |---|---|---|---|---|
   | GSM8K | 31 | 27 | 13 | 17 |
   | MATH-500 | 78 | 68 | 58 | 17 |

   This is why the offline simulation (scored on the trial answer) is higher than live DEER.
   Keeping the trial answer, or forcing `\boxed{` after `</think>`, is a natural fix to test next.
3. **AdaptiveDEER-MLP is on par with DEER, not better** (held-out ROC-AUC 0.92 vs. 0.93 for DEER's raw
   confidence). On MATH the "same trial answer twice in a row" rule did slightly better than
   DEER τ = 0.95 (84.2% vs. 82.5% at a similar token count), within noise at ~31 test questions.

## `never_exit/`: never-exit trajectory runs (Person 1)

Made with `notebooks/02_checkpoint3_never_exit.ipynb` on a Colab T4.
DEER's loop with τ = 1.1 (never reached), so the model reasons to the end while a trial answer and its
confidence are logged at every "Wait" checkpoint (up to 50).

| Setting | Value |
|---|---|
| Model | DeepSeek-R1-Distill-Qwen-1.5B, float16, vLLM 0.26.0 |
| Questions | first 100 of GSM8K test, first 100 of MATH-500 |
| Budget | 8000 tokens total, `--think_ratio 0.6` (4800 thinking tokens) |
| DEER settings | `--policy avg1 --max_judge_steps 50`, trial answer at DEER's default temperature (1.0) |

| File | What |
|---|---|
| `never_<ds>.jsonl` | raw run output, with per-checkpoint logs |
| `never_<ds>.scored.jsonl` | same, plus grading, token counts and `trial_correct` per checkpoint |
| `never_<ds>.summary.json` | accuracy, tokens, exit reasons (and per-level for MATH) |
| `never_<ds>.meta.json` | run arguments and wall time |

### Results (full reasoning, DEER's strict `\boxed{}` grading)

| | GSM8K | MATH-500 |
|---|---|---|
| Accuracy | 76.0% (±8.4) | 81.0% (±7.7) |
| Avg / median reasoning tokens | 855 / 198 | 2570 / 1951 |
| Avg output tokens | 1123 | 2937 |
| Hit the thinking budget | 10 | 30 |
| Questions with ≥1 checkpoint | 36 | 97 |
| Checkpoints logged | 356 | 1050 |
| Wall time | 8.8 min | 22.4 min |

MATH accuracy by level (first 100): L1 9/11, L2 24/25, L3 16/19, L4 18/22, L5 14/23.

### Notes for the analysis
- **64 of 100 GSM8K questions never produce a "Wait"**, so DEER cannot exit early on them.
  Most GSM8K controller training data comes from the other 36 questions; MATH has far more.
- **GSM8K strict accuracy is held down by formatting**: 9 of the 24 wrong answers have the right
  number but no `\boxed{}` (or a unit like `\text{ pounds}` inside it). This is DEER's own grading rule
  (`check.py` only reads `\boxed{}`), so we keep it, but it should be mentioned.
- **Offline-simulated DEER is optimistic on GSM8K**: questions that exit early are scored on the
  trial answer, which is always boxed, so the simulated τ = 0.95 accuracy (83%) beats full reasoning (76%).
  Compare against the live DEER run before reporting simulated numbers.
- The confidence is often near 1.0 (GSM8K: 111 of 356 checkpoints ≥ 0.999; MATH: 338 of 1050).
