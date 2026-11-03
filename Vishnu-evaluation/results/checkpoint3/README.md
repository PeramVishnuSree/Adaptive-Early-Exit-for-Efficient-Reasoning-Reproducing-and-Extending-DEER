# Checkpoint 3 results

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
