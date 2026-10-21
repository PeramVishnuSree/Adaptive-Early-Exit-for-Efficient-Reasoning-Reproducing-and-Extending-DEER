# How the official DEER code works (and what to watch for)

Notes from reading `vllm-deer.py`, `check.py` and `utils/` in
[iie-ycx/DEER](https://github.com/iie-ycx/DEER) (cloned 2026-10-20). Line
numbers refer to our copy `third_party/deer/vllm_deer_official.py`.

## The loop

Each question is a small state machine, and all questions are batched through vLLM together:

1. **think**: generate greedily (`temperature=0`) until the model writes `Wait`,
   `</think>` or EOS (stop strings, l. 268). `Wait` is the *transition point*.
   (`--points 0` uses `Alternatively` instead.)
2. **probe** (`needs_prob_check`): append `\n**Final Answer**\n\boxed` to the
   sequence and generate up to 20 tokens, stopping at a closing brace
   (l. 269, 415-437). This is the trial answer. Its text is **not kept** in the sequence.
3. **confidence** (l. 100-149): average max-token probability of the probe,
   from the 2nd generated token to the end. `--policy avg1` = arithmetic mean
   (used for R1-Distill), `avg2` = geometric mean (Qwen3), `min` = minimum.
4. **decide** (l. 636): if `confidence > threshold` *or* the thinking budget
   is nearly used up, go to **answer**. Otherwise re-append `Wait` and go back
   to **think**.
5. **answer**: append `\n</think>\n\n` and let the model write its final answer
   freely (l. 441-479). Note that the answer DEER is scored on is this new text,
   not the trial answer.

Budgets: total `--max_generated_tokens` (16000 in the README); thinking is
capped at `think_ratio * max_len` (0.6 for R1-Distill in the README, but the code's
default is 0.9). After `--max_judge_steps` (10) checkpoints it stops probing and
just thinks until `</think>`.

## Things that matter for our reproduction

| # | Finding | Impact / what we do |
|---|---|---|
| 1 | The trial answers and per-checkpoint confidences are **discarded**. Output only has the final text and flags (`high_prob`, `too_long`, `regular_end`, `thinking_steps`). | Our `vllm_deer_logged.py` records every checkpoint: tokens so far, trial answer, confidence, per-token probs, probe length. |
| 2 | The probe's `temperature` is commented out (l. 434) despite the "Greedy decoding" comment, so vLLM's default **1.0** applies: trial answers are *sampled*, and confidence uses the sampled token's prob. | DEER runs are not deterministic even though thinking is greedy. We keep the official behaviour by default; `--trial_temperature 0` is available as an ablation. Use ≥2 seeds/runs when comparing close numbers. |
| 3 | The probe stops on stop *strings* containing `}`. The token that closes the box is probably still in `logprobs`, and it is usually ~1.0. | Short answers (most GSM8K answers are 1–3 tokens, see data exploration) get confidence pulled toward 1. **Verify in the smoke-test logs** (`token_probs`, `n_trial_tokens`). If true, it is a strong motivation for a learned controller. |
| 4 | `check.py` crashes as written: it calls `AutoTokenizer.from_pretrained(args...)` at module level before `args` exists. | We score with `scripts/score_run.py` (same grader). |
| 5 | `check.py` matches outputs to questions by row position. | Fine for prefixes (`--limit N`). Our scripts also store `idx` and check the question text. |
| 6 | `os.environ['CUDA_VISIBLE_DEVICES']` is required (KeyError if unset). | Patched to default to `0`. |
| 7 | Token length in the paper = tokenizer length of the full response text, which includes the inserted `Wait`s and `</think>`. Probe tokens are not counted. | `output_tokens` matches the paper. `compute_tokens` adds probe tokens, which is DEER's real overhead. |
| 8 | `requirements.txt` pins `latex2sympy2==1.9.1` (needs ANTLR 4.7) together with `antlr4-python3-runtime==4.11.1` (needed by sympy's LaTeX parser). They can't both install. | We use `latex2sympy2_extended` with ANTLR 4.11. Grader check: 500/500 MATH reference solutions grade correct. |
| 9 | `bf16` default. | T4/V100 GPUs need `--dtype float16`. |
| 10 | The prompt is passed as text that already contains the BOS token, so vLLM may add a second BOS. | Same in both our Full CoT and DEER runs, so the comparison stays fair. Note it in the report. |

## Where the adaptive controller plugs in

The exit rule is one line (l. 636 in the official file, the `if state['pred_prob'] > args.threshold`
line in ours). An adaptive policy replaces it with
`controller(features_of(state['checkpoints'])) > 0.5`. Before touching the
GPU loop, policies can be evaluated offline with `deer_eval/simulate.py`
on a never-exit run (`--threshold 1.1 --max_judge_steps 50`).
