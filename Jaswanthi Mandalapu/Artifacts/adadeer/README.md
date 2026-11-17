# AdaDEER-E: the variant that beats DEER (measured)

**Rule** (replaces DEER's `c_k > 0.95`; everything else unchanged):

```
z_k = logit(c_k)
E_k = z_k + ρ·E_{k-1}  if the trial answer equals the previous one, else z_k
exit when E_k > θ          (frozen: θ = 4.6 = logit(0.99), ρ = 1)
```
Equivalently, it uses an adaptive threshold τ_k = sigmoid(θ − ρ·E_{k−1}):
* a new answer needs c > 0.99;
* a repeated, confident answer needs much less;
* the threshold resets when the answer changes.

With ρ = 0 it is exactly DEER. θ and ρ were fixed on dev problems (MATH-500 #100–131,
GSM8K #100–147) **before** the evaluation problems were scored.

**Setup:** R1-Distill-Qwen-1.5B (bf16, MLX, greedy), 8k budget, thinking budget 4,800, ≤ 10 checkpoints,
"Wait" monitor, avg1 confidence, early exits scored on the trial answer. The evaluation set is the first 100 GSM8K and
first 100 MATH-500 problems. Trajectories are in `results/traj/`, and the full tables are in `results/{random,kfold}/results.md`.

| All 100 problems, frozen settings | GSM8K acc | GSM8K tok | MATH-500 acc | MATH-500 tok |
|---|---|---|---|---|
| Full reasoning (never-exit) | 81.0 | 2,017 | 81.0 | 2,856 |
| DEER τ = 0.95 | 84.0 | 683 | 79.0 | 1,409 |
| DEER τ = 0.99 | 85.0 | 846 | 79.0 | 1,670 |
| **AdaDEER-E** | **85.0** | **667** | **81.0** | **1,345** |

On the Check-In 3 Table III protocol (5 × 70/30 held-out splits), AdaDEER-E gets:
* GSM8K: 86.7% at 643 tokens, vs DEER 0.95 at 86.0% / 663.
* MATH-500: 77.3% at 1,275 tokens, vs DEER 0.95 at 76.7% / 1,365.

Its accuracy-vs-tokens curve lies at or above DEER's curve (`results/random/tradeoff.png`).

## Full MATH-500 (all 500 problems)

Same frozen settings (θ = 4.6, ρ = 1), with no retuning. Trajectories are in `results/traj/full_math500.jsonl`, and the
tables and plots are in `results/math500_full_{random,kfold}/`.

| All 500 problems | Acc | Tokens | Exits |
|---|---|---|---|
| Full reasoning (never-exit) | 79.0 | 3,209 | 0 |
| DEER τ = 0.90 | 75.0 | 1,438 | 435 |
| DEER τ = 0.95 | 76.6 | 1,623 | 416 |
| DEER τ = 0.99 | 79.0 | 1,892 | 379 |
| AdaDEER defaults | 80.0 | 1,694 | 401 |
| **AdaDEER-E** | **79.6** | **1,488** | 426 |

* **vs DEER 0.95 (paired):**
  * AdaDEER-E is right where DEER is wrong on 19 problems; the reverse happens on 4. McNemar **p = 0.003**.
  * Accuracy difference: 95% CI [+1.2, +5.0] points.
  * Token difference: 95% CI [−236, −44] per problem.
  * So AdaDEER-E is **more accurate and cheaper**, significantly.
* **vs DEER 0.99:** accuracy is the same (p = 0.69), with 26% fewer tokens (95% CI [−521, −294]).
* **Matches or beats full reasoning** at 46% of its tokens.
* **Excluding the 32 dev problems (#100–131)** used to pick θ and ρ: 79.9 / 1,448 vs DEER 0.95 at 76.7 / 1,569 (same 19 vs 4 wins/losses, p = 0.003).
* **Held-out 5 × 70/30 splits:** AdaDEER-E 79.1% / 1,547 vs DEER 0.95 at 75.1% / 1,671. Iso-token gain over the DEER τ curve: +4.7 ± 1.1 points.
* **By level (acc / tokens):**

| Level | DEER 0.95 | DEER 0.99 | AdaDEER-E |
|---|---|---|---|
| 1 (n=43) | 90.7 / 769 | 93.0 / 853 | 93.0 / 632 |
| 2 (n=90) | 90.0 / 883 | 93.3 / 1,051 | 92.2 / 869 |
| 3 (n=105) | 85.7 / 1,194 | 86.7 / 1,374 | 90.5 / 1,147 |
| 4 (n=128) | 75.8 / 1,596 | 78.9 / 1,982 | 78.9 / 1,587 |
| 5 (n=134) | 56.7 / 2,756 | 59.0 / 3,112 | 59.0 / 2,352 |

**Limitation:** on the 100-problem sets alone the gains were small. Out of fold, AdaDEER-E is right where DEER is wrong on 4 problems and the
reverse on 1 (McNemar p = 0.375), so with 100 problems per dataset the improvement is consistent
but not statistically significant. The prompt puts the instruction in the user turn, so GSM8K
produces far more "Wait" checkpoints than in Check-In 3, and absolute GSM8K numbers differ from it.
The MATH-500 baselines match Check-In 3: 81% never-exit and 79% DEER.

# AdaDEER: adaptive-threshold early exit for reasoning models

DEER (Yang et al., ICLR 2026) stops reasoning at a transition point ("Wait") when the
confidence of an induced trial answer exceeds one global threshold, `c_k > λ` with `λ = 0.95`.
AdaDEER keeps everything else in DEER the same (transition monitor, answer inducer and
confidence) and changes **one line**: the fixed threshold becomes a threshold `τ_k` computed
per question and per checkpoint from that question's own trajectory so far.

## 1. Why a single threshold is suboptimal (evidence)

| Observation | Source |
|---|---|
| The best exit position differs from problem to problem and depends on difficulty; a fixed rule is "suboptimal". | DEER §2, Fig. 2 |
| Confidence is U-shaped (polarized). The band 0.90–0.97 is an "error-prone gray zone". | DEER App. J |
| With R1-Distill-1.5B, trial answers with confidence 0.95–0.99 are correct only 54.9% (GSM8K) and 63.1% (MATH-500) of the time. That is worse than the 0.90–0.95 bin. τ = 0.95 accepts exactly these answers. | Check-In 3 §V-D5 |
| Answer repetition is a strong, cheap signal. "Consistency (2 in a row)" beats DEER τ = 0.95 on held-out MATH-500 (84.2 vs 82.5, similar tokens). | Check-In 3, Table III |
| Hard problems rarely cross τ early, and Full CoT is cut off at the budget on 32/100 MATH-500 problems. Exiting on a good trial answer beats being truncated. | Check-In 3 §V-B3/4 |
| 87% of live DEER exits stop on a correct trial answer, but the *regenerated* final answer is often unboxed. | Check-In 3 §V-D4 |

## 2. The algorithm

At checkpoint `k` of question `x`, let `ŷ_k` be the trial answer, `c_k` its DEER confidence
(avg1 or geometric mean, unchanged from DEER), and `L_k` the number of reasoning tokens so far. AdaDEER then computes four signals from checkpoints `1..k` only:

* **Stability** `s_k`: length of the current run of identical (normalized) trial answers.
* **Volatility** `v_k`: mean |Δc| over the last 3 checkpoints. This separates a steady rise from a
  single spike after large swings.
* **Switching rate** `u_k`: (number of answer changes) / (k − 1). This is an online difficulty
  estimate: a model that keeps changing its answer is struggling with that question.
* **Budget pressure** `b_k = clip((L_k/B − ω)/(1 − ω), 0, 1)`, where `B` is the thinking budget
  (`think_ratio × max_len`) and `ω = 0.5`.

```
τ_k = clip( τ0 − α·min(s_k − 1, s_max) + γ·v_k + β·u_k − η·b_k ,  τ_min, τ_max )
exit at the first k with  ŷ_k parseable  and  c_k > τ_k
```

**Defaults:** τ0 = 0.99, α = 0.06, s_max = 3, γ = 0.5, β = 0.05, η = 0.10, τ_min = 0.85, τ_max = 0.999.
These were chosen from the evidence above *before* looking at any held-out result:

* A **new** answer must clear 0.99, which sits outside the gray zone.
* An answer seen 2× needs > 0.93, and one seen 3× needs > 0.87.
* A jumpy or flip-flopping trajectory pushes the bar back toward 0.999.
* Near the thinking budget, the bar drops by up to 0.10, because the alternative is a forced `</think>`.

**Properties:**

* With α = γ = β = η = 0 and τ_min ≤ τ0, AdaDEER is exactly DEER with λ = τ0. This is verified on every
  problem by the replay code, and it makes AdaDEER a strict generalization of DEER.
* Its extra cost per checkpoint is a few arithmetic operations. It needs no training, no extra probes and no
  extra forward passes.

**Answer finalization (recommended for live runs, applied equally to DEER):** at exit, keep the
trial answer, which is already inside `\boxed{…}`. Alternatively, append `</think>\n\n**Final Answer**\n\boxed{` instead of
plain `</think>`. Report DEER with the same fix, so that the comparison isolates the threshold.

```
Algorithm: AdaDEER (differences from DEER marked ◆)
Input: model f, question x, inducer I = "\n**Final Answer**\n\boxed", think budget B, max checkpoints K
 1  R ← [], prev ← ∅, s ← 0, switches ← 0, C ← []
 2  while reasoning not finished and |R| < B:
 3      generate tokens into R until the next "Wait" (or </think>)
 4      if a "Wait" was reached and checkpoint index k ≤ K:
 5          ŷ_k, p_k ← greedy probe f(x, R + I)              # ≤ 20 tokens, rewound afterwards
 6          c_k ← avg1(p_k)                                    # unchanged from DEER
 7  ◆       s ← s+1 if norm(ŷ_k) = prev else 1;  switches += [k>1 and norm(ŷ_k) ≠ prev]
 8  ◆       v ← mean|Δc| over last 3 C;  u ← switches/(k−1);  b ← clip((|R|/B − 0.5)/0.5, 0, 1)
 9  ◆       τ_k ← clip(τ0 − α·min(s−1, 3) + γ·v + β·u − η·b, τ_min, τ_max)
10  ◆       if ŷ_k parseable and c_k > τ_k:  return \boxed{ŷ_k}   # keep the trial answer
11          prev ← norm(ŷ_k); C.append(c_k)
12  append </think>; generate and return the conclusion     # same as DEER
```

```python
policy = AdaDEER()                       # or FixedThresholdDEER(0.95)
policy.reset(think_budget=0.6 * max_len)
# inside DEER's loop, at every "Wait" checkpoint, replace  `if conf > threshold:`  with
if policy.should_exit(trial_answer_text, conf, reasoning_tokens_so_far):
    ...  # exit exactly as DEER does (+ finalization fix)
```

## 3. Experiments (same as Check-In 3 / DEER)

* **Model and setup:** DeepSeek-R1-Distill-Qwen-1.5B with greedy reasoning, "Wait" monitor, avg1 confidence, ≤ 10 checkpoints, think_ratio 0.6, an 8k budget (thinking budget 4,800), on the first 100 GSM8K and MATH-500 problems. Use `--think-budget 9830` for the paper's 16k setting.
* **Offline replay on the never-exit trajectories:** this uses the same simulator semantics as Check-In 3. An exit is scored on the trial answer at
  the exit checkpoint, and runs that never exit receive the never-exit outcome.
* **Protocol `random`:** 5 × 70/30 splits by problem, i.e. Table III. **Protocol `kfold`:** 5-fold, so every problem is
  tested exactly once. This adds a paired bootstrap and a McNemar test against DEER τ = 0.95 over all 200 problems.
* **Tuned vs tuned:** DEER's λ and AdaDEER's coefficients are each chosen on the training split with the
  *same* objective. The objective is either `match-deer` (most accurate policy using no more tokens than DEER 0.95) or `tradeoff`.
  The AdaDEER grid contains DEER itself.
* **Reported:**
  * Table III-style accuracy, tokens and exits.
  * An iso-token gain against the whole DEER τ curve (Pareto envelope).
  * Component ablations.
  * Per-level MATH-500 results.
  * The accuracy–token plot (`tradeoff.png`).

```bash
python -m adadeer.experiments --data gsm8k_never_exit.jsonl math500_never_exit.jsonl \
       --out results/random --protocol random
python -m adadeer.experiments --data gsm8k_never_exit.jsonl math500_never_exit.jsonl \
       --out results/kfold  --protocol kfold
```

The input format (one problem per JSON line) is documented at the top of `replay.py`. Before
trusting any number, check that `DEER tau=0.95` reproduces Check-In 3. Using your own split seeds,
it should give 84.5% / 522 tokens on GSM8K and 82.5% / 1,454 tokens on MATH-500.

**What "better than DEER" means here:** higher held-out accuracy at no more tokens than DEER τ = 0.95,
or a positive iso-token gain against the DEER curve, consistently across splits. With ~30 test
problems per split, differences of 1–2 points are noise. Use the k-fold paired test before
claiming significance.

`synthetic.py` exists only to smoke-test the code. Its numbers mean nothing.
