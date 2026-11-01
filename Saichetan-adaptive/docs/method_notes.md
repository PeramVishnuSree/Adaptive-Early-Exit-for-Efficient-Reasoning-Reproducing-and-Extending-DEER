# Method notes — Trajectory-aware Adaptive DEER

Author: Sai Chetan Muppalla (018310504). Source of truth for Check-In 2:
`notebooks/Experiment2_Sai_Chetan.ipynb`.

## Motivation

Fixed-threshold DEER exits when \(c_k \ge \tau\). That ignores confidence
momentum, uncertainty inside the trial answer, and whether the trial answer
is stable across checkpoints. On hard problems a high \(c_k\) can still be
wrong; on easy ones a slightly lower \(c_k\) with a long stability streak can
be safe.

## Checkpoint feature vector \(x_k \in \mathbb{R}^{12}\)

| # | Name | Meaning |
|---|---|---|
| 1 | `trial_confidence` | Instantaneous \(c_k\) |
| 2 | `delta_c1` | \(\Delta c_k = c_k - c_{k-1}\) |
| 3 | `delta_c2` | \(\Delta^2 c_k\) |
| 4 | `ema_conf` | EMA of confidence |
| 5 | `conf_drawdown` | Peak confidence so far minus \(c_k\) |
| 6 | `token_entropy` | Shannon entropy \(H_k\) of trial tokens |
| 7 | `dispersion_min_prob` | Weakest token prob in trial answer |
| 8 | `conf_entropy_ratio` | \(c_k / (H_k + 0.1)\) |
| 9 | `ans_match_prev1` | \(\mathbb{I}[A_k = A_{k-1}]\) |
| 10 | `ans_stability_streak` | Consecutive equal trial answers |
| 11 | `norm_token_length` | \(L_k / 1500\) (clipped) |
| 12 | `norm_checkpoint_pos` | \(k / 15\) (clipped) |

Training label \(y_k\) = `is_trial_correct` (quality-gated: stop only when the
trial answer matches gold). At inference, \(y_k\) is unknown — only \(x_k\) is used.

## Controllers

- **Static DEER** — \(c_k \ge \tau\) (baseline, \(\tau=0.80\) in Exp. 2)
- **AdaptiveDEER-LR** — L2 logistic regression
- **AdaptiveDEER-GBDT** — `HistGradientBoostingClassifier`
- **AdaptiveDEER-MLP** — residual MLP + BatchNorm + SiLU + Dropout + asymmetric focal loss (proposed)
- **AdaptiveDEER-GRU** — 2-layer GRU over the causal feature sequence

Stop decision: \(\sigma(f_\theta(x_k)) \ge \theta\) with \(\theta \approx 0.62\) for neural policies.

## Experiments in Check-In 2

1. Multi-scale trajectory corpus (1.5B / 7B × GSM8K / MATH-500)
2. Architecture comparison on held-out 1.5B problems
3. Feature-group ablation + \(c_k\)–\(\Delta c_k\) decision boundary
4. Cross-dataset zero-shot (GSM8K ↔ MATH-500)
5. Cross-scale transfer (1.5B-trained controller → 7B traces)

## Caveat

Notebook Cell 2 used `enable_live_llm=False` and synthetic trajectory generation
for pipeline validation. Replace with Vishnu never-exit logged DEER JSONL /
Jaswanthi measured logs before reporting live-LLM numbers.
