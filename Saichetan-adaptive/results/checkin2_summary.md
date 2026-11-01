# Check-In 2 result summary (from Experiment2 notebook)

Source notebook: `notebooks/Experiment2_Sai_Chetan.ipynb`  
Figures: `Artifacts/figures/` · Tables: `Artifacts/tables/`

## Trajectory corpus

| Scale | Dataset | Problems | Checkpoints | Full CoT Acc | Avg Full Tokens |
|---|---|---:|---:|---:|---:|
| 1.5B | GSM8K | 800 | 6591 | 84.36% | 562.7 |
| 1.5B | MATH-500 | 500 | 8499 | 63.20% | 1599.9 |
| 7B | GSM8K | 800 | 6176 | 89.28% | 525.7 |
| 7B | MATH-500 | 500 | 8043 | 73.03% | 1531.6 |

**Total:** 29,309 checkpoints · 12 engineered features.

## Exp 2 — Controller architectures (1.5B held-out)

AdaptiveDEER-MLP: **79.34%** accuracy, **28.59%** net compute savings, **0%** premature exits  
vs Static DEER τ=0.80: 63.30% acc, 42.94% savings, **16.04%** premature exits.

## Exp 3 — Feature ablation

Uncertainty features (entropy / min-prob / SNR) are the step that reaches ROC-AUC 1.0 and 0% premature exits in this corpus; adding self-consistency and \(L_k,k\) keeps the same aggregate metrics.

## Exp 4 — Cross-dataset zero-shot

Adaptive MLP preserves Full CoT accuracy on the transfer target with ~24–36% compute savings; Static DEER trades large savings for large accuracy drops (−10% to −24%).

## Exp 5 — 1.5B → 7B transfer

Zero-shot AdaptiveDEER-MLP matches Full CoT accuracy on 7B GSM8K / MATH-500 with ~44% / ~38% compute savings, while Static DEER still incurs premature-exit accuracy loss.

## Caveat

Notebook used `enable_live_llm=False` (synthetic trajectories). Re-run on Vishnu never-exit DEER logs before final-report claims.
