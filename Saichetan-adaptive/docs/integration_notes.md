# Integration with peer pipelines

## Vishnu Sree Peram (`Vishnu-evaluation/`)

| Peer asset | How Adaptive DEER uses it |
|---|---|
| `third_party/deer/vllm_deer_logged.py` | Produce never-exit runs (`--threshold 1.1`) with `checkpoints[]` |
| `deer_eval/simulate.py` (`checkpoint_table`, `simulate_policy`) | Offline feature rows + policy replay without GPU |
| `deer_eval/scoring.py` | Grade trial answers for training labels |
| `docs/deer_code_notes.md` | Exit hook is `pred_prob > threshold` — replace with controller |

### Recommended offline loop

1. Run logged DEER never-exit on GSM8K / MATH-500.
2. Score checkpoints (`trial_correct`).
3. Map logged fields → our 12-D features (`adaptive_exit.features`).
4. Train `AdaptiveDEER_MLP`; save `Artifacts/checkpoints/adaptive_deer_mlp_15b.pt`.
5. Replay with `should_stop(cps, rec)` via `simulate_policy` or `scripts/eval_controller.py`.

### Online hook (TODO)

In `vllm_deer_logged.py`, after a checkpoint is appended:

```python
# Official:
# if state['pred_prob'] > args.threshold or thinking_limit_reached:
# Adaptive:
# if controller.should_stop(state['checkpoints'], state) or thinking_limit_reached:
```

Do not edit Vishnu’s tree until the team agrees on a shared branch; keep the
controller load path under `Saichetan-adaptive/`.

## Jaswanthi Mandalapu (`Jaswanthi Mandalapu/`)

| Peer asset | Use |
|---|---|
| `Artifacts/deer_20_results/...jsonl` | Small measured DEER run (τ=0.95, 20 GSM8K) for smoke integration |
| Full CoT baseline numbers | Accuracy / token reference for reporting |
| Plots | Side-by-side with AdaptiveDEER figures in the joint report |

Her measured schema (`reasoning_tokens`, `confidence_at_stop`, …) is coarser
than Vishnu’s per-checkpoint log. Prefer Vishnu never-exit JSONL for training;
use Jaswanthi’s 20-Q run as an end-to-end smoke check once online exit is wired.
