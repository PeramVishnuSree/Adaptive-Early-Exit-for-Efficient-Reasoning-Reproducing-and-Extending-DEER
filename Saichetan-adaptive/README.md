# Adaptive Early-Exit Method (Sai Chetan Muppalla)

CMPE 258 Group 10: *Adaptive Early Exit for Efficient Reasoning: Reproducing and Extending DEER*.

This folder is my part of the project (trajectory-aware adaptive stopping):
- 12-dimensional checkpoint feature engineering beyond raw confidence \(c_k\)
- neural stopping controllers (`AdaptiveDEER-MLP`, GRU, plus LR/GBDT baselines)
- feature ablation, cross-dataset zero-shot, and 1.5B→7B transfer experiments
- Check-In 2 Colab notebook + result tables/figures

All paths below are relative to this folder.

## Layout

| Path | What |
|---|---|
| `notebooks/Experiment2_Sai_Chetan.ipynb` | Check-In 2 Colab: features, controllers, ablation, zero-shot |
| `Artifacts/tables/` | Result CSVs (Tables 1–5) extracted from notebook outputs |
| `Artifacts/figures/` | Publication figures from the notebook |
| `Artifacts/data/` | Placeholder for parquet trajectory dumps (sync from Drive) |
| `Artifacts/checkpoints/` | Placeholder for `adaptive_deer_mlp_15b.pt` |
| `Plots/` | Convenience copies of key figures |
| `adaptive_exit/` | Feature engineering + MLP/GRU controllers + offline stop simulation |
| `scripts/` | CLI: build synthetic trajectories, extract features, train, eval |
| `docs/method_notes.md` | Feature vector, oracle label, controllers |
| `docs/integration_notes.md` | How this plugs into Vishnu logged DEER / `simulate.py` |
| `tests/` | Unit tests on synthetic checkpoints |

## Experiment outline (from notebook)

1. **Trajectory corpus** — multi-scale (1.5B & 7B) × GSM8K + MATH-500 → 29,309 checkpoints  
2. **12-D features** — confidence dynamics, entropy/dispersion, answer stability, normalized \(L_k\), \(k\)  
3. **Controller bake-off** — Static DEER \(\tau{=}0.80\) vs LR / GBDT / GRU / Residual MLP  
4. **Feature ablation** — confidence-only → full 12-D  
5. **Cross-dataset zero-shot** — GSM8K ↔ MATH-500  
6. **Cross-scale transfer** — controller trained on 1.5B applied to 7B  

### Headline held-out results (1.5B test split)

| Controller | Acc (%) | Net compute save (%) | Premature exit (%) |
|---|---:|---:|---:|
| Static DEER (\(c_k \ge 0.80\)) | 63.30 | 42.94 | 16.04 |
| AdaptiveDEER-MLP (proposed) | **79.34** | 28.59 | **0.00** |

See `Artifacts/tables/` for full tables and `Plots/` for figures.

### Important data note

The Check-In 2 notebook ran with `enable_live_llm=False` and **synthesized** multi-scale trajectories for pipeline validation. Live DEER logs from Vishnu/Jaswanthi should replace `Artifacts/data/*.parquet` before claiming production LLM numbers.

## Quick start

```bash
cd Saichetan-adaptive
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python scripts/extract_features.py --synthetic --output Artifacts/data/features_synth.csv
python scripts/train_controller.py --input Artifacts/data/features_synth.csv --output results/controller.pt
python scripts/eval_controller.py --input Artifacts/data/features_synth.csv --checkpoint results/controller.pt
```

Peer integration (offline policy replay):

```bash
# After a never-exit logged DEER run from Vishnu-evaluation:
python scripts/eval_controller.py --deer-jsonl ../Vishnu-evaluation/outputs/deer_never_exit.jsonl \
  --checkpoint Artifacts/checkpoints/adaptive_deer_mlp_15b.pt
```

## Drive sync

Colab path used in the notebook: `/content/drive/MyDrive/CMPE258_Group10_DEER/artifacts/`.  
Copy parquet + `.pt` into `Artifacts/data/` and `Artifacts/checkpoints/` when available.
