# Evaluation & Experiment Pipeline (Vishnu Sree Peram)

CMPE 258 Group 10: *Adaptive Early Exit for Efficient Reasoning: Reproducing and Extending DEER*.
This folder is my part of the project (evaluation and experimental analysis):
- the evaluation pipeline (scoring, metrics, plots, offline early-exit simulation)
- the scripts that run Full CoT and DEER on a GPU
- the Colab smoke-test notebook
- dataset exploration and notes on the official DEER code

All paths below are relative to this folder.

## Layout

| Path | What |
|---|---|
| `third_party/deer/vllm_deer_official.py` | Unmodified official DEER script (MIT), for reference |
| `third_party/deer/vllm_deer_logged.py` | Same script plus per-checkpoint logging, `--limit`, `--offset`, `--output_file`, `--trial_temperature`, `--final_answer`. Changes are marked `[OURS]` |
| `scripts/run_full_cot.py` | Full chain-of-thought baseline (same prompt and budget, no early exit) |
| `scripts/score_run.py` | Grade one run and write `.scored.jsonl` + `.summary.json` |
| `scripts/compare_runs.py` | Comparison table + accuracy-vs-tokens plot (+ offline threshold sweep) |
| `scripts/explore_datasets.py` | Dataset statistics and figures → `results/data_exploration/` |
| `scripts/make_train_splits.py` | Builds `data/gsm8k_train` (1,000 GSM8K train) and `data/math_train` (500 MATH train, none in MATH-500) |
| `scripts/compare_stopping_methods.py` | Trains stopping methods on training trajectories, tests on test trajectories, with paired bootstrap CIs vs. DEER |
| `deer_eval/` | Scoring (DEER's own grader, vendored), metrics, offline exit simulation, plots |
| `notebooks/01_smoke_test.ipynb` | Colab notebook: Full CoT vs. DEER τ=0.95 on a subset |
| `notebooks/02_checkpoint3_never_exit.ipynb` | Checkpoint 3 never-exit trajectories (first 100 test problems) |
| `notebooks/04_training_trajectories_and_answer_fix.ipynb` | Never-exit trajectories on the training sets; DEER with `--final_answer trial` / `boxed` |
| `docs/deer_code_notes.md` | How the official code works, and reproduction pitfalls |
| `data/` | GSM8K test and MATH-500 in DEER's format (from the DEER repo), plus `gsm8k_train/` and `math_train/` (each file is named `test.jsonl` because the DEER script always reads that name) |

## Metrics

- **accuracy**: DEER's rule-based grader on the last `\boxed{}`
- **reasoning tokens**: tokens before `</think>`
- **output tokens**: reasoning + answer. This is the paper's "length"
- **compute tokens**: output tokens plus DEER's trial-answer probe tokens
- **token reduction ratio**: `1 - output_tokens / output_tokens(Full CoT)`
- **wall time**: whole batched run, from each run's `.meta.json`

## Quick start

Local (CPU, analysis only):

```bash
cd Vishnu-evaluation
python3 -m venv .venv && .venv/bin/pip install -r requirements-eval.txt
.venv/bin/python -m pytest            # 32 tests, includes a fake-vLLM run of the DEER script
.venv/bin/python scripts/explore_datasets.py
```

GPU runs: open `notebooks/01_smoke_test.ipynb` in Colab (T4), or on the HPC:

```bash
pip install -r requirements-gpu.txt
python scripts/run_full_cot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dataset gsm8k --limit 50 --output_file outputs/fullcot_gsm8k.jsonl
python third_party/deer/vllm_deer_logged.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_dir data --dataset gsm8k --limit 50 --threshold 0.95 --think_ratio 0.6 --policy avg1 \
    --max_generated_tokens 16000 --dtype float16 --output_file outputs/deer_t0.95_gsm8k.jsonl
python scripts/score_run.py --input outputs/deer_t0.95_gsm8k.jsonl --dataset gsm8k
```

## Grading environment note

DEER pins `latex2sympy2==1.9.1` with `antlr4-python3-runtime==4.11.1`, and those two
can't be installed together. We use `latex2sympy2_extended` (ANTLR 4.11) instead. Checked:
all 500 MATH-500 reference solutions grade as correct.

## DEER script options added for this project

| Flag | Default | What it does |
|---|---|---|
| `--offset K` | 0 | Skip the first K questions, so a dataset can run in chunks. Output `idx` stays the position in the full file. |
| `--final_answer` | `regenerate` | `regenerate`: official behavior (free-form answer after `</think>`). `trial`: on an early exit, keep the trial answer DEER stopped on. `boxed`: the answer after `</think>` is prompted to start with `**Final Answer**\n\boxed{`. |
| `--trial_temperature T` | -1 | Temperature of the trial-answer probe; -1 keeps the official behavior (vLLM default 1.0). |
