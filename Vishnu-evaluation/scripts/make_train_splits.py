"""Build the training sets for the adaptive controller (never-exit trajectories).

Writes, in DEER's data format (one JSON object per line):
  data/gsm8k_train/test.jsonl  random 1,000 of the 7,473 GSM8K train problems
  data/math_train/test.jsonl   random 500 of the 7,500 MATH train problems,
                               excluding any problem that appears in MATH-500

The file is called test.jsonl only because the DEER script always reads
<dataset_dir>/<dataset>/test.jsonl. None of these problems are in our test sets.

Usage (from Vishnu-evaluation/):
    python scripts/make_train_splits.py
"""
import collections
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from deer_eval.grading import check_is_correct, extract_answer  # noqa: E402
from deer_eval.io import read_jsonl, write_jsonl  # noqa: E402

SEED = 0
N_GSM8K = 1000
N_MATH = 500

MATH_SUBJECTS = {  # EleutherAI/hendrycks_math config -> MATH-500 subject name
    "algebra": "Algebra",
    "counting_and_probability": "Counting & Probability",
    "geometry": "Geometry",
    "intermediate_algebra": "Intermediate Algebra",
    "number_theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
}


def gold_is_gradable(ans):
    try:
        return bool(ans) and bool(check_is_correct(ans, ans))
    except Exception:
        return False


def build_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for i, ex in enumerate(ds):
        cot, _, ans = ex["answer"].rpartition("####")
        rows.append({"problem": ex["question"], "answer": ans.strip().replace(",", ""),
                     "cot": cot.strip(), "id": f"gsm8k-train-{i}"})
    rng = random.Random(SEED)
    rows = rng.sample(rows, N_GSM8K)
    rows = [r for r in rows if gold_is_gradable(r["answer"])]
    return rows


def build_math():
    test_problems = {r["problem"].strip() for r in read_jsonl(ROOT / "data/math/test.jsonl")}
    rows = []
    for cfg, subject in MATH_SUBJECTS.items():
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for i, ex in enumerate(ds):
            m = re.match(r"Level (\d)", ex["level"])
            if not m or ex["problem"].strip() in test_problems:
                continue
            ans = extract_answer(ex["solution"])
            if not gold_is_gradable(ans):
                continue
            rows.append({"problem": ex["problem"], "solution": ex["solution"], "answer": ans,
                         "subject": subject, "level": int(m.group(1)),
                         "unique_id": f"train/{cfg}/{i}"})
    rng = random.Random(SEED)
    return rng.sample(rows, N_MATH)


def main():
    gsm = build_gsm8k()
    write_jsonl(gsm, ROOT / "data/gsm8k_train/test.jsonl")
    print(f"gsm8k_train: {len(gsm)} problems")

    math = build_math()
    write_jsonl(math, ROOT / "data/math_train/test.jsonl")
    levels = collections.Counter(r["level"] for r in math)
    print(f"math_train: {len(math)} problems, levels {dict(sorted(levels.items()))}")


if __name__ == "__main__":
    main()
