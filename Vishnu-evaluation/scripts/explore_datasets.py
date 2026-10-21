"""Dataset exploration for the report (Section III) and for experiment design.

Focuses on facts that affect our experiments:
  * size, MATH level/subject balance  -> stratified analysis by difficulty
  * question / reference-solution length -> rough difficulty proxy
  * answer format and answer token count -> DEER's confidence is an average
    over the trial answer's tokens, so short answers = few tokens averaged
  * grader sanity check on reference solutions
  * duplicates / train-test overlap -> leakage for the controller
  * our MATH file (from the DEER repo) vs. HuggingFaceH4/MATH-500

    python scripts/explore_datasets.py   # writes results/data_exploration/
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deer_eval.io import REPO_ROOT, load_benchmark, write_json  # noqa: E402
from deer_eval.scoring import extract_pred, gsm8k_gold_from_solution, is_correct  # noqa: E402

OUT = REPO_ROOT / "results" / "data_exploration"
TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def answer_type(a):
    a = a.strip()
    if re.fullmatch(r"-?\d+", a.replace(",", "")):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+", a):
        return "decimal"
    if re.fullmatch(r"-?\\d?frac\{-?\d+\}\{\d+\}", a) or re.fullmatch(r"-?\d+/\d+", a):
        return "fraction"
    if a.startswith(("(", "[", "\\left(", "\\left[")) or re.search(r"\d\s*,\s*-?\d", a):
        return "tuple/interval/list"
    if "\\text" in a or re.fullmatch(r"[A-Za-z ]+", a):
        return "text/choice"
    return "expression (sqrt, pi, vars...)"


def qtokens(tok, texts):
    return np.array([len(tok.encode(t, add_special_tokens=False)) for t in texts])


def trial_answer_tokens(tok, answers):
    # DEER's probe generates "{<answer>" and averages max-probs from the
    # second generated token on, so this is the number of tokens averaged.
    return np.array([max(len(tok.encode("{" + a, add_special_tokens=False)) - 1, 0) for a in answers])


def hist(ax, data, labels, title, xlabel, bins=40):
    for d, lab in zip(data, labels):
        ax.hist(d, bins=bins, alpha=0.55, label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    stats = {}

    # ---------------- GSM8K ----------------
    g_test = load_benchmark("gsm8k")
    g_train = list(load_dataset("openai/gsm8k", "main", split="train"))
    g_train_gold = [gsm8k_gold_from_solution(r["answer"]) for r in g_train]
    steps_test = np.array([len(re.findall(r"<<", r["cot"])) for r in g_test])
    steps_train = np.array([len(re.findall(r"<<", r["answer"])) for r in g_train])
    qt_g = qtokens(tok, [r["problem"] for r in g_test])
    at_g = trial_answer_tokens(tok, [r["answer"] for r in g_test])
    test_q = Counter(r["problem"].strip() for r in g_test)
    train_q = set(r["question"].strip() for r in g_train)
    stats["gsm8k"] = {
        "n_test": len(g_test), "n_train": len(g_train),
        "question_tokens_test": {"mean": float(qt_g.mean()), "median": float(np.median(qt_g)), "max": int(qt_g.max())},
        "calc_steps_test": {"mean": float(steps_test.mean()), "dist": {int(k): int(v) for k, v in sorted(Counter(steps_test).items())}},
        "calc_steps_train_mean": float(steps_train.mean()),
        "answer_types_test": dict(Counter(answer_type(r["answer"]) for r in g_test)),
        "negative_answers_test": sum(r["answer"].strip().startswith("-") for r in g_test),
        "trial_answer_tokens_test": {"mean": float(at_g.mean()), "dist": {int(k): int(v) for k, v in sorted(Counter(at_g).items())}},
        "duplicate_questions_in_test": sum(c - 1 for c in test_q.values() if c > 1),
        "test_questions_also_in_train": sum(q in train_q for q in test_q),
        "train_gold_parse_failures": sum(a is None for a in g_train_gold),
    }

    # ---------------- MATH-500 ----------------
    m = load_benchmark("math")
    levels = Counter(r["level"] for r in m)
    subjects = Counter(r["subject"] for r in m)
    qt_m = qtokens(tok, [r["problem"] for r in m])
    st_m = qtokens(tok, [r["solution"] for r in m])
    at_m = trial_answer_tokens(tok, [r["answer"] for r in m])
    types = Counter(answer_type(r["answer"]) for r in m)
    grade_fail = [(r["unique_id"], r["answer"], extract_pred(r["solution"]))
                  for r in m if not is_correct(extract_pred(r["solution"]), r["answer"])]
    by_level = {}
    for lv in sorted(levels):
        idx = [i for i, r in enumerate(m) if r["level"] == lv]
        by_level[lv] = {
            "n": len(idx),
            "question_tokens_mean": float(qt_m[idx].mean()),
            "solution_tokens_mean": float(st_m[idx].mean()),
            "trial_answer_tokens_mean": float(at_m[idx].mean()),
            "share_integer_answers": float(np.mean([answer_type(m[i]["answer"]) == "integer" for i in idx])),
        }
    hf500 = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    ours_ids = set(r["unique_id"] for r in m)
    missing = [r["unique_id"] for r in hf500 if r["unique_id"] not in ours_ids]
    stats["math500"] = {
        "n": len(m), "n_hf_math500": len(hf500), "missing_vs_hf": missing,
        "levels": dict(sorted(levels.items())), "subjects": dict(subjects.most_common()),
        "question_tokens": {"mean": float(qt_m.mean()), "median": float(np.median(qt_m)), "max": int(qt_m.max())},
        "answer_types": dict(types.most_common()),
        "trial_answer_tokens": {"mean": float(at_m.mean()), "median": float(np.median(at_m)), "max": int(at_m.max())},
        "by_level": by_level,
        "reference_solutions_graded_correct": len(m) - len(grade_fail),
        "grader_failures_on_reference": grade_fail,
    }
    write_json(stats, OUT / "stats.json")

    # ---------------- Figures ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].bar([str(k) for k in sorted(levels)], [levels[k] for k in sorted(levels)], color="#2a6fdb")
    axes[0].set_title("MATH-500: problems per level")
    axes[0].set_xlabel("level")
    sub = subjects.most_common()
    axes[1].barh([s for s, _ in sub][::-1], [c for _, c in sub][::-1], color="#2a6fdb")
    axes[1].set_title("MATH-500: problems per subject")
    lv = sorted(levels)
    axes[2].boxplot([st_m[[i for i, r in enumerate(m) if r["level"] == k]] for k in lv])
    axes[2].set_xticks(range(1, len(lv) + 1), [str(k) for k in lv])
    axes[2].set_title("MATH-500: reference solution length")
    axes[2].set_xlabel("level")
    axes[2].set_ylabel("tokens")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "math500_levels_subjects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    hist(axes[0], [qt_g, qt_m], ["GSM8K test", "MATH-500"], "Question length", "tokens")
    axes[1].bar(*zip(*sorted(Counter(steps_test).items())), color="#d9480f")
    axes[1].set_title("GSM8K test: calculation steps in reference")
    axes[1].set_xlabel("# <<calc>> steps")
    axes[1].grid(alpha=0.3)
    bins = np.arange(0, max(at_g.max(), at_m.max()) + 2) - 0.5
    hist(axes[2], [np.clip(at_g, 0, 30), np.clip(at_m, 0, 30)], ["GSM8K test", "MATH-500"],
         "Tokens DEER averages for confidence", "trial-answer tokens (gold answer, clipped at 30)",
         bins=bins[bins <= 31])
    fig.tight_layout()
    fig.savefig(OUT / "lengths_and_answer_tokens.png", dpi=180)
    plt.close(fig)

    # ---------------- Markdown summary ----------------
    g, mm = stats["gsm8k"], stats["math500"]
    lines = [
        "# Dataset exploration", "",
        "Generated by `scripts/explore_datasets.py`. Token counts use the DeepSeek-R1-Distill-Qwen tokenizer.", "",
        "## GSM8K", "",
        f"- Test {g['n_test']} / train {g['n_train']}. Question length: mean {g['question_tokens_test']['mean']:.0f} tokens, max {g['question_tokens_test']['max']}.",
        f"- Reference solutions use {g['calc_steps_test']['mean']:.1f} calculation steps on average (test), {g['calc_steps_train_mean']:.1f} (train).",
        f"- Answer types (test): {g['answer_types_test']}; {g['negative_answers_test']} negative.",
        f"- Trial-answer tokens DEER averages over: mean {g['trial_answer_tokens_test']['mean']:.2f}; distribution {g['trial_answer_tokens_test']['dist']}.",
        f"- Duplicate questions inside test: {g['duplicate_questions_in_test']}. Test questions also in train: {g['test_questions_also_in_train']}.",
        "",
        "## MATH-500", "",
        f"- {mm['n']} problems in the DEER repo's file vs {mm['n_hf_math500']} in HuggingFaceH4/MATH-500. Missing: {mm['missing_vs_hf']}.",
        f"- Levels: {mm['levels']}.",
        f"- Subjects: {mm['subjects']}.",
        f"- Answer types: {mm['answer_types']}.",
        f"- Trial-answer tokens: mean {mm['trial_answer_tokens']['mean']:.1f}, median {mm['trial_answer_tokens']['median']:.0f}, max {mm['trial_answer_tokens']['max']}.",
        f"- Grader sanity check: {mm['reference_solutions_graded_correct']}/{mm['n']} reference solutions grade as correct.",
        "", "| Level | n | Question tok | Solution tok | Trial-answer tok | Integer answers |", "|---|---|---|---|---|---|",
    ]
    for k, v in mm["by_level"].items():
        lines.append(f"| {k} | {v['n']} | {v['question_tokens_mean']:.0f} | {v['solution_tokens_mean']:.0f} | "
                     f"{v['trial_answer_tokens_mean']:.1f} | {100 * v['share_integer_answers']:.0f}% |")
    if mm["grader_failures_on_reference"]:
        lines += ["", "### Reference solutions the grader marks wrong", "", "| id | gold | extracted |", "|---|---|---|"]
        for uid, gold, pred in mm["grader_failures_on_reference"]:
            lines.append(f"| {uid} | `{gold}` | `{pred}` |")
    lines += ["", "![levels](math500_levels_subjects.png)", "", "![lengths](lengths_and_answer_tokens.png)", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
