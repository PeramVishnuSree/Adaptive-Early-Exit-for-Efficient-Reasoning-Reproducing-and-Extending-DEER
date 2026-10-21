"""Score one generation run and write a summary.

Works on output from the official vllm-deer.py, our logged DEER script, or
our full-CoT script.

    python scripts/score_run.py --input outputs/deer_gsm8k_t0.95.jsonl --dataset gsm8k

Writes <input>.scored.jsonl (per question) and <input>.summary.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deer_eval.io import load_benchmark, read_jsonl, write_json, write_jsonl  # noqa: E402
from deer_eval.metrics import annotate, group_summary, summarize  # noqa: E402
from deer_eval.simulate import score_checkpoints  # noqa: E402

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def stem(path):
    for ext in (".scored.jsonl", ".jsonl"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--dataset", required=True, help="gsm8k or math (folder under data/)")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                   help="Used only when the run file has no stored token counts.")
    args = p.parse_args()

    records = read_jsonl(args.input)
    bench = load_benchmark(args.dataset)

    tokenizer = None
    if records and "num_thinking_tokens" not in records[0]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    annotated = score_checkpoints(annotate(records, bench, tokenizer))

    meta_path = stem(args.input) + ".meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    summary = summarize(annotated, wall_time_s=meta.get("wall_time_s"))
    summary["input"] = os.path.basename(args.input)
    summary["dataset"] = args.dataset
    if meta:
        summary["run_meta"] = meta
    if any("level" in r for r in annotated):
        summary["by_level"] = group_summary(annotated, "level")

    out = stem(args.input)
    write_jsonl(annotated, out + ".scored.jsonl")
    write_json(summary, out + ".summary.json")

    print(f"{summary['input']}  n={summary['n']}")
    print(f"  accuracy           {100 * summary['accuracy']:.1f}% (+/- {100 * summary['accuracy_ci95']:.1f})")
    print(f"  avg reasoning tok  {summary['avg_reasoning_tokens']:.0f}")
    print(f"  avg output tok     {summary['avg_output_tokens']:.0f}")
    print(f"  avg compute tok    {summary['avg_compute_tokens']:.0f}")
    print(f"  exit reasons       {summary['exit_reasons']}")
    print(f"  wrote {out}.scored.jsonl and .summary.json")


if __name__ == "__main__":
    main()
