"""Full chain-of-thought baseline: plain greedy generation, no early exit.

Uses the same system prompt, chat template and total token budget as DEER's
vllm-deer.py so the two are directly comparable. Needs a CUDA GPU + vLLM.

    python scripts/run_full_cot.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
        --dataset gsm8k --limit 50 --dtype float16 --output_file outputs/fullcot_gsm8k.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deer_eval.io import load_benchmark, write_jsonl  # noqa: E402

# Identical to vllm-deer.py
SYS_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_generated_tokens", type=int, default=16000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", default="float16", help="float16 on T4/V100; bfloat16 on A100/H100")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--output_file", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_benchmark(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": r["problem"]}],
            tokenize=False, add_generation_prompt=True)
        for r in rows
    ]
    llm = LLM(model=args.model, dtype=args.dtype, max_model_len=args.max_generated_tokens + 4000,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    params = SamplingParams(max_tokens=args.max_generated_tokens, temperature=args.temperature,
                            top_p=args.top_p)

    start = time.time()
    outputs = llm.generate(prompts, params)
    wall = time.time() - start

    records = []
    for i, (row, out) in enumerate(zip(rows, outputs)):
        o = out.outputs[0]
        think, sep, answer = o.text.rpartition("</think>")
        if not sep:
            think, answer = o.text, ""
        records.append({
            "idx": i,
            "question": row["problem"],
            "gold_answer": row["answer"],
            "generated_responses": [o.text],
            "num_thinking_tokens": len(tok.encode(think, add_special_tokens=False)),
            "num_answer_tokens": len(tok.encode(answer, add_special_tokens=False)) if answer else 0,
            "exit_reason": "budget" if o.finish_reason == "length" else "natural",
            "checkpoints": [],
        })
    write_jsonl(records, args.output_file)
    meta = args.output_file[: -len(".jsonl")] + ".meta.json"
    with open(meta, "w") as f:
        json.dump({"wall_time_s": wall, "n_questions": len(rows), "args": vars(args)}, f, indent=2)
    print(f"Wrote {len(records)} records to {args.output_file} in {wall:.1f}s")


if __name__ == "__main__":
    main()
