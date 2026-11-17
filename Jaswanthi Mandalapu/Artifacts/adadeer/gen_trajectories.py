"""Log never-exit DEER trajectories with DeepSeek-R1-Distill-Qwen-1.5B on Apple silicon (MLX).

Equivalent to running DEER's loop with an unreachable threshold (tau = 1.1):
  * greedy reasoning, a trial answer is probed at every "Wait" (up to --max-ck),
    the probe is rewound and reasoning continues unchanged;
  * thinking is capped at think_ratio * budget (forced "\\n</think>\\n\\n"),
    then the conclusion is generated up to the total budget.

Because probes are rewound, the reasoning chain does not depend on them, so we
generate it first (phase 1, batched) and replay the probes on a KV cache that
is extended incrementally along the chain (phase 2).

python -m adadeer.gen_trajectories --data data/math500.jsonl --start 0 --end 100 \
       --model <path> --out traj/math500_0_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mlx.core as mx
from mlx_lm import load
import mlx_lm.generate as _G
from mlx_lm.generate import BatchGenerator

# Wiring model memory makes decoding ~100x slower on this 16 GB machine; keep it unwired.
_G.maybe_set_recommended_wired_limit = lambda *a, **k: None
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from .grading import brace_answer, is_correct, last_boxed

INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
INDUCER = "\n**Final Answer**\n\\boxed"
FORCE_END = "\n</think>\n\n"


def build_prompt(tok, question):
    p = tok.apply_chat_template([{"role": "user", "content": question + INSTR}],
                                add_generation_prompt=True, tokenize=False)
    if not p.rstrip().endswith("<think>"):
        p += "<think>\n"
    return tok.encode(p, add_special_tokens=False)


def run_batch(model, prompts, max_tokens, stop, batch):
    gen = BatchGenerator(model, max_tokens=max_tokens, stop_tokens=stop,
                         completion_batch_size=batch, prefill_batch_size=min(batch, 8))
    uids = gen.insert(prompts, max_tokens=max_tokens if isinstance(max_tokens, list) else None)
    out = {u: [] for u in uids}
    reason = {}
    t0, n = time.time(), 0
    while len(reason) < len(uids):
        for r in gen.next_generated():
            out[r.uid].append(int(r.token))
            n += 1
            if r.finish_reason is not None:
                reason[r.uid] = r.finish_reason
        if n and n % 20000 < len(uids) - len(reason):
            print(f"    {n} tok, {n / (time.time() - t0):.0f} tok/s, done {len(reason)}/{len(uids)}", flush=True)
    gen.close()
    return [out[u] for u in uids], [reason[u] for u in uids]


def probe_all(model, tok, prompt_ids, reasoning, wait_pos, eos):
    """Trial answer + max-token probabilities at each checkpoint (reasoning prefix R[:j])."""
    ind = tok.encode(INDUCER, add_special_tokens=False)
    seq = prompt_ids + reasoning
    cache = make_prompt_cache(model)
    pos, res = 0, []
    for j in wait_pos:
        target = len(prompt_ids) + j
        while pos < target:
            step = min(2048, target - pos)
            model(mx.array([seq[pos:pos + step]]), cache=cache)
            mx.eval([c.state for c in cache])
            pos += step
        logits = model(mx.array([ind]), cache=cache)[:, -1, :]
        fed, toks, probs, ans, closed = len(ind), [], [], None, False
        for _ in range(20):
            p = mx.softmax(logits.astype(mx.float32), axis=-1)[0]
            t = int(mx.argmax(p).item())
            toks.append(t)
            probs.append(float(p[t].item()))
            ans, closed = brace_answer(tok.decode(toks))
            if closed or t in eos:
                break
            logits = model(mx.array([[t]]), cache=cache)[:, -1, :]
            fed += 1
        trim_prompt_cache(cache, fed)
        res.append({"reasoning_tokens": j, "answer": ans if closed else None,
                    "token_probs": [round(x, 6) for x in probs], "probe_tokens": len(toks),
                    "probe_text": tok.decode(toks)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--think-ratio", type=float, default=0.6)
    ap.add_argument("--max-ck", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    from .policy import confidence_from_probs
    rows = [json.loads(l) for l in open(a.data)][a.start:a.end]
    model, tok = load(a.model)
    think = int(a.think_ratio * a.budget)
    think_end = tok.convert_tokens_to_ids("</think>")
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    wait_ids = {i for s, i in tok._tokenizer.get_vocab().items()
                if s.lstrip("Ġ").lstrip("Ċ") == "Wait" and tok.decode([i]).strip() == "Wait"}
    prompts = [build_prompt(tok, r["question"]) for r in rows]
    print(f"{len(rows)} problems, think budget {think}, Wait token ids {sorted(wait_ids)}", flush=True)

    p1 = a.out + ".phase1.json"
    if os.path.exists(p1):
        reasoning, natural = json.load(open(p1))
    else:
        t0 = time.time()
        reasoning, fin = run_batch(model, prompts, think, [[think_end]], a.batch)
        natural = [bool(r) and r[-1] == think_end for r in reasoning]
        # conclusion
        conc_prompts, conc_max = [], []
        for p, r, nat in zip(prompts, reasoning, natural):
            tail = [] if nat else tok.encode(FORCE_END, add_special_tokens=False)
            if nat:
                r[:] = r[:-1]          # drop </think>; re-added below so lengths count reasoning only
                tail = [think_end]
            conc_prompts.append(p + r + tail)
            conc_max.append(max(16, a.budget - len(r) - len(tail)))
        conclusions, _ = run_batch(model, conc_prompts, conc_max, [[e] for e in eos], a.batch)
        reasoning = [{"r": r, "c": c, "tail": len(tp) - len(p) - len(r)}
                     for r, c, p, tp in zip(reasoning, conclusions, prompts, conc_prompts)]
        json.dump([reasoning, natural], open(p1, "w"))
        print(f"phase 1 done in {(time.time() - t0) / 60:.1f} min", flush=True)

    t0 = time.time()
    with open(a.out, "w") as f:
        for row, p, rc, nat in zip(rows, prompts, reasoning, natural):
            r, c = rc["r"], rc["c"]
            waits = [j for j, t in enumerate(r) if t in wait_ids][: a.max_ck]
            cks = probe_all(model, tok, p, r, waits, eos)
            for ck in cks:
                ck["confidence"] = confidence_from_probs(ck["token_probs"], "avg1")
                ck["correct"] = is_correct(ck["answer"], row["gold"])
            conc_text = tok.decode([t for t in c if t not in eos])
            final = last_boxed(conc_text)
            f.write(json.dumps({
                "id": row["id"], "dataset": row["dataset"], "level": row.get("level"),
                "gold": row["gold"], "final_answer": final,
                "full_correct": is_correct(final, row["gold"]),
                "full_reasoning_tokens": len(r), "full_output_tokens": len(r) + rc["tail"] + len(c),
                "budget_hit": not nat, "checkpoints": cks,
                "conclusion": conc_text[-600:]}) + "\n")
            f.flush()
    print(f"phase 2 done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
