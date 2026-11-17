"""Synthetic never-exit trajectories -- FOR SMOKE-TESTING THE PIPELINE ONLY.

Numbers produced from this data are not evidence about DEER or AdaDEER; use
the real never-exit logs for every reported result.
"""
import argparse
import json
import random


def gen(n, dataset, seed):
    rng = random.Random(seed)
    for i in range(n):
        level = rng.randint(1, 5) if dataset == "math500" else None
        hard = (level or 1) / 5
        n_ck = 0 if (dataset == "gsm8k" and rng.random() < 0.64) else rng.randint(1, int(4 + 14 * hard))
        gold, L, cps, have = "42", rng.randint(150, 400), [], False
        for k in range(n_ck):
            L += rng.randint(80, int(200 + 400 * hard))
            have = have or rng.random() < 0.45 - 0.25 * hard
            if have and rng.random() > 0.08:
                ans, conf = gold, (rng.uniform(0.99, 1.0) if rng.random() < 0.8 else rng.uniform(0.85, 0.99))
            else:
                ans = str(rng.choice([17, 24, 41, 43, 84]))
                conf = rng.uniform(0.3, 0.9) if rng.random() < 0.7 else rng.uniform(0.93, 0.999)
            cps.append({"reasoning_tokens": L, "answer": ans, "confidence": round(conf, 4),
                        "correct": ans == gold, "probe_tokens": rng.randint(4, 7)})
        full_correct = have and rng.random() < 0.9
        yield {"id": f"{dataset}-{i}", "dataset": dataset, "level": level,
               "full_correct": full_correct, "full_reasoning_tokens": L + 150,
               "full_output_tokens": L + 400, "checkpoints": cps}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        for ds, s in [("gsm8k", 1), ("math500", 2)]:
            for r in gen(100, ds, s):
                f.write(json.dumps(r) + "\n")
