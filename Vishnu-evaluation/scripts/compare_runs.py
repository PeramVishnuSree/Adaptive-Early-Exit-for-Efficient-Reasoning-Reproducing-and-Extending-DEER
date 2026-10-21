"""Compare scored runs on the same questions: table + tradeoff plot.

    python scripts/compare_runs.py \\
        --runs "Full CoT=outputs/fullcot_gsm8k.scored.jsonl" \\
               "DEER t=0.95=outputs/deer_gsm8k_t0.95.scored.jsonl" \\
        --baseline "Full CoT" --out_dir results/smoke_gsm8k --title "GSM8K (n=50)"

Optional: --sweep <never-exit run .scored.jsonl> adds an offline threshold
sweep curve ("DEER (simulated)").
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deer_eval.io import read_jsonl, write_json  # noqa: E402
from deer_eval.metrics import group_summary, relative_to, summarize  # noqa: E402
from deer_eval.plots import group_bar_plot, tradeoff_plot  # noqa: E402
from deer_eval.simulate import sweep_thresholds  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help='"Label=path.scored.jsonl"')
    p.add_argument("--baseline", default=None, help="Label of the baseline run")
    p.add_argument("--sweep", default=None, help="Never-exit run for an offline threshold sweep")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()

    runs = {}
    for spec in args.runs:
        label, _, path = spec.rpartition("=")  # last "=", so labels like "DEER t=0.95" work
        runs[label] = read_jsonl(path)

    # Compare only questions present in every run.
    common = set.intersection(*(set(r["idx"] for r in recs) for recs in runs.values()))
    runs = {k: [r for r in v if r["idx"] in common] for k, v in runs.items()}
    summaries = {k: summarize(v) for k, v in runs.items()}
    base = summaries.get(args.baseline)

    lines = [
        f"# {args.title or 'Run comparison'}", "", f"Questions compared: {len(common)}", "",
        "| Method | Accuracy | Avg reasoning tok | Avg output tok | Avg compute tok | Token reduction | Acc. delta |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, s in summaries.items():
        rel = relative_to(s, base) if base else None
        red = f"{100 * rel['token_reduction_ratio']:.1f}%" if rel else "-"
        dacc = f"{100 * rel['accuracy_delta']:+.1f}" if rel else "-"
        lines.append(
            f"| {label} | {100 * s['accuracy']:.1f} ± {100 * s['accuracy_ci95']:.1f} | "
            f"{s['avg_reasoning_tokens']:.0f} | {s['avg_output_tokens']:.0f} | "
            f"{s['avg_compute_tokens']:.0f} | {red} | {dacc} |"
        )

    curves = None
    if args.sweep:
        sweep = [r for r in read_jsonl(args.sweep) if r["idx"] in common]
        curves = {"DEER (simulated)": sweep_thresholds(sweep)}
        lines += ["", "## Offline threshold sweep (trial answer used as exit answer)", "",
                  "| tau | Accuracy | Avg reasoning tok |", "|---|---|---|"]
        for s in curves["DEER (simulated)"]:
            lines.append(f"| {s['tau']:g} | {100 * s['accuracy']:.1f} | {s['avg_reasoning_tokens']:.0f} |")

    os.makedirs(args.out_dir, exist_ok=True)
    tradeoff_plot(summaries, os.path.join(args.out_dir, "tradeoff.png"), args.title, curves=curves)
    if any("level" in r for recs in runs.values() for r in recs):
        by_level = {k: group_summary(v, "level") for k, v in runs.items()}
        group_bar_plot(by_level, os.path.join(args.out_dir, "accuracy_by_level.png"),
                       title=args.title)
        group_bar_plot(by_level, os.path.join(args.out_dir, "tokens_by_level.png"),
                       metric="avg_reasoning_tokens", title=args.title)
    write_json({"summaries": summaries, "n_common": len(common)},
               os.path.join(args.out_dir, "comparison.json"))
    with open(os.path.join(args.out_dir, "comparison.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote figures and tables to {args.out_dir}")


if __name__ == "__main__":
    main()
