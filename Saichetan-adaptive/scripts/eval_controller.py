#!/usr/bin/env python3
"""Offline eval of AdaptiveDEER-MLP vs static confidence threshold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from adaptive_exit.controller import StoppingController
from adaptive_exit.features import FEATURE_COLS, engineer_trajectory_features
from adaptive_exit.simulate import simulate_sequential_stopping


def _try_vishnu_simulate(jsonl: Path, ctrl: StoppingController):
    """Optional offline replay via Vishnu deer_eval.simulate if importable."""
    vishnu = ROOT.parent / "Vishnu-evaluation"
    if not vishnu.exists():
        return None
    sys.path.insert(0, str(vishnu))
    try:
        from deer_eval.io import read_jsonl  # type: ignore
        from deer_eval.simulate import score_checkpoints, simulate_policy  # type: ignore
    except Exception as e:
        print(f"[WARN] Could not import Vishnu deer_eval: {e}")
        return None
    rows = read_jsonl(jsonl)
    score_checkpoints(rows)
    results = simulate_policy(rows, ctrl.should_stop)
    n = len(results)
    acc = 100.0 * sum(r["correct"] for r in results) / max(n, 1)
    tok = sum(r["reasoning_tokens"] for r in results) / max(n, 1)
    return {"n": n, "Accuracy (%)": round(acc, 2), "Avg Reasoning Tokens": round(tok, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Engineered feature CSV/parquet")
    ap.add_argument("--deer-jsonl", help="Vishnu never-exit logged JSONL")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--static-tau", type=float, default=0.80)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    ctrl = StoppingController().load(args.checkpoint)
    summary = {}

    if args.deer_jsonl:
        v = _try_vishnu_simulate(Path(args.deer_jsonl), ctrl)
        if v:
            summary["vishnu_simulate"] = v

    if args.input:
        path = Path(args.input)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        if not set(FEATURE_COLS).issubset(df.columns):
            df, _ = engineer_trajectory_features(df)
        if "y_correct" not in df.columns:
            df["y_correct"] = df["is_trial_correct"]
        df["prob_mlp"] = ctrl.predict_proba(df[FEATURE_COLS].values)
        summary["adaptive_mlp"] = simulate_sequential_stopping(
            df, "prob_mlp", ctrl.threshold, "AdaptiveDEER-MLP"
        )
        summary["static_deer"] = simulate_sequential_stopping(
            df, "trial_confidence", args.static_tau, f"Static DEER tau={args.static_tau}"
        )

    print(json.dumps(summary, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
