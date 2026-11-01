#!/usr/bin/env python3
"""Train AdaptiveDEER-MLP on an engineered feature table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from adaptive_exit.controller import StoppingController
from adaptive_exit.features import FEATURE_COLS


def _group_split(df: pd.DataFrame, test_size: float = 0.3, seed: int = 42):
    """Hold out a fraction of problem_ids (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df["problem_id"].unique()))
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_size)))
    test_groups = set(groups[:n_test])
    test_mask = df["problem_id"].isin(test_groups)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Engineered features CSV/parquet")
    ap.add_argument("--output", required=True, help="Path for controller .pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--threshold", type=float, default=0.62)
    args = ap.parse_args()

    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "y_correct" not in df.columns and "is_trial_correct" in df.columns:
        df["y_correct"] = df["is_trial_correct"]

    train, test = _group_split(df, test_size=0.3, seed=42)

    ctrl = StoppingController(threshold=args.threshold)
    losses = ctrl.fit(
        train[FEATURE_COLS].values,
        train["y_correct"].values,
        epochs=args.epochs,
    )
    out = Path(args.output)
    ctrl.save(out)
    meta = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "final_loss": float(losses[-1]) if losses else None,
        "threshold": args.threshold,
        "checkpoint": str(out),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
