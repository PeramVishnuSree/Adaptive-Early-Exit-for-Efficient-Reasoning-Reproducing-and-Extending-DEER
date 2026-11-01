#!/usr/bin/env python3
"""Extract 12-D trajectory features from synthetic data or a parquet/CSV dump."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from adaptive_exit.features import engineer_trajectory_features
from adaptive_exit.synthetic import make_synthetic_trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Raw trajectory parquet/csv")
    ap.add_argument("--synthetic", action="store_true", help="Build tiny synthetic corpus")
    ap.add_argument("--n-problems", type=int, default=40)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    if args.synthetic or not args.input:
        df = make_synthetic_trajectories(n_problems=args.n_problems)
    else:
        path = Path(args.input)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    feat, cols = engineer_trajectory_features(df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        feat.to_parquet(out, index=False)
    else:
        feat.to_csv(out, index=False)
    print(f"[OK] {len(feat)} rows, {len(cols)} features -> {out}")


if __name__ == "__main__":
    main()
