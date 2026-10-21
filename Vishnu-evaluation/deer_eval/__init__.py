"""Evaluation pipeline for reproducing and extending DEER.

Modules:
    io        - read/write jsonl run files and benchmark files
    scoring   - extract answers and grade them against gold answers
    metrics   - accuracy / token / compute summaries for one run
    simulate  - offline early-exit simulation from per-checkpoint logs
    plots     - accuracy-vs-tokens tradeoff figures
"""
