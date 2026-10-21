"""Figures for the report. All functions save a PNG and return the path."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed colors per method so every figure in the report matches.
METHOD_COLORS = {
    "Full CoT": "#4c4c4c",
    "DEER": "#2a6fdb",
    "DEER (simulated)": "#8fb3ef",
    "Adaptive DEER": "#d9480f",
}
FALLBACK = ["#2f9e44", "#ae3ec9", "#f08c00", "#0c8599"]


def _color(label, i):
    for k in sorted(METHOD_COLORS, key=len, reverse=True):  # longest match first
        if label.startswith(k):
            return METHOD_COLORS[k]
    return FALLBACK[i % len(FALLBACK)]


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def tradeoff_plot(points, path, title="", x_key="avg_reasoning_tokens", curves=None):
    """Accuracy vs. average tokens.

    points: {label: summary} single runs, drawn as markers with error bars.
    curves: {label: [summary, ...]} e.g. a threshold sweep, drawn as lines;
        summaries may carry 'tau' for point labels.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    i = 0
    for label, rows in (curves or {}).items():
        rows = sorted(rows, key=lambda s: s[x_key])
        xs = [s[x_key] for s in rows]
        ys = [100 * s["accuracy"] for s in rows]
        ax.plot(xs, ys, "-o", ms=4, color=_color(label, i), label=label)
        for s in rows:
            if "tau" in s:
                ax.annotate(f"{s['tau']:g}", (s[x_key], 100 * s["accuracy"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
        i += 1
    for label, s in points.items():
        ax.errorbar(s[x_key], 100 * s["accuracy"], yerr=100 * s.get("accuracy_ci95", 0),
                    fmt="s", ms=7, capsize=3, color=_color(label, i), label=label)
        i += 1
    ax.set_xlabel(x_key.replace("avg_", "Average ").replace("_", " "))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def group_bar_plot(group_summaries, path, metric="accuracy", title="", xlabel="MATH level"):
    """Grouped bars of one metric per group (e.g. per MATH level) for each run.

    group_summaries: {run_label: {group_value: summary}}
    """
    labels = list(group_summaries)
    groups = sorted({g for gs in group_summaries.values() for g in gs}, key=str)
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, label in enumerate(labels):
        vals = [group_summaries[label].get(g, {}).get(metric, 0) for g in groups]
        if metric == "accuracy":
            vals = [100 * v for v in vals]
        ax.bar([j + i * width for j in range(len(groups))], vals, width,
               color=_color(label, i), label=label)
    ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in range(len(groups))])
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)" if metric == "accuracy" else metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)
