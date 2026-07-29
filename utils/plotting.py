"""
Shared plotting utilities used across centralized and federated experiments.
Every plot is saved as a PNG (not just shown), since these all get used in
Phase 8 (the report).
"""

import os
import matplotlib
matplotlib.use("Agg")  # no display needed -- just save files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

LABEL_NAMES = ["Negative", "Neutral", "Positive"]


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # Row-normalize so classes with fewer examples (Negative/Neutral) are
    # still readable, not swamped visually by the much larger Positive class
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp_counts = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
    disp_counts.plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title(f"{title}\n(raw counts)")

    disp_norm = ConfusionMatrixDisplay(cm_normalized, display_labels=LABEL_NAMES)
    disp_norm.plot(ax=axes[1], cmap="Blues", colorbar=False, values_format=".2f")
    axes[1].set_title(f"{title}\n(row-normalized)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {save_path}")


def plot_metrics_comparison(results: dict, save_path: str,
                             title: str = "Experiment Comparison"):
    """
    results: {"experiment_name": {"accuracy": .., "precision": .., "recall": .., "f1": ..}, ...}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    experiment_names = list(results.keys())

    x = np.arange(len(metric_names))
    width = 0.8 / len(experiment_names)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, exp_name in enumerate(experiment_names):
        values = [results[exp_name].get(m, 0.0) for m in metric_names]
        offset = (i - (len(experiment_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=exp_name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metric_names])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics comparison -> {save_path}")


def plot_label_distribution(label_counts: dict, save_path: str,
                             title: str = "Label Distribution by Client"):
    """
    label_counts: {"client_name": {0: count, 1: count, 2: count}, ...}
    Useful for visualizing the IID vs non-IID data skew (Phase 6/7).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clients = list(label_counts.keys())
    x = np.arange(len(clients))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label_id in enumerate([0, 1, 2]):
        values = [label_counts[c].get(label_id, 0) for c in clients]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=LABEL_NAMES[label_id])

    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.set_ylabel("Number of examples")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved label distribution -> {save_path}")
