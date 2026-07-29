"""
Phase 8: Final comparison across all completed experiments --
centralized (frozen/unfrozen) and federated (non-IID). Reads whatever
metrics.json files exist; skips any experiment that isn't there yet
(e.g. if federated IID never completed).

Usage:
    python centralized/plot_results.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from sklearn.model_selection import train_test_split  # noqa: E402

import torch  # noqa: E402
from transformers import DistilBertForSequenceClassification  # noqa: E402

from utils.dataset import ReviewDataset, load_csv_dataset, get_tokenizer  # noqa: E402
from utils.plotting import plot_confusion_matrix, plot_metrics_comparison  # noqa: E402

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "combined.csv")
CENTRALIZED_ROOT = os.path.join(PROJECT_ROOT, "results", "centralized")
FEDERATED_ROOT = os.path.join(PROJECT_ROOT, "results", "federated")
PLOTS_DIR = os.path.join(CENTRALIZED_ROOT, "plots")


def get_predictions(model_dir: str, eval_df):
    tokenizer = get_tokenizer()
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = ReviewDataset(eval_df, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())

    return all_labels, all_preds


def main():
    df = load_csv_dataset(DATA_PATH)
    # SAME split (seed 42, 20%) used during training -- must match exactly
    # for these results to be valid
    _, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    comparison = {}

    for mode in ["frozen", "unfrozen"]:
        model_dir = os.path.join(CENTRALIZED_ROOT, mode, "final_model")
        metrics_path = os.path.join(CENTRALIZED_ROOT, mode, "metrics.json")

        if not os.path.isdir(model_dir):
            print(f"Skipping '{mode}': no saved model at {model_dir}")
            continue

        print(f"\nLoading '{mode}' model and running inference on {len(eval_df)} examples...")
        y_true, y_pred = get_predictions(model_dir, eval_df)

        plot_confusion_matrix(
            y_true, y_pred,
            title=f"Centralized ({mode})",
            save_path=os.path.join(PLOTS_DIR, f"confusion_matrix_{mode}.png"),
        )

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            comparison[f"Centralized ({mode})"] = {
                "accuracy": m.get("eval_accuracy", 0),
                "precision": m.get("eval_precision", 0),
                "recall": m.get("eval_recall", 0),
                "f1": m.get("eval_f1", 0),
            }

    # Pull in completed federated results. NOTE: "iid" is deliberately
    # excluded here -- that run never got real client responses (a Windows
    # deployment-mode connectivity issue, documented in the report as a
    # known limitation) and its metrics.json reflects an untrained model,
    # not a real result. Only non_iid is a valid, trained result.
    for mode in ["non_iid"]:
        metrics_path = os.path.join(FEDERATED_ROOT, mode, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"Skipping federated '{mode}': no metrics.json found (run didn't complete)")
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        comparison[f"Federated ({mode})"] = {
            "accuracy": m.get("central_eval_accuracy", 0),
            "precision": m.get("central_eval_precision", 0),
            "recall": m.get("central_eval_recall", 0),
            "f1": m.get("central_eval_f1", 0),
        }

    if comparison:
        plot_metrics_comparison(
            comparison,
            save_path=os.path.join(PLOTS_DIR, "all_experiments_comparison.png"),
            title="All Experiments: Centralized vs Federated",
        )

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
