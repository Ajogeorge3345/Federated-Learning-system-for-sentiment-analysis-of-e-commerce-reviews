"""
Phase 2/3: Centralized DistilBERT sentiment classifier.

Supports two modes, per your professor's recommendation to compare both
before moving to federated experiments:

    --freeze     Feature extraction: DistilBERT's base weights are frozen,
                 only the classification head is trained.
    (default)    Full fine-tuning: all weights are trainable (what you
                 already ran as the Phase 2 baseline).

This is the baseline you compare federated results against.
Run this FIRST, before touching Flower at all.

Usage:
    python centralized/train.py                # unfrozen (full fine-tune)
    python centralized/train.py --freeze        # frozen (feature extraction)
"""

import argparse
import json
import os
import sys

# allow "python centralized/train.py" to import from utils/ at project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: on Windows, import sklearn BEFORE torch -- importing torch first can
# cause a native DLL conflict (access violation, exit code -1073741819)
# between torch's bundled MKL/OpenMP runtime and scikit-learn's.
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.dataset import ReviewDataset, load_csv_dataset, MODEL_NAME, get_tokenizer
from utils.metrics import compute_metrics
from utils.plotting import plot_confusion_matrix

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "combined.csv",
)
RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "centralized",
)


def freeze_base_model(model: DistilBertForSequenceClassification) -> None:
    """Freeze all DistilBERT encoder weights; leave classifier head trainable."""
    for name, param in model.named_parameters():
        if name.startswith("distilbert."):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen mode: {trainable:,} / {total:,} parameters trainable "
          f"({100 * trainable / total:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze DistilBERT base weights (feature extraction) instead of "
             "full fine-tuning.",
    )
    args = parser.parse_args()

    mode = "frozen" if args.freeze else "unfrozen"
    output_dir = os.path.join(RESULTS_ROOT, mode)
    print(f"\n=== Mode: {mode} ===\n")

    print(f"Loading data from {DATA_PATH} ...")
    df = load_csv_dataset(DATA_PATH)

    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train size: {len(train_df)}  |  Eval size: {len(eval_df)}")

    tokenizer = get_tokenizer()
    train_dataset = ReviewDataset(train_df, tokenizer)
    eval_dataset = ReviewDataset(eval_df, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3  # Negative / Neutral / Positive
    )

    if args.freeze:
        freeze_base_model(model)
        # Only a small head is trained -> a higher LR converges much faster
        learning_rate = 1e-3
    else:
        learning_rate = 2e-5

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=torch.cuda.is_available(),  # mixed precision speedup on GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    # Confusion matrix from the same eval set, using the final model
    predictions = trainer.predict(eval_dataset)
    y_pred = predictions.predictions.argmax(axis=-1)
    y_true = predictions.label_ids
    plot_confusion_matrix(
        y_true, y_pred,
        title=f"Centralized ({mode})",
        save_path=os.path.join(RESULTS_ROOT, "plots", f"confusion_matrix_{mode}.png"),
    )

    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

    # Save metrics as JSON so Phase 3's comparison script can read them back
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"mode": mode, **metrics}, f, indent=2)

    print(f"\nModel saved to {os.path.join(output_dir, 'final_model')}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

