"""
Phase 2: Centralized DistilBERT sentiment classifier.

This is the baseline you compare federated results against.
Run this FIRST, before touching Flower at all.

Usage:
    python centralized/train.py
"""

import os
import sys

import torch

# allow "python centralized/train.py" to import from utils/ at project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.dataset import ReviewDataset, load_csv_dataset, MODEL_NAME, get_tokenizer
from utils.metrics import compute_metrics

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "combined.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "centralized",
)


def main():
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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
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

    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    print(f"\nModel saved to {os.path.join(OUTPUT_DIR, 'final_model')}")


if __name__ == "__main__":
    main()