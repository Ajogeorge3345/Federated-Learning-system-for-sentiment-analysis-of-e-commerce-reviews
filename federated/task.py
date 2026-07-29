"""
Phase 4/5/6/7: Shared logic for the federated ClientApp and ServerApp.

Supports two partitioning strategies over the same 4 clients you already
built in Phase 1 (electronics, books, clothing, beauty):

    non_iid  -> each client gets exactly one category (realistic FL, Phase 7)
    iid      -> all categories shuffled together and split evenly (Phase 6)

Switch between them via pyproject.toml's [tool.flwr.app.config]
partition-strategy value, or override on the command line:

    flwr run . --run-config "partition-strategy=\"iid\""
    flwr run . --run-config "partition-strategy=\"non_iid\""
"""

import os
import sys

# Flower installs the app into an isolated directory (under
# C:\Users\<you>\.flwr\apps\...) and runs it from there -- NOT from your
# actual project folder. Bundling data/ into that install via pyproject.toml
# packaging turned out to be unreliable, so instead we just read data
# straight from your real project folder via this env var. Set it in EVERY
# terminal that runs a Flower process (SuperLink, every SuperNode, and the
# terminal running `flwr run`) before starting anything:
#
#   $env:FEDSENT_PROJECT_ROOT = "D:\project\Federated-Learning-system-for-sentiment-analysis-of-e-commerce-reviews"
#
# Falls back to the installed copy's own location if unset (won't find the
# data there, but avoids a crash on import).
INSTALLED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.environ.get("FEDSENT_PROJECT_ROOT", INSTALLED_ROOT)
sys.path.append(INSTALLED_ROOT)

# NOTE: same Windows DLL-conflict reasoning as centralized/train.py --
# import sklearn before torch.
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification

from utils.dataset import ReviewDataset, MODEL_NAME, get_tokenizer

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# Fixed order -> partition_id 0..3 maps onto these categories in non_iid mode.
# Must match num-supernodes=4 in the federation config.
CATEGORIES = ["electronics", "books", "clothing", "beauty"]


def load_model(freeze: bool = False) -> DistilBertForSequenceClassification:
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )
    if freeze:
        for name, param in model.named_parameters():
            if name.startswith("distilbert."):
                param.requires_grad = False
    return model


def _partition_dataframe(partition_id: int, num_partitions: int,
                          partition_strategy: str) -> pd.DataFrame:
    if partition_strategy == "non_iid":
        if num_partitions != len(CATEGORIES):
            raise ValueError(
                f"non_iid mode expects num-supernodes={len(CATEGORIES)} "
                f"(one per category), got {num_partitions}"
            )
        category = CATEGORIES[partition_id]
        path = os.path.join(DATA_DIR, f"{category}.csv")
        return pd.read_csv(path)

    elif partition_strategy == "iid":
        combined_path = os.path.join(DATA_DIR, "combined.csv")
        df = pd.read_csv(combined_path)
        # Shuffle once, deterministically, then slice into equal chunks
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        chunk_size = len(df) // num_partitions
        start = partition_id * chunk_size
        end = start + chunk_size if partition_id < num_partitions - 1 else len(df)
        return df.iloc[start:end].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown partition_strategy: {partition_strategy}")


def _safe_train_test_split(df, test_size=0.2, random_state=42):
    """Stratified split, falling back to a plain (non-stratified) split if
    any class is too small to stratify -- avoids crashing an entire client
    over one rare label in a small partition."""
    try:
        return train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df["label"],
        )
    except ValueError:
        return train_test_split(
            df, test_size=test_size, random_state=random_state
        )
def load_data(partition_id: int, num_partitions: int, batch_size: int,
              partition_strategy: str = "non_iid"):
    """Returns (trainloader, testloader) for one client's local partition."""
    df = _partition_dataframe(partition_id, num_partitions, partition_strategy)
    df = df.dropna(subset=["text", "label"])

    # Quick-iteration escape hatch: set FEDSENT_DEBUG_SAMPLES=50 (or any
    # small number) to cap each client's data for a fast sanity-check run
    # (minutes instead of an hour), without touching real experiment runs.
    debug_cap = os.environ.get("FEDSENT_DEBUG_SAMPLES")
    if debug_cap:
        df = df.sample(n=min(int(debug_cap), len(df)), random_state=42)

    train_df, test_df = _safe_train_test_split(df, test_size=0.2, random_state=42)

    tokenizer = get_tokenizer()
    train_dataset = ReviewDataset(train_df, tokenizer)
    test_dataset = ReviewDataset(test_df, tokenizer)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size * 2)
    return trainloader, testloader


def load_central_testloader(batch_size: int = 32):
    """Held-out test set for server-side evaluation -- SAME split (seed 42,
    20%) used by centralized/train.py, so federated results are directly
    comparable to your Phase 2/3 centralized baselines."""
    combined_path = os.path.join(DATA_DIR, "combined.csv")
    df = pd.read_csv(combined_path).dropna(subset=["text", "label"])

    debug_cap = os.environ.get("FEDSENT_DEBUG_SAMPLES")
    if debug_cap:
        df = df.sample(n=min(int(debug_cap) * 4, len(df)), random_state=42)

    _, test_df = _safe_train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = get_tokenizer()
    test_dataset = ReviewDataset(test_df, tokenizer)
    return DataLoader(test_dataset, batch_size=batch_size)


def train_fn(model, trainloader, epochs: int, lr: float, device) -> float:
    """One or more local epochs of standard supervised fine-tuning."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    running_loss, num_batches = 0.0, 0
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

    return running_loss / max(num_batches, 1)


def test_fn(model, testloader, device):
    """Returns (avg_loss, accuracy, f1_macro) on the given dataloader."""
    model.to(device)
    model.eval()

    total_loss, num_batches = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return avg_loss, accuracy, f1
