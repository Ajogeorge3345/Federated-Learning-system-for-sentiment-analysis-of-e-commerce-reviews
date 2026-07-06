"""
Shared dataset utilities used by both the centralized baseline
and (later) the federated clients.
"""

import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    return _tokenizer


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"<.*?>", " ", text)               # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text


class ReviewDataset(Dataset):
    """Wraps a pandas DataFrame with 'text' and 'label' columns."""

    def __init__(self, df: pd.DataFrame, tokenizer=None, max_length: int = MAX_LENGTH):
        self.texts = [clean_text(t) for t in df["text"].tolist()]
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer or get_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_csv_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df
