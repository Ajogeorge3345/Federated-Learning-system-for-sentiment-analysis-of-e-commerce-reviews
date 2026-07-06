"""
Phase 1: Dataset preparation.

Downloads a subset of the McAuley-Lab/Amazon-Reviews-2023 dataset
for 4 product categories that will later become our 4 simulated FL clients:

    Client 1 -> Electronics
    Client 2 -> Books (using "Books" category)
    Client 3 -> Clothing_Shoes_and_Jewelry
    Client 4 -> All_Beauty

For each category we keep only the review text + rating, map the rating
to a sentiment label, and save a clean CSV to data/processed/<category>.csv

Rating -> Sentiment mapping (per project spec):
    1-2 -> Negative (0)
    3   -> Neutral  (1)
    4-5 -> Positive (2)

NOTE ON IMPLEMENTATION:
The `datasets` library (>=4.0) dropped support for script-based dataset
loaders, and McAuley-Lab/Amazon-Reviews-2023 still ships one (a leftover
"Amazon-Reviews-2023.py"), which makes `load_dataset("McAuley-Lab/...")`
fail even though the repo *also* has a valid YAML config mapping category
names to plain .jsonl files.

Workaround: bypass the repo's custom script entirely by using datasets'
built-in generic "json" loader, pointed directly at the raw file URL, with
streaming=True. Files here are huge (Books.jsonl alone is ~20 GB), so we
only pull the first N examples and never materialize the rest.
"""

import os
import pandas as pd
from datasets import load_dataset

# Map our friendly client names to the actual file names in the dataset repo
# (see raw/review_categories/*.jsonl at
#  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
CATEGORY_FILES = {
    "electronics": "Electronics",
    "books": "Books",
    "clothing": "Clothing_Shoes_and_Jewelry",
    "beauty": "All_Beauty",
}

BASE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/{}.jsonl"
)

# How many reviews to sample per category (keep small while iterating!)
SAMPLES_PER_CATEGORY = 5000

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")


def rating_to_sentiment(rating: float) -> int:
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def process_category(client_name: str, file_stem: str, n_samples: int) -> pd.DataFrame:
    url = BASE_URL.format(file_stem)
    print(f"\n[{client_name}] Streaming {url} ...")

    # "json" is datasets' built-in generic loader -- no custom script involved,
    # so it isn't affected by the repo's deprecated Amazon-Reviews-2023.py
    ds = load_dataset("json", data_files=url, split="train", streaming=True)

    rows = []
    for example in ds:
        if len(rows) >= n_samples:
            break
        text = (example.get("text") or "").strip()
        rating = example.get("rating")
        if not text or rating is None:
            continue
        rows.append(
            {
                "text": text,
                "rating": rating,
                "label": rating_to_sentiment(rating),
                "category": client_name,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[{client_name}] Collected {len(df)} reviews. "
          f"Label distribution:\n{df['label'].value_counts()}")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_frames = []
    for client_name, file_stem in CATEGORY_FILES.items():
        df = process_category(client_name, file_stem, SAMPLES_PER_CATEGORY)
        out_path = os.path.join(OUTPUT_DIR, f"{client_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[{client_name}] Saved -> {out_path}")
        all_frames.append(df)

    # Also save one combined file, useful for the centralized baseline
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined dataset ({len(combined)} rows) -> {combined_path}")


if __name__ == "__main__":
    main()
