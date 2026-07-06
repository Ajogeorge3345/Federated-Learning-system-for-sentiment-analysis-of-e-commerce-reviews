# Federated Sentiment Analysis (DistilBERT + Flower)

Seminar project: Federated Sentiment Analysis on Distributed E-commerce
Reviews using DistilBERT and Flower.

## Roadmap (from planning doc)

- [x] Phase 1: Dataset preparation
- [x] Phase 2: Centralized DistilBERT baseline (skeleton ready)
- [ ] Phase 3: Evaluate baseline
- [ ] Phase 4: Flower setup
- [ ] Phase 5: Federated training
- [ ] Phase 6: IID experiment
- [ ] Phase 7: Non-IID experiment
- [ ] Phase 8: Analysis and report

## 1. Environment setup (Windows)

Your machine has Python 3.14.2, which is too new for some ML packages
(torch/transformers wheels lag behind new Python releases). Use Python 3.11
instead:

```powershell
# See which Python versions are installed
py -0

# If 3.11 is not listed, install it from python.org, then create the venv:
py -3.11 -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Phase 1: Download & preprocess data

```powershell
python data/download_data.py
```

This creates:
```
data/processed/electronics.csv
data/processed/books.csv
data/processed/clothing.csv
data/processed/beauty.csv
data/processed/combined.csv
```

Each row has: `text, rating, label, category`
(label: 0=Negative, 1=Neutral, 2=Positive)

## 3. Phase 2: Train the centralized baseline

```powershell
python centralized/train.py
```

This fine-tunes DistilBERT on the combined dataset and reports
accuracy / precision / recall / F1. Record these numbers — they're
your Experiment 1 baseline that federated results (Experiment 2) get
compared against.

## Next steps (not yet built)

- `federated/client.py`, `federated/server.py`, `federated/strategy.py` (Flower, Phase 4-5)
- IID vs non-IID client partitioning (Phase 6-7) — the `category` column in
  each CSV already gives you a natural non-IID split (one category per client);
  IID will need shuffling categories together and re-splitting evenly.
