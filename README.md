# Federated Sentiment Analysis (DistilBERT + Flower)

Seminar project: Federated Sentiment Analysis on Distributed E-commerce
Reviews using DistilBERT and Flower.

## Roadmap (from planning doc)

- [x] Phase 1: Dataset preparation
- [x] Phase 2: Centralized DistilBERT baseline (unfrozen -- 86.5% acc / 67.0% F1)
- [x] Phase 3: Frozen vs unfrozen comparison (frozen -- 85.4% acc / 54.2% F1)
- [x] Phase 4: Flower setup (installed, flwr 1.32.1) -- federated/ code ready, untested
- [ ] Phase 5: Federated training (run non_iid first, it's the default)
- [ ] Phase 6: IID experiment (`--run-config "partition-strategy=\"iid\""`)
- [ ] Phase 7: Non-IID experiment (default config)
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

## 4. Phase 4-7: Federated learning with Flower

New files: `federated/task.py`, `federated/client_app.py`,
`federated/server_app.py`, and `pyproject.toml` at the project root.
This uses Flower's current (1.32) `ClientApp`/`ServerApp` "Message API" --
not the older `NumPyClient`/`start_simulation` pattern you may see in
older tutorials.

First, install the app itself (registers the `federated` package so
`flwr run` can find it):

```powershell
pip install -e .
```

Then run a federated simulation. Flower installs your app into an isolated
directory under `.flwr\apps\` and runs it from there, so set this env var
first so results still get saved back into *this* project folder instead
of disappearing into that isolated copy:

```powershell
$env:FEDSENT_PROJECT_ROOT = (Get-Location).Path
```

Use the GPU federation profile so it doesn't fall back to painfully slow
CPU training:

```powershell
flwr run . local-simulation-gpu --stream
```

This runs **non-IID** (one product category per client) by default, since
that's set in `pyproject.toml`. To run the **IID** experiment instead,
override it on the command line:

```powershell
flwr run . local-simulation-gpu --stream --run-config "partition-strategy=\"iid\""
```

You can also override any other hyperparameter the same way, e.g. more
rounds:

```powershell
flwr run . local-simulation-gpu --stream --run-config "num-server-rounds=5"
```

Each run prints per-round central-eval metrics (accuracy/F1 on the same
held-out test split as your centralized baselines) and saves a final
`metrics.json` + `final_model.pt` under
`results/federated/<non_iid|iid>/` -- directly comparable to
`results/centralized/<frozen|unfrozen>/metrics.json` from Phase 2-3.

If `flwr run` fails, paste the full error -- Flower's simulation engine
uses Ray under the hood, which has its own set of Windows quirks separate
from the grpcio-tools issue we already dodged.

## 3. Phase 2 & 3: Train the centralized baselines (frozen vs unfrozen)

Your professor recommended comparing frozen (feature extraction) vs
unfrozen (full fine-tuning) DistilBERT before the federated experiments.
Run both:

```powershell
python centralized/train.py            # unfrozen: full fine-tuning
python centralized/train.py --freeze     # frozen: only classifier head trains
```

Each saves its own model + metrics under `results/centralized/<mode>/`.
Then compare them:

```powershell
python centralized/compare_baselines.py
```

Record all four numbers (accuracy/precision/recall/F1 x frozen/unfrozen) —
these are two of your four baseline experiments. Federated IID and
federated non-IID (Phase 6-7) get compared against both.

## Next steps (not yet built)

- `federated/client.py`, `federated/server.py`, `federated/strategy.py` (Flower, Phase 4-5)
- IID vs non-IID client partitioning (Phase 6-7) — the `category` column in
  each CSV already gives you a natural non-IID split (one category per client);
  IID will need shuffling categories together and re-splitting evenly.
