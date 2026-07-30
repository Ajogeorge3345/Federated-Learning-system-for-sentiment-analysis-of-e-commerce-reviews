# Federated Sentiment Analysis (DistilBERT + Flower)

Seminar project (L.E.A.D.D., Universität Bielefeld): Federated Sentiment
Analysis on Distributed E-commerce Reviews using DistilBERT and Flower.
Compares centralized (frozen vs. unfrozen) training against Federated
Learning (FedAvg) across four simulated clients built from real Amazon
product categories, under both non-IID and IID client data.

Full written report: see `report.tex` / `report.pdf`.

## Roadmap (final status)

- [x] Phase 1: Dataset preparation
- [x] Phase 2: Centralized DistilBERT baseline (unfrozen -- 86.7% acc / 67.3% F1 macro)
- [x] Phase 3: Frozen vs unfrozen comparison (frozen -- 85.4% acc / 54.2% F1 macro)
- [x] Phase 4: Flower setup -- switched from simulation to **deployment mode**
      (see [Deployment mode, not simulation](#why-deployment-mode-not-simulation) below)
- [x] Phase 5: Federated training pipeline built and working end-to-end
- [x] Phase 7: Non-IID experiment -- **completed**, 5 rounds, 81.0% acc / 29.8% F1 macro
      (see [Known limitation: majority-class collapse](#known-limitation-majority-class-collapse-under-non-iid))
- [ ] Phase 6: IID experiment -- **not completed**, blocked by a platform-level
      connectivity fault (see [Known issues](#known-issues--troubleshooting))
- [x] Phase 8: Analysis and report

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

If `pip install` fails while building `grpcio-tools`, it's almost always
because some other pinned dependency is dragging in an old `grpcio-tools`
version that has no prebuilt Windows wheel and falls back to a source
build. Don't pin old versions of unrelated packages -- let pip resolve a
current `grpcio-tools`.

**Import order matters on Windows.** If you see a process crash with exit
code `0xC0000005` and no Python traceback (just the process silently
dying), it's a DLL conflict between PyTorch's and scikit-learn's bundled
OpenMP/MKL runtimes. Fix: `import sklearn` before `import torch` anywhere
both are used, and set:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
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

The dataset is `McAuley-Lab/Amazon-Reviews-2023` (5,000 reviews per
category, 20,000 total). Its Hugging Face loading script is deprecated
under current `datasets` versions -- `download_data.py` bypasses it and
streams the raw `.jsonl` files directly via the generic `"json"` loader,
so only the first 5,000 rows per category are ever pulled.

## 3. Phase 2 & 3: Train the centralized baselines (frozen vs unfrozen)

```powershell
python centralized/train.py            # unfrozen: full fine-tuning
python centralized/train.py --freeze     # frozen: only classifier head trains
```

Each saves its own model + metrics under `results/centralized/<mode>/`.
Then compare them:

```powershell
python centralized/compare_baselines.py
```

| Mode     | Accuracy | Precision | Recall | F1 (macro) | Train time |
|----------|----------|-----------|--------|------------|------------|
| Frozen   | 85.4%    | 63.3%     | 53.3%  | 54.2%      | 3m 43s     |
| Unfrozen | 86.7%    | 68.6%     | 66.3%  | 67.3%      | 18m 03s    |

Frozen only trains the 592,899-parameter classification head (0.89% of
the model); the gap between the two is concentrated almost entirely in
the Neutral class (5% recall frozen vs. 41% unfrozen) -- see the report
for the full confusion-matrix breakdown.

## 4. Phase 4-7: Federated learning with Flower

Files: `federated/task.py`, `federated/client_app.py`,
`federated/server_app.py`, `pyproject.toml` at the project root. Uses
Flower's current (1.32) `ClientApp`/`ServerApp` "Message API" -- not the
older `NumPyClient`/`start_simulation` pattern from older tutorials.

Install the app itself (registers the `federated` package so `flwr run`
can find it):

```powershell
pip install -e .
```

### Why deployment mode, not simulation

The original plan was to use Flower's simulation engine (all clients as
virtual processes in one Python process, via Ray, with a GPU profile).
**This never worked on Windows** -- Ray's simulation mode hit repeated,
unresolved access-violation crashes that are documented as a known,
long-standing class of bug in Ray's own GitHub issues, unrelated to this
project's code. After losing significant time to this, the pipeline was
switched to Flower's **deployment mode** instead: genuinely separate
processes (one `SuperLink` for the server, one `SuperNode` per client)
talking over real gRPC connections on `localhost`, with no Ray involved.

Because deployment mode doesn't share GPU context across processes the
way the simulation profile did, federated training runs on **CPU**, not
GPU. With 4 clients training sequentially per round, each round takes
roughly 25-30 minutes.

Flower's deployment mode installs your app into an isolated directory
under `~/.flwr/apps/...`, separate from this project folder, and doesn't
reliably bundle data files there. Both the client and server code read
the dataset and results directories from an explicit env var instead of
relying on Flower's packaging -- set this first, every session:

```powershell
$env:FEDSENT_PROJECT_ROOT = (Get-Location).Path
```

### Running it

Start the server:

```powershell
flwr-superlink --insecure
```

In separate terminals, start four SuperNodes (one per client):

```powershell
flwr-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9094
flwr-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9095
flwr-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9096
flwr-supernode --insecure --superlink 127.0.0.1:9092 --clientappio-api-address 127.0.0.1:9097
```

Then launch the run:

```powershell
flwr run . local-deployment --stream
```

This runs **non-IID** (one product category per client) by default, per
`pyproject.toml`. To run **IID** instead, override it on the command
line -- **note this is the config that currently hangs, see below**:

```powershell
flwr run . local-deployment --stream --run-config "partition-strategy=\"iid\""
```

PowerShell quotes command-line overrides differently from bash --
`\"iid\"` above is the PowerShell-safe form. If an override silently has
no effect, edit the default value in `pyproject.toml` directly instead of
fighting the quoting.

Each run prints per-round central-eval metrics (accuracy/precision/recall
/F1 on the same held-out test split as the centralized baselines) and
saves `metrics.json` + `final_model.pt` under
`results/federated/<non_iid|iid>/`.

## Results

| Experiment            | Accuracy | Precision | Recall | F1 (macro) | Status         |
|------------------------|----------|-----------|--------|------------|----------------|
| Centralized, frozen    | 85.4%    | 63.3%     | 53.3%  | 54.2%      | Done           |
| Centralized, unfrozen  | 86.7%    | 68.6%     | 66.3%  | 67.3%      | Done           |
| Federated, non-IID     | 81.0%    | 27.0%     | 33.3%  | 29.8%      | Done           |
| Federated, IID         | --       | --        | --     | --         | Not completed  |

### Known limitation: majority-class collapse under non-IID

From round 2 onward, the non-IID federated model's precision and recall
lock at exactly 0.270 / 0.333 -- the exact signature of a model that has
collapsed to always predicting "Positive" (the ~81% majority class),
even though training loss keeps dropping every round. This matches the
client-drift failure mode documented for FedAvg on non-IID data: each
client only sees one product category, so local models drift apart
during local training, and averaging them collapses onto the one thing
they all agree on. Full derivation and discussion in the report
(Discussion, Section 5.2).

## Known issues / troubleshooting

- **`grpcio-tools` build failure**: see [Environment setup](#1-environment-setup-windows) above.
- **Silent `0xC0000005` crash, no traceback**: PyTorch/scikit-learn DLL conflict, see above.
- **Ray simulation-mode crashes on Windows**: unresolved upstream Ray bug; use deployment mode instead (see [Why deployment mode](#why-deployment-mode-not-simulation)).
- **Flower deployment-mode packaging doesn't see project files**: set `FEDSENT_PROJECT_ROOT` before every `flwr run`.
- **PowerShell `--run-config` override has no effect**: quoting difference from bash; edit `pyproject.toml` directly if in doubt.
- **`ValueError: least populated class has only 1 member` on train/test split**: happens when a client partition is small; the split code falls back to a non-stratified split automatically -- if you see this uncaught, you're on an older version of `federated/task.py`, pull latest.
- **IID run (`partition-strategy="iid"`) never returns round results**: known unresolved issue. Both attempts ran for ~10.3 hours (within 5 seconds of each other) before being killed, which points to a repeated communication timeout rather than genuinely slow training -- not yet root-caused. If you get further on this, please update this section. Suspected next step: try on native Linux or WSL2 to rule out a Windows-specific gRPC/socket problem.

## Next steps

- Root-cause and complete the IID experiment (Linux/WSL2 first, to rule out Windows networking as the cause).
- Try FedProx or another heterogeneity-aware aggregation strategy against the non-IID majority-class collapse.
- Run the federated pipeline with `--freeze` (frozen head only) to measure the communication-efficiency trade-off (593K vs 66.9M parameters per round).
- More than 5 communication rounds, ideally with GPU-accelerated client training, to check whether the F1 plateau is a durable ceiling or just needs more rounds.

## Repository

Code: https://github.com/Ajogeorge3345/Federated-Learning-system-for-sentiment-analysis-of-e-commerce-reviews.git
