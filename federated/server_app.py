"""
Phase 4/5/6/7: Flower ServerApp.

Runs FedAvg across the 4 simulated clients, evaluates the aggregated
global model on the same held-out test split used by the centralized
baselines (so results are directly comparable), and saves final
metrics.json under results/federated/<partition-strategy>/.
"""

import json
import os
import sys

# Flower installs the app into an isolated directory (e.g. under
# C:\Users\<you>\.flwr\apps\...) and runs it from there -- NOT from your
# actual project folder. That's fine for reading code/data (which gets
# bundled into the install via force-include in pyproject.toml), but we
# do NOT want results saved into that obscure hashed directory where
# you'd never find them. Set FEDSENT_PROJECT_ROOT to your real project
# path before running `flwr run` so results land in the usual place:
#
#   $env:FEDSENT_PROJECT_ROOT = (Get-Location).Path
#
# Falls back to the installed copy's own location if unset (works, just
# less convenient to find).
INSTALLED_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.environ.get("FEDSENT_PROJECT_ROOT", INSTALLED_ROOT)
sys.path.append(INSTALLED_ROOT)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402

import torch  # noqa: E402
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord  # noqa: E402
from flwr.serverapp import Grid, ServerApp  # noqa: E402
from flwr.serverapp.strategy import FedAvg  # noqa: E402

from federated.task import load_central_testloader, load_model  # noqa: E402

app = ServerApp()

# Set by main() before strategy.start() calls global_evaluate
_EVAL_STATE = {}


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the aggregated global model on the centralized held-out set."""
    model = _EVAL_STATE["model"]
    testloader = _EVAL_STATE["testloader"]
    device = _EVAL_STATE["device"]

    model.load_state_dict(arrays.to_torch_state_dict())
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(f"[Round {server_round}] central eval -- "
          f"loss={avg_loss:.4f} acc={accuracy:.4f} f1={f1:.4f}")

    return MetricRecord({
        "central_eval_loss": avg_loss,
        "central_eval_accuracy": accuracy,
        "central_eval_precision": precision,
        "central_eval_recall": recall,
        "central_eval_f1": f1,
    })


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds: int = int(context.run_config["num-server-rounds"])
    lr: float = float(context.run_config["learning-rate"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    freeze: bool = bool(context.run_config.get("freeze-base", False))
    partition_strategy: str = context.run_config["partition-strategy"]
    save_model: bool = bool(context.run_config.get("save-model", True))

    print(f"\n=== Federated run: partition-strategy={partition_strategy}, "
          f"freeze-base={freeze}, rounds={num_rounds} ===\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global_model = load_model(freeze=freeze)
    _EVAL_STATE["model"] = load_model(freeze=freeze)
    _EVAL_STATE["testloader"] = load_central_testloader()
    _EVAL_STATE["device"] = device

    arrays = ArrayRecord(global_model.state_dict())
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    output_dir = os.path.join(
        PROJECT_ROOT, "results", "federated", partition_strategy
    )
    os.makedirs(output_dir, exist_ok=True)

    if save_model:
        state_dict = result.arrays.to_torch_state_dict()
        model_path = os.path.join(output_dir, "final_model.pt")
        torch.save(state_dict, model_path)
        print(f"\nSaved final global model -> {model_path}")

    # Pull out the last round's central-eval metrics for the summary file
    final_metrics = {}
    if result.evaluate_metrics_serverapp:
        last_round = max(result.evaluate_metrics_serverapp.keys())
        final_metrics = dict(result.evaluate_metrics_serverapp[last_round])

    summary = {
        "mode": f"federated_{partition_strategy}" + ("_frozen" if freeze else ""),
        "partition_strategy": partition_strategy,
        "freeze_base": freeze,
        "num_rounds": num_rounds,
        **final_metrics,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")
