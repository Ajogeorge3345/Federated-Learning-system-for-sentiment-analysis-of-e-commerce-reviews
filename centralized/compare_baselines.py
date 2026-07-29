"""
Phase 3: Compare frozen (feature extraction) vs unfrozen (full fine-tuning)
centralized DistilBERT baselines.

Run centralized/train.py in both modes first:

    python centralized/train.py            # unfrozen
    python centralized/train.py --freeze    # frozen

Then run this script:

    python centralized/compare_baselines.py
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "centralized")

METRICS_OF_INTEREST = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]


def load_metrics(mode: str):
    path = os.path.join(RESULTS_ROOT, mode, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    frozen = load_metrics("frozen")
    unfrozen = load_metrics("unfrozen")

    if frozen is None:
        print("No frozen results found yet. Run: python centralized/train.py --freeze")
    if unfrozen is None:
        print("No unfrozen results found yet. Run: python centralized/train.py")
    if frozen is None or unfrozen is None:
        return

    print(f"\n{'Metric':<18}{'Frozen':>12}{'Unfrozen':>12}{'Difference':>14}")
    print("-" * 56)
    for key in METRICS_OF_INTEREST:
        f_val = frozen.get(key, float("nan"))
        u_val = unfrozen.get(key, float("nan"))
        diff = u_val - f_val
        label = key.replace("eval_", "").capitalize()
        print(f"{label:<18}{f_val:>12.4f}{u_val:>12.4f}{diff:>+14.4f}")

    print(
        "\nInterpretation: unfrozen (full fine-tuning) almost always wins on "
        "raw performance, since it can adapt every layer to your domain. The "
        "interesting question for your report is *how much* it wins by -- a "
        "small gap suggests frozen features are 'good enough', which matters "
        "a lot for federated learning: frozen mode sends far less data per "
        "round (only the tiny classifier head), which could be a real "
        "communication-efficiency argument for your federated experiments."
    )


if __name__ == "__main__":
    main()
