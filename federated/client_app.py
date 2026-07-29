"""
Phase 4/5: Flower ClientApp.

Each simulated client (one per product category in non_iid mode) trains
locally on its own partition and reports back updated weights + metrics.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated.task import load_data, load_model, test_fn, train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the received global model on this client's local data."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    freeze = bool(context.run_config.get("freeze-base", False))
    partition_strategy = context.run_config["partition-strategy"]
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])

    model = load_model(freeze=freeze)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size, partition_strategy
    )

    train_loss = train_fn(
        model, trainloader, local_epochs, float(msg.content["config"]["lr"]), device
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition_id": partition_id,
    }
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on this client's local test split."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    freeze = bool(context.run_config.get("freeze-base", False))
    partition_strategy = context.run_config["partition-strategy"]
    batch_size = int(context.run_config["batch-size"])

    model = load_model(freeze=freeze)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, testloader = load_data(
        partition_id, num_partitions, batch_size, partition_strategy
    )

    loss, accuracy, f1 = test_fn(model, testloader, device)

    metrics = {
        "eval_loss": loss,
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        "num-examples": len(testloader.dataset),
        "partition_id": partition_id,
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
