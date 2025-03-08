import flwr as fl
import numpy as np

# -------------------------------
# Define Weighted Federated Averaging (WFedAvg)
# -------------------------------
def weighted_fedavg(metrics):
    """
    Computes weighted average of client updates based on dataset size.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_acc = sum(num_examples * m.get("accuracy", 0) for num_examples, m in metrics) / total_examples
    return {"accuracy": weighted_acc}  # Return weighted accuracy

# Use Weighted FedAvg instead of FedAvg
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_fedavg,  # Weighted accuracy
    fit_metrics_aggregation_fn=weighted_fedavg  # Weighted loss
)

# -------------------------------
# Start FL Server with WFedAvg
# -------------------------------
fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(num_rounds=10),  # Try more rounds
    strategy=strategy
)
