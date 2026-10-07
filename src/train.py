import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import argparse
import os
from src.model import MLPBaseline, GCNModel, GATModel


# 1. Edge splitting
def split_edges(edge_index, num_nodes, val_ratio=0.1, test_ratio=0.1):
    num_edges = edge_index.shape[1] // 2  # divide by 2 because bidirectional
    indices = torch.randperm(num_edges)

    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)

    val_idx = indices[:num_val]
    test_idx = indices[num_val : num_val + num_test]
    train_idx = indices[num_val + num_test :]

    # Only using one direction of edges (src < dst) to avoid leakage
    mask = edge_index[0] < edge_index[1]
    unique_edges = edge_index[:, mask]

    train_edges = unique_edges[:, train_idx]
    val_edges = unique_edges[:, val_idx]
    test_edges = unique_edges[:, test_idx]

    print(f"Train edges: {train_edges.shape[1]}")
    print(f"Val edges:   {val_edges.shape[1]}")
    print(f"Test edges:  {test_edges.shape[1]}")

    return train_edges, val_edges, test_edges


# 2. Evaluation
def evaluate(model, x, edge_index, pos_edges, neg_edges, device):
    model.eval()
    with torch.no_grad():
        pos_edges = pos_edges.to(device)
        neg_edges = neg_edges.to(device)

        pos_scores = torch.sigmoid(model(x, edge_index, pos_edges))
        neg_scores = torch.sigmoid(model(x, edge_index, neg_edges))

        scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        labels = torch.cat(
            [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
        ).numpy()

        roc_auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    return roc_auc, ap


# 3. Training loop
def train(model_name="gat", epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph
    data = torch.load("data/processed/ddi_graph.pt")
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = data.num_nodes

    # Split edges
    train_edges, val_edges, test_edges = split_edges(edge_index, num_nodes)
    train_edges = train_edges.to(device)
    val_edges = val_edges.to(device)
    test_edges = test_edges.to(device)

    # Initialize model
    if model_name == "mlp":
        model = MLPBaseline(input_dim=2048).to(device)
    elif model_name == "gcn":
        model = GCNModel(input_dim=2048).to(device)
    else:
        model = GATModel(input_dim=2048).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_auc = 0

    print(f"\nTraining {model_name.upper()} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()

        # Sample negative edges for this epoch
        neg_edges = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_edges.shape[1],
        ).to(device)

        # Forward pass
        pos_scores = model(x, edge_index, train_edges)
        neg_scores = model(x, edge_index, neg_edges)

        # Binary cross entropy loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat(
            [torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]
        ).to(device)

        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0:
            neg_val = negative_sampling(edge_index, num_nodes, val_edges.shape[1]).to(
                device
            )
            val_auc, val_ap = evaluate(model, x, edge_index, val_edges, neg_val, device)
            print(
                f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}"
            )

            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), f"models/best_{model_name}.pt")

    # Final test evaluation
    print(f"\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(f"models/best_{model_name}.pt"))
    neg_test = negative_sampling(edge_index, num_nodes, test_edges.shape[1]).to(device)
    test_auc, test_ap = evaluate(model, x, edge_index, test_edges, neg_test, device)
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Test Avg Precision: {test_ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="gat", choices=["mlp", "gcn", "gat"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    train(args.model, args.epochs, args.lr)
