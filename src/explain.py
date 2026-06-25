import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('.')

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from src.model import GATModel, GCNModel, MLPBaseline
from src.dataset import load_twosides, build_drug_vocab

# ── Helpers ───────────────────────────────────────────────────────────

def load_graph(device):
    data = torch.load(
        'data/processed/ddi_graph.pt',
        weights_only=False
    )
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    return x, edge_index

def load_model(model_name, device):
    if model_name == 'mlp':
        model = MLPBaseline(input_dim=2048).to(device)
    elif model_name == 'gcn':
        model = GCNModel(input_dim=2048).to(device)
    else:
        model = GATModel(input_dim=2048).to(device)

    path = f'models/best_{model_name}.pt'
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=False)
    )
    model.eval()
    print(f"Loaded {model_name.upper()} from {path}")
    return model

def get_prediction(model, x, edge_index, drug1_idx, drug2_idx, device):
    target = torch.tensor(
        [[drug1_idx], [drug2_idx]],
        dtype=torch.long
    ).to(device)
    with torch.no_grad():
        score = torch.sigmoid(
            model(x, edge_index, target)
        ).item()
    return score

# ── Part 1: GNNExplainer on GCN ──────────────────────────────────────

def explain_with_gnn(drug1_idx, drug2_idx, device):
    from torch_geometric.explain import Explainer, GNNExplainer

    print("\n" + "="*50)
    print("PART 1: GNNExplainer on GCN Model")
    print("="*50)

    x, edge_index = load_graph(device)
    model = load_model('gcn', device)

    score = get_prediction(model, x, edge_index, drug1_idx, drug2_idx, device)
    print(f"GCN Interaction probability: {score*100:.1f}%")

    # Wrap model for GNNExplainer compatibility
    class GCNWrapper(torch.nn.Module):
        def __init__(self, model, target_edge):
            super().__init__()
            self.model = model
            self.target_edge = target_edge

        def forward(self, x, edge_index):
            return self.model(x, edge_index, self.target_edge)

    target_edge = torch.tensor(
        [[drug1_idx], [drug2_idx]],
        dtype=torch.long
    ).to(device)

    wrapped = GCNWrapper(model, target_edge)

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'
        )
    )

    explanation = explainer(x=x, edge_index=edge_index)

    # Top influential nodes (neighboring drugs)
    if explanation.node_mask is not None:
        node_importance = explanation.node_mask.sum(dim=1).cpu().numpy()
        top_nodes = np.argsort(node_importance)[::-1][:5]
        print(f"\nTop 5 most influential drug nodes:")
        for rank, node_idx in enumerate(top_nodes):
            if node_idx not in [drug1_idx, drug2_idx]:
                print(f"  {rank+1}. Drug node {node_idx} (importance: {node_importance[node_idx]:.4f})")

    # Top influential fingerprint features for each drug
    if explanation.node_mask is not None:
        for drug_idx, label in [(drug1_idx, 'Drug 1'), (drug2_idx, 'Drug 2')]:
            feat_importance = explanation.node_mask[drug_idx].cpu().numpy()
            top_bits = np.argsort(feat_importance)[::-1][:5]
            print(f"\nTop 5 fingerprint bits for {label} (node {drug_idx}):")
            for rank, bit in enumerate(top_bits):
                print(f"  {rank+1}. Bit {bit}: {feat_importance[bit]:.4f}")

    return score, explanation

# ── Part 2: SHAP on MLP ───────────────────────────────────────────────

def explain_with_shap(drug1_idx, drug2_idx, device):
    import shap

    print("\n" + "="*50)
    print("PART 2: SHAP Explanation on MLP Model")
    print("="*50)

    x, edge_index = load_graph(device)
    model = load_model('mlp', device)

    # Original model prediction
    score = get_prediction(
        model,
        x,
        edge_index,
        drug1_idx,
        drug2_idx,
        device
    )

    print(f"MLP Interaction probability: {score*100:.1f}%")

    # Build pair feature vector
    x_cpu = x.cpu()

    pair_feat = torch.cat(
        [
            x_cpu[drug1_idx],
            x_cpu[drug2_idx]
        ]
    ).numpy().reshape(1, -1)

    # Background dataset
    num_nodes = x_cpu.shape[0]

    np.random.seed(42)

    bg_idx1 = np.random.randint(
        0,
        num_nodes,
        100
    )

    bg_idx2 = np.random.randint(
        0,
        num_nodes,
        100
    )

    background = np.hstack([
        x_cpu[bg_idx1].numpy(),
        x_cpu[bg_idx2].numpy()
    ])

    # SHAP prediction wrapper
    def model_predict(pair_features):

        tensor = torch.tensor(
            pair_features,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():

            h = F.relu(model.fc1(tensor))
            h = model.dropout(h)

            h = F.relu(model.fc2(h))
            h = model.dropout(h)

            logits = model.fc3(h)

            probs = torch.sigmoid(logits)

        return probs.cpu().numpy()

    # Sanity check
    wrapper_score = float(model_predict(pair_feat)[0])

    print(f"Original model score : {score:.6f}")
    print(f"Wrapper model score  : {wrapper_score:.6f}")

    print("Running SHAP (this takes ~30 seconds)...")

    explainer = shap.KernelExplainer(
        model_predict,
        background
    )

    shap_values = explainer.shap_values(
        pair_feat,
        nsamples=100
    )

    shap_values = np.asarray(
        shap_values
    ).squeeze()

    print("SHAP shape:", shap_values.shape)

    # Handle different SHAP output formats
    if shap_values.ndim > 1:
        shap_values = shap_values.reshape(-1)

    if len(shap_values) != 4096:
        print(
            f"Unexpected SHAP length: {len(shap_values)} "
            f"(expected 4096)"
        )
        return score, shap_values

    shap_drug1 = shap_values[:2048]
    shap_drug2 = shap_values[2048:]

    # Drug 1 features
    top_bits_1 = np.argsort(
        np.abs(shap_drug1)
    )[::-1][:10]

    print("\nTop 10 important fingerprint bits for Drug 1:")

    for rank, bit in enumerate(top_bits_1):

        value = float(shap_drug1[bit])

        direction = (
            "↑ increases"
            if value > 0
            else "↓ decreases"
        )

        print(
            f"  {rank+1}. "
            f"Bit {bit:4d}: "
            f"SHAP={value:.4f} "
            f"({direction} interaction probability)"
        )

    # Drug 2 features
    top_bits_2 = np.argsort(
        np.abs(shap_drug2)
    )[::-1][:10]

    print("\nTop 10 important fingerprint bits for Drug 2:")

    for rank, bit in enumerate(top_bits_2):

        value = float(shap_drug2[bit])

        direction = (
            "↑ increases"
            if value > 0
            else "↓ decreases"
        )

        print(
            f"  {rank+1}. "
            f"Bit {bit:4d}: "
            f"SHAP={value:.4f} "
            f"({direction} interaction probability)"
        )

    return score, shap_values

    # Wrapper for SHAP
    def model_predict(pair_features):
        tensor = torch.tensor(pair_features, dtype=torch.float).to(device)
        src_feat = tensor[:, :2048]
        dst_feat = tensor[:, 2048:]
        combined = torch.cat([src_feat, dst_feat], dim=1)

        # Run through MLP layers directly
        with torch.no_grad():
            h = torch.relu(model.fc1(combined))
            h = model.dropout(h)
            h = torch.relu(model.fc2(h))
            h = model.dropout(h)
            out = torch.sigmoid(model.fc3(h))
        return out.cpu().numpy()

    print("Running SHAP (this takes ~30 seconds)...")
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(pair_feat, nsamples=100)

    # Top features for drug 1 (first 2048 bits)
    shap_drug1 = shap_values[0][:2048]
    shap_drug2 = shap_values[0][2048:]

    for shap_vals, label in [(shap_drug1, 'Drug 1'), (shap_drug2, 'Drug 2')]:
        top_bits = np.argsort(np.abs(shap_vals))[::-1][:10]
        print(f"\nTop 10 important fingerprint bits for {label}:")
        for rank, bit in enumerate(top_bits):
            direction = "↑ increases" if shap_vals[bit] > 0 else "↓ decreases"
            print(f"  {rank+1}. Bit {bit:4d}: SHAP={shap_vals[bit]:.4f} ({direction} interaction probability)")

    return score, shap_values

#  Main

def explain_drug_pair(drug1_stitch, drug2_stitch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    pairs = load_twosides('data/raw/TWOSIDES.csv')
    drug2idx = build_drug_vocab(pairs)

    if drug1_stitch not in drug2idx:
        print(f"Drug {drug1_stitch} not found in graph")
        return
    if drug2_stitch not in drug2idx:
        print(f"Drug {drug2_stitch} not found in graph")
        return

    drug1_idx = drug2idx[drug1_stitch]
    drug2_idx = drug2idx[drug2_stitch]

    print(f"\nExplaining: {drug1_stitch} (node {drug1_idx}) + {drug2_stitch} (node {drug2_idx})")

    # Run both explanations
    gcn_score, gnn_explanation = explain_with_gnn(drug1_idx, drug2_idx, device)
    mlp_score, shap_values = explain_with_shap(drug1_idx, drug2_idx, device)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Drug pair: {drug1_stitch} + {drug2_stitch}")
    print(f"MLP prediction (best model): {mlp_score*100:.1f}% interaction probability")
    print(f"GCN prediction:              {gcn_score*100:.1f}% interaction probability")
    print("\nExplainability:")
    print("  GNNExplainer → which neighboring drugs and graph edges matter")
    print("  SHAP         → which molecular fingerprint bits drive the prediction")

if __name__ == "__main__":
    # Use first two drugs in dataset as example
    pairs = load_twosides('data/raw/TWOSIDES.csv')
    drug2idx = build_drug_vocab(pairs)
    drugs = list(drug2idx.keys())

    drug1 = drugs[0]
    drug2 = drugs[1]

    explain_drug_pair(drug1, drug2)