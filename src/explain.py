import torch
from torch_geometric.explain import Explainer, GNNExplainer
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os

def load_model_and_data(model_name='gat'):
    from src.model import GATModel, GCNModel, MLPBaseline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load graph
    data = torch.load('data/processed/ddi_graph.pt')
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Load model
    if model_name == 'gat':
        model = GATModel(input_dim=2048).to(device)
    elif model_name == 'gcn':
        model = GCNModel(input_dim=2048).to(device)
    else:
        model = MLPBaseline(input_dim=2048).to(device)

    model.load_state_dict(torch.load(f'models/best_{model_name}.pt', map_location=device))
    model.eval()
    print(f"Loaded {model_name.upper()} model and graph")
    return model, x, edge_index, device

def explain_prediction(drug1_idx, drug2_idx, model, x, edge_index, device):
    target_edge = torch.tensor([[drug1_idx], [drug2_idx]], dtype=torch.long).to(device)

    # Get raw prediction score
    with torch.no_grad():
        score = torch.sigmoid(model(x, edge_index, target_edge)).item()

    print(f"\nPrediction score: {score:.4f} ({score*100:.1f}% interaction probability)")

    # Run GNNExplainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='edge',
            return_type='raw'
        )
    )

    explanation = explainer(
        x=x,
        edge_index=edge_index,
        target=target_edge
    )

    return score, explanation

def get_top_features(explanation, drug_idx, top_k=10):
    # Get feature importance for a specific drug node
    node_mask = explanation.node_mask  # [num_nodes, 2048]
    drug_importance = node_mask[drug_idx].cpu().numpy()

    # Get top-k most important feature indices
    top_indices = np.argsort(drug_importance)[::-1][:top_k]
    top_scores = drug_importance[top_indices]

    print(f"\nTop {top_k} important fingerprint bits for drug {drug_idx}:")
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores)):
        print(f"  {rank+1}. Bit {idx}: importance {score:.4f}")

    return top_indices, top_scores

def explain_drug_pair(drug1_name, drug2_name, drug2idx):
    # Look up drug indices
    idx2drug = {v: k for k, v in drug2idx.items()}

    if drug1_name not in drug2idx or drug2_name not in drug2idx:
        print(f"Drug not found in graph. Available drugs: {list(drug2idx.keys())[:5]}...")
        return

    drug1_idx = drug2idx[drug1_name]
    drug2_idx = drug2idx[drug2_name]

    model, x, edge_index, device = load_model_and_data('gat')
    score, explanation = explain_prediction(drug1_idx, drug2_idx, model, x, edge_index, device)

    print(f"\n--- Explanation for {drug1_name} + {drug2_name} ---")
    get_top_features(explanation, drug1_idx, top_k=10)
    get_top_features(explanation, drug2_idx, top_k=10)

    return score, explanation

if __name__ == "__main__":
    # Load drug vocabulary
    import sys
    sys.path.append('.')
    from src.dataset import load_twosides, build_drug_vocab

    pairs = load_twosides('data/raw/TWOSIDES.csv')
    drug2idx = build_drug_vocab(pairs)

    # Example — explain first drug pair in dataset
    sample_drug1 = list(drug2idx.keys())[0]
    sample_drug2 = list(drug2idx.keys())[1]
    print(f"Explaining interaction between {sample_drug1} and {sample_drug2}")

    explain_drug_pair(sample_drug1, sample_drug2, drug2idx)