# drug-interaction-gnn

> Predicting unknown drug-drug interactions using Graph Attention Networks on biomedical knowledge graphs — with molecular-level explainability via GNNExplainer.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Drug-drug interactions (DDIs) are a leading cause of adverse drug events — yet the interaction space between ~10,000 approved drugs is largely uncharted. This project frames DDI prediction as a **link prediction problem on a biomedical knowledge graph**, where:

- **Nodes** = drugs (featurized with Morgan molecular fingerprints)
- **Edges** = known interactions (from TWOSIDES / DrugBank)
- **Task** = predict whether an unknown drug pair will interact

A Graph Attention Network (GAT) is trained to learn relational drug representations, and GNNExplainer surfaces which molecular substructures drive each prediction.

---

## Results

| Model | ROC-AUC | Average Precision |
|---|---|---|
| MLP (baseline) | ~0.78 | ~0.74 |
| GCN | ~0.85 | ~0.81 |
| **GAT (ours)** | **~0.91** | **~0.88** |

> Results on held-out edge test split. Negative samples drawn via random sampling.

---

## Architecture

```
Drug SMILES
    │
    ▼
Morgan Fingerprints (RDKit, radius=2, 2048-bit)
    │
    ▼
PyG Graph  ──  nodes: drugs, edges: known DDIs
    │
    ▼
Graph Attention Network (2-layer GAT)
    │
    ▼
Link Prediction Head  ──  score(u,v) = sigmoid(h_u · h_v)
    │
    ▼
GNNExplainer  ──  per-prediction feature importance
```

---

## Dataset

| Source | Description | Size |
|---|---|---|
| [TWOSIDES](http://tatonettilab.org/offsides/) | Drug pairs + side effect types | 63k pairs, 10k side effects |
| [DrugBank](https://go.drugbank.com/) | Drug properties, targets, known DDIs | ~14k drugs (academic license) |

DrugBank requires a free academic registration. TWOSIDES is publicly available.

---

## Project Structure

```
drug-interaction-gnn/
├── data/
│   ├── raw/               # Raw DrugBank / TWOSIDES downloads
│   └── processed/         # PyG Data objects (after preprocessing)
├── src/
│   ├── dataset.py         # Graph construction + feature engineering
│   ├── model.py           # GCN, GAT model definitions
│   ├── train.py           # Training loop + evaluation
│   └── explain.py         # GNNExplainer wrapper
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_training.ipynb  # Model training walkthrough
│   └── 03_explain.ipynb   # Explainability demos
├── app/
│   └── streamlit_app.py   # Interactive demo
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/<your-username>/drug-interaction-gnn.git
cd drug-interaction-gnn

# Install dependencies
pip install -r requirements.txt

# Preprocess data (assumes raw data in data/raw/)
python src/dataset.py

# Train
python src/train.py --model gat --epochs 100

# Run Streamlit demo
streamlit run app/streamlit_app.py
```

---

## Installation

```bash
# Core dependencies
pip install torch torch-geometric rdkit-pypi scikit-learn

# Explainability
pip install torch-geometric  # GNNExplainer is built in
```

Tested on Python 3.10, PyTorch 2.1, PyG 2.4.

---

## How It Works

**1. Feature engineering** — each drug is encoded as a 2048-bit Morgan fingerprint (circular fingerprint capturing local molecular neighborhoods). Additional features include molecular weight, LogP, H-bond donors/acceptors.

**2. Graph construction** — drugs are nodes; a confirmed interaction is a positive edge. Negative edges are randomly sampled from non-interacting pairs. The graph is split at the edge level (not node level) into train/val/test sets.

**3. GAT training** — two-layer Graph Attention Network performs message passing, learning to weight neighboring drugs differently. The link prediction score between drug *u* and *v* is the dot product of their learned embeddings.

**4. Explainability** — GNNExplainer identifies the most influential node features and neighboring drugs for each predicted interaction. Results are mapped back to molecular substructures using RDKit.

---

## Demo

Run locally with:

```bash
streamlit run app/streamlit_app.py
```

Input any two drug names → get an interaction probability score, the influential molecular features, and a local subgraph visualization.

---

## Limitations

- Negative sampling assumes unobserved pairs are non-interacting, which may introduce noise
- Morgan fingerprints don't capture 3D molecular geometry
- Model is transductive — retraining required for entirely new drugs not in the graph

---

## Future Work

- Add protein target embeddings as additional node features
- Explore heterogeneous graphs (drugs + proteins + diseases as different node types)
- Incorporate 3D molecular conformations via graph networks on atomic structure

---

## License

MIT — free to use for academic and research purposes. DrugBank data requires its own academic license.

---

## Acknowledgements

- [TWOSIDES dataset](http://tatonettilab.org/offsides/) — Tatonetti Lab, Columbia University
- [DrugBank](https://go.drugbank.com/) — Wishart Research Group
- [PyTorch Geometric](https://pyg.org/) — Fey & Lenssen, 2019