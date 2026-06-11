# drug-interaction-gnn

> Predicting unknown drug-drug interactions using Graph Attention Networks on biomedical knowledge graphs — with molecular-level explainability via GNNExplainer.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?style=flat-square)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8.0-purple?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

Drug-drug interactions (DDIs) are a leading cause of adverse drug events — yet the interaction space between thousands of approved drugs is largely uncharted. This project frames DDI prediction as a **link prediction problem on a biomedical knowledge graph**, where:

- **Nodes** = drugs (featurized with Morgan molecular fingerprints via RDKit)
- **Edges** = known polypharmacy interactions (from BioSNAP-TWOSIDES)
- **Task** = predict whether an unknown drug pair will interact

A Graph Attention Network (GAT) is trained to learn relational drug representations, and GNNExplainer surfaces which molecular substructures drive each prediction.

---

## Results

| Model | ROC-AUC | Average Precision |
|---|---|---|
| MLP (baseline) | TBD | TBD |
| GCN | TBD | TBD |
| **GAT (ours)** | **TBD** | **TBD** |

> Results on held-out edge test split. Negative samples drawn via random sampling. Will be updated after training.

---

## Architecture

```
Drug SMILES  (fetched from PubChem REST API)
    │
    ▼
Morgan Fingerprints (RDKit, radius=2, 2048-bit)
    │
    ▼
PyG Graph  ──  nodes: 645 drugs, edges: 63,473 known DDIs
    │
    ▼
Graph Attention Network (2-layer GAT, 4 attention heads)
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
| [BioSNAP-TWOSIDES](https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html) | Polypharmacy side effects for drug pairs, filtered to interactions with strong statistical evidence (PRR score, min 500 drug pairs) | 645 drugs, 63,473 pairs, 1,317 side effects |
| [PubChem REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) | SMILES strings for molecular fingerprint computation | Free, no registration |

**Why BioSNAP over raw TWOSIDES?** The raw TWOSIDES dataset contains 4.6M rows with weak signals. BioSNAP is a cleaned version filtered by Stanford researchers, keeping only interactions with strong statistical evidence. This is the version used in most published DDI prediction benchmarks, making our results directly comparable to state-of-the-art papers.

**No manual downloads required** — running `src/dataset.py` automatically fetches SMILES strings from PubChem and caches them locally.

---

## Project Structure

```
drug-interaction-gnn/
├── data/
│   ├── raw/               # BioSNAP-TWOSIDES download
│   └── processed/         # PyG Data objects + SMILES cache
├── src/
│   ├── dataset.py         # Graph construction + feature engineering
│   ├── model.py           # MLP baseline, GCN, GAT model definitions
│   ├── train.py           # Training loop + evaluation
│   └── explain.py         # GNNExplainer wrapper
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_training.ipynb  # Model training walkthrough
│   └── 03_explain.ipynb   # Explainability demos
├── app/
│   └── streamlit_app.py   # Interactive demo
├── models/                # Saved model checkpoints
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/kritanlamichhane/drug-interaction-gnn.git
cd drug-interaction-gnn

# Create environment
conda create -n ddi python=3.10 -y
conda activate ddi

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# Download BioSNAP-TWOSIDES dataset
# Place ChChSe-Decagon_polypharmacy.csv in data/raw/TWOSIDES.csv

# Build graph + compute node features (fetches SMILES from PubChem automatically)
python src/dataset.py

# Train
python src/train.py --model gat --epochs 100

# Run Streamlit demo
streamlit run app/streamlit_app.py
```

---

## Installation Notes

Tested on:
- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- PyTorch Geometric 2.8.0
- RDKit (conda-forge)
- Windows 11, NVIDIA RTX 3050 6GB

RDKit **must** be installed via conda, not pip — it has C++ bindings that conda handles cleanly.

---

## How It Works

**1. Data pipeline** — BioSNAP-TWOSIDES provides 63,473 unique drug pairs with confirmed polypharmacy side effects. Each unique pair becomes one edge in the graph regardless of how many side effects it causes (binary interaction task).

**2. Feature engineering** — SMILES strings are fetched from PubChem's REST API for each of the 645 drugs and cached locally. RDKit computes 2048-bit Morgan fingerprints (radius=2) capturing local molecular substructure around each atom.

**3. Graph construction** — drugs are nodes; confirmed interactions are positive edges. Negative edges are randomly sampled from unobserved pairs. The graph is split at the edge level (not node level) into 80/10/10 train/val/test sets.

**4. Model progression** — three models are trained in order: MLP baseline (no graph), GCN (uniform neighbor aggregation), GAT (attention-weighted aggregation). The progression shows how graph structure and attention each contribute to performance.

**5. Explainability** — GNNExplainer identifies the most influential node features and neighboring drugs for each predicted interaction, mapped back to molecular substructures via RDKit.

---

## Limitations

- Negative sampling assumes unobserved pairs are non-interacting, which may introduce noise since many pairs are simply unstudied
- Morgan fingerprints don't capture 3D molecular geometry
- Model is transductive — retraining required for entirely new drugs not in the graph
- Binary task only — does not predict which specific side effect will occur

---

## Future Work

- Multi-label classification to predict specific side effect types (1,317 classes)
- Add protein target embeddings as additional node features
- Heterogeneous graphs with drugs, proteins, and diseases as different node types
- Inductive setting — generalize to entirely unseen drugs using molecular features alone

---

## License

MIT — free to use for academic and research purposes.

---

## Acknowledgements

- [BioSNAP-TWOSIDES](https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html) — Stanford Network Analysis Project
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) — National Library of Medicine
- [PyTorch Geometric](https://pyg.org/) — Fey & Lenssen, 2019
- Original TWOSIDES — Tatonetti Lab, Columbia University