# drug-interaction-gnn

> Predicting unknown drug-drug interactions using Graph Neural Networks on biomedical knowledge graphs — with molecular-level explainability via GNNExplainer and SHAP.

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

Three models are trained and compared — MLP baseline, GCN, and GAT. Explainability is provided via GNNExplainer (graph-level) and SHAP (molecular feature-level). An interactive Streamlit demo allows real-time prediction and explanation for any drug pair.

---

## Results

| Model | ROC-AUC | Avg Precision | Uses Graph |
|---|---|---|---|
| **MLP (best)** | **0.9584** | **0.9602** | No |
| GCN | 0.9334 | 0.9398 | Yes |
| GAT | 0.9222 | 0.9266 | Yes |

> Results on held-out edge test split (10%). Negative samples drawn via random sampling.

**Key finding:** MLP outperforms GNNs because Morgan fingerprints are highly expressive on this dense interaction graph. Graph aggregation causes over-smoothing — blurring the molecular signal rather than sharpening it. This is consistent with known over-smoothing literature and is itself an interesting result.

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
    ├── MLP Baseline  ──  concatenate pair fingerprints → 3-layer MLP
    ├── GCN           ──  2-layer graph convolution + dot product decoder  
    └── GAT           ──  2-layer graph attention (4 heads) + dot product decoder
    │
    ▼
Explainability
    ├── GNNExplainer  ──  influential neighbors + graph edges (GCN)
    └── SHAP          ──  fingerprint bit importance (MLP)
    │
    ▼
Streamlit Demo  ──  real-time prediction + explanation for any drug pair
```

---

## Dataset

| Source | Description | Size |
|---|---|---|
| [BioSNAP-TWOSIDES](https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html) | Polypharmacy side effects for drug pairs, filtered to high-confidence interactions | 645 drugs, 63,473 pairs, 1,317 side effects |
| [PubChem REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) | SMILES strings for molecular fingerprint computation | Free, no registration |

**Why BioSNAP over raw TWOSIDES?** Raw TWOSIDES has 4.6M rows with weak signals. BioSNAP is filtered by Stanford researchers to high-confidence interactions, used in most published DDI benchmarks — making results directly comparable to state-of-the-art papers.

**No manual downloads required** — `src/dataset.py` automatically fetches SMILES from PubChem and caches them locally.

---

## Project Structure

```
drug-interaction-gnn/
├── data/
│   ├── raw/               # BioSNAP-TWOSIDES download
│   └── processed/         # PyG Data object + SMILES cache (git-ignored)
├── src/
│   ├── dataset.py         # Graph construction + Morgan fingerprint features
│   ├── model.py           # MLP baseline, GCN, GAT definitions
│   ├── train.py           # Training loop + ROC-AUC evaluation
│   └── explain.py         # GNNExplainer (GCN) + SHAP (MLP)
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_training.ipynb  # Model training walkthrough
│   └── 03_explain.ipynb   # Explainability demos
├── app/
│   └── streamlit_app.py   # Interactive demo
├── models/                # Saved checkpoints (git-ignored)
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

# Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyG
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric

# Install RDKit (must use conda)
conda install -c conda-forge rdkit -y

# Install remaining dependencies
pip install -r requirements.txt

# Download BioSNAP-TWOSIDES
# Place ChChSe-Decagon_polypharmacy.csv in data/raw/TWOSIDES.csv

# Build graph + compute node features (fetches SMILES from PubChem)
python src/dataset.py

# Train all three models
python -m src.train --model mlp --epochs 50
python -m src.train --model gcn --epochs 100
python -m src.train --model gat --epochs 100

# Run explainability
python -m src.explain

# Launch Streamlit demo
streamlit run app/streamlit_app.py
```

---

## Explainability

Two complementary approaches are used:

**GNNExplainer (on GCN)** — identifies which neighboring drugs and graph edges are most influential for a specific prediction. Answers: *"which part of the drug network drove this?"*

**SHAP (on MLP)** — identifies which of the 2048 Morgan fingerprint bits most influenced the prediction. Answers: *"which molecular substructures of these two drugs drove this?"*

Together they provide both network-level and molecular-level explanations for every prediction.

---

## Installation Notes

Tested on:
- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- PyTorch Geometric 2.8.0
- RDKit (conda-forge)
- Windows 11, NVIDIA GeForce RTX 3050 6GB Laptop GPU

RDKit **must** be installed via conda, not pip.

---

## Limitations

- Negative sampling assumes unobserved pairs are non-interacting — may introduce noise since many pairs are simply unstudied
- Morgan fingerprints don't capture 3D molecular geometry
- Model is transductive — retraining required for entirely new drugs not in the graph
- Binary task only — does not predict which specific side effect will occur

---

## Future Work

- Multi-label classification to predict specific side effect types (1,317 classes)
- Add protein target embeddings as additional node features
- Heterogeneous graphs with drugs, proteins, and diseases as different node types
- Inductive setting — generalize to unseen drugs using molecular features alone
- Fix over-smoothing via residual connections or PairNorm

---


## License

MIT — free to use for academic and research purposes.

---

## Acknowledgements

- [BioSNAP-TWOSIDES](https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html) — Stanford Network Analysis Project
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) — National Library of Medicine
- [PyTorch Geometric](https://pyg.org/) — Fey & Lenssen, 2019
- Original TWOSIDES — Tatonetti Lab, Columbia University