# BioSNAP Drug Interaction Engine (GNN + Explainability)

> Predicting unknown drug-drug interactions (DDIs) using Graph Neural Networks and molecular fingerprints on biomedical knowledge graphs — with dual molecular & network explainability via GNNExplainer and SHAP.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange?style=flat-square)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8.0-purple?style=flat-square)
![RDKit](https://img.shields.io/badge/RDKit-2024.03+-green?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

##  Key Highlights

-  **Interactive Streamlit Web Dashboard**: Search drugs by **Generic Drug Names** (e.g. *Ampicillin*, *Lisinopril*, *Fentanyl*, *Aspirin*, *Warfarin*) with local 2D RDKit chemical rendering.
-  **Multi-Model Consensus**: Compares predictions from an MLP baseline (**0.958 ROC-AUC**), Graph Convolutional Network (GCN, **0.933 ROC-AUC**), and Graph Attention Network (GAT, **0.922 ROC-AUC**).
-  **Dual Explainability Suite**:
  - **Network Topology (GNNExplainer)**: Pinpoints the most influential neighboring context drugs in the knowledge graph.
  - **Molecular Fingerprint Attributions (SHAP)**: Pinpoints the 2048-bit Morgan circular fingerprint sub-structures driving interaction risk.
-  **Knowledge Graph Neighborhood Explorer**: Inspects shared mutual interactors and local connectivity across the 63,473 BioSNAP interaction network.

---

##  Benchmark Results

Evaluated on held-out test splits (10% test edges) with uniform negative sampling:

| Model Architecture | Test ROC-AUC | Test Avg Precision (AP) | Graph Modality | Primary Mechanism |
|---|---|---|---|---|
| **MLP Baseline (Best)** | **0.9584** | **0.9602** | No (Direct Molecular) | 2048-bit Morgan Fingerprint concatenation |
| **GCN Model** | **0.9334** | **0.9398** | Yes (Topology + Features) | 2-layer spectral graph convolution |
| **GAT Model** | **0.9222** | **0.9266** | Yes (Attention + Features) | 2-layer graph attention with 4 heads |

### a Finding: Why does MLP outperform GNNs?
On this biomedical dataset, the **MLP baseline outperforms GNNs** due to **Graph Over-Smoothing**:
- The BioSNAP-TWOSIDES graph is dense (**63,473 edges among 645 drugs**, ~30.5% graph density).
- Repeated message aggregation across densely connected vertices causes node embeddings to become overly similar, diluting distinct chemical signatures.
- Direct evaluation of Morgan fingerprints avoids this smoothing effect and preserves fine-grained chemical pharmacophores.

---

##  Architecture

```
Drug SMILES  (PubChem PUG REST API / Local Cache)
    │
    ▼
Morgan Fingerprints (RDKit, radius=2, 2048-bit bit vectors)
    │
    ▼
PyG Knowledge Graph  ──  nodes: 645 drugs, edges: 63,473 known DDIs
    │
    ├── MLP Baseline  ──  Concat pair fingerprints → 3-layer MLP
    ├── GCN           ──  2-layer GCNConv + inner-product decoder
    └── GAT           ──  2-layer GATConv (4 heads) + inner-product decoder
    │
    ▼
Dual Explainability
    ├── GNNExplainer  ──  Topological subgraph & influential drug neighbors (GCN)
    └── SHAP          ──  Attributed fingerprint sub-structural bits (MLP)
    │
    ▼
Streamlit Web App  ──  Search by drug name + 2D structures + consensus + explorer
```

---

##  Project Structure

```
drug-interaction-gnn/
├── data/
│   ├── raw/               # BioSNAP-TWOSIDES dataset (TWOSIDES.csv)
│   └── processed/         # PyG Data object (ddi_graph.pt) & smiles_cache.csv
├── src/
│   ├── dataset.py         # Graph construction, PubChem SMILES fetcher, Morgan featurizer
│   ├── model.py           # MLP Baseline, GCN, and GAT neural network architectures
│   ├── train.py           # Training pipelines & evaluation loops
│   └── explain.py         # GNNExplainer & SHAP standalone explainers
├── app/
│   ├── streamlit_app.py   # Full interactive Streamlit Web Application
│   └── drug_names.py      # Drug name resolver & STITCH ID mapping dictionary
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_training.ipynb  # Interactive model training walkthrough
│   └── 03_explain.ipynb   # Visual explainability demonstrations
├── models/                # Checkpoints (best_mlp.pt, best_gcn.pt, best_gat.pt)
├── requirements.txt
└── README.md
```

---

##  Quickstart & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/kritanlamichhane/drug-interaction-gnn.git
cd drug-interaction-gnn

# Create and activate conda environment
conda create -n ddi python=3.10 -y
conda activate ddi

# Install PyTorch with CUDA support (e.g., CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install torch-geometric

# Install RDKit
conda install -c conda-forge rdkit -y

# Install remaining Python packages
pip install -r requirements.txt
```

### 2. Prepare Graph & Train Models
```bash
# Build PyG graph and fetch/cache SMILES
python src/dataset.py

# Train models
python -m src.train --model mlp --epochs 50
python -m src.train --model gcn --epochs 100
python -m src.train --model gat --epochs 100

# Run terminal explainability checks
python -m src.explain
```

### 3. Launch the Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```

---

##  Streamlit App Features

1. **Drug Selection by Generic Name**: Search for compounds like `Ampicillin`, `Fentanyl`, `Lisinopril`, `Aspirin`, `Warfarin`, or search by STITCH ID.
2. **2D Chemical Structure Visualizer**: Instant local RDKit 2D structure generation with direct link to PubChem compound records.
3. **Consensus Prediction Card**: Side-by-side risk scorecards for MLP, GCN, and GAT with visual risk meters and clinical recommendations.
4. **Dual Explainability Suite**:
   - GNNExplainer: Node importance chart of the most influential surrounding drug nodes.
   - SHAP: Top positive/negative Morgan fingerprint bits driving the MLP score.
5. **Knowledge Graph Neighborhood Explorer**: View known interaction partners and mutual shared interactors in the BioSNAP network.

---

##  Acknowledgements & References

- [BioSNAP-TWOSIDES](https://snap.stanford.edu/biodata/datasets/10017/10017-ChChSe-Decagon.html) — Stanford Network Analysis Project
- [PubChem REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) — National Center for Biotechnology Information (NCBI)
- [PyTorch Geometric (PyG)](https://pyg.org/) — Fey & Lenssen, 2019
- [TWOSIDES Database](https://nsides.io/) — Tatonetti Lab, Columbia University

---

##  License

MIT License — Free for academic, scientific, and educational research purposes.