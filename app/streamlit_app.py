import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import os

# Append current directory to path to allow importing from src
sys.path.append('.')

from src.dataset import load_twosides, build_drug_vocab
from src.model import MLPBaseline, GCNModel, GATModel

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioSNAP Drug Interaction Engine",
    page_icon="💊",
    layout="wide"
)

# ── Inject Premium Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Custom fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main body background & styling */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #030712 100%);
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Banners and Titles */
    .hero-container {
        background: radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.1) 90%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 30px 40px;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5);
    }

    .hero-title {
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #9ca3af;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Cards & Containers */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    }

    .glass-card-title {
        font-size: 1.25rem;
        color: #e5e7eb;
        margin-bottom: 16px;
        border-left: 4px solid #8b5cf6;
        padding-left: 10px;
        font-family: 'Outfit', sans-serif;
    }

    /* High-fidelity Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: rgba(31, 41, 55, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(31, 41, 55, 0.6);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }

    .metric-badge {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 8px;
    }

    .badge-mlp { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-gcn { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-gat { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.3); }

    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        margin: 6px 0;
        font-family: 'Outfit', sans-serif;
    }

    .metric-desc {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    .risk-high {
        color: #f87171 !important;
        font-weight: 600;
    }

    .risk-low {
        color: #34d399 !important;
        font-weight: 600;
    }

    /* Custom Safety Warning Banners */
    .custom-alert {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
    }

    .alert-danger {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }

    .alert-success {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #a7f3d0;
    }

    .alert-icon {
        font-size: 2rem;
    }

    .alert-text {
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Molecular info panels */
    .mol-panel {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        text-align: center;
    }

    .mol-image {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        margin: 12px auto;
        display: block;
    }

    /* SVG Molecular Highlights Container */
    .svg-container {
        background-color: #ffffff;
        border: 3px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px;
        display: inline-block;
        margin: 10px auto;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }

    /* Tables */
    .premium-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    
    .premium-table th {
        background: rgba(31, 41, 55, 0.7);
        color: #e5e7eb;
        font-weight: 600;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.08);
    }

    .premium-table td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
    }

    .premium-table tr:hover {
        background: rgba(255, 255, 255, 0.02);
    }

    /* Custom footer styles */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ── Load data and models (cached) ─────────────────────────────────────
@st.cache_resource
def load_everything():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load graph
    data = torch.load('data/processed/ddi_graph.pt', weights_only=False)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Load MLP model
    mlp = MLPBaseline(input_dim=2048).to(device)
    mlp.load_state_dict(torch.load('models/best_mlp.pt', map_location=device, weights_only=False))
    mlp.eval()

    # Load GCN model
    gcn = GCNModel(input_dim=2048).to(device)
    gcn.load_state_dict(torch.load('models/best_gcn.pt', map_location=device, weights_only=False))
    gcn.eval()

    # Load GAT model (if available)
    gat = None
    if os.path.exists('models/best_gat.pt'):
        gat = GATModel(input_dim=2048).to(device)
        gat.load_state_dict(torch.load('models/best_gat.pt', map_location=device, weights_only=False))
        gat.eval()

    # Load drug vocabulary
    pairs = load_twosides('data/raw/TWOSIDES.csv')
    drug2idx = build_drug_vocab(pairs)
    idx2drug = {v: k for k, v in drug2idx.items()}

    # Load SMILES cache
    smiles_dict = {}
    if os.path.exists('data/processed/smiles_cache.csv'):
        try:
            smiles_df = pd.read_csv('data/processed/smiles_cache.csv', index_col=0)
            smiles_df.index = smiles_df.index.map(str)
            smiles_dict = smiles_df['smiles'].dropna().to_dict()
        except Exception as e:
            print(f"Error loading smiles cache: {e}")

    return mlp, gcn, gat, x, edge_index, drug2idx, idx2drug, smiles_dict, device

# ── Helpers ───────────────────────────────────────────────────────────
def predict(model, x, edge_index, idx1, idx2, device):
    if model is None:
        return 0.0
    target = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x, edge_index, target)).item()
    return score

def get_pubchem_cid(stitch_id):
    try:
        cid = int(stitch_id.replace("CID", "").replace("CIDs", ""))
        if cid >= 100000000:
            cid -= 100000000
        return cid
    except Exception:
        return None

def get_neighbors(edge_index, node_idx, max_neighbors=15):
    mask = edge_index[0] == node_idx
    neighbors = edge_index[1][mask].cpu().numpy()
    return neighbors[:max_neighbors]

# ── SHAP explanation ──────────────────────────────────────────────────
@st.cache_data
def get_shap_values(_mlp, _x, idx1, idx2, _device):
    import shap
    x_cpu = _x.cpu()
    num_nodes = x_cpu.shape[0]
    np.random.seed(42)
    bg_idx1 = np.random.randint(0, num_nodes, 50)
    bg_idx2 = np.random.randint(0, num_nodes, 50)
    background = np.hstack([x_cpu[bg_idx1].numpy(), x_cpu[bg_idx2].numpy()])
    pair_feat = np.hstack([x_cpu[idx1].numpy(), x_cpu[idx2].numpy()]).reshape(1, -1)

    def model_predict(pair_features):
        tensor = torch.tensor(pair_features, dtype=torch.float).to(_device)
        with torch.no_grad():
            h = torch.relu(_mlp.fc1(tensor))
            h = _mlp.dropout(h)
            h = torch.relu(_mlp.fc2(h))
            h = _mlp.dropout(h)
            out = torch.sigmoid(_mlp.fc3(h))
        return out.cpu().numpy()

    explainer = shap.KernelExplainer(model_predict, background)
    shap_vals = explainer.shap_values(pair_feat, nsamples=50)
    
    shap_vals = np.asarray(shap_vals).squeeze()
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.reshape(-1)
        
    return shap_vals

# ── GNNExplainer explanation ──────────────────────────────────────────
@st.cache_data
def get_gnn_explanation(_gcn, _x, _edge_index, idx1, idx2, _device):
    from torch_geometric.explain import Explainer, GNNExplainer
    
    class GCNWrapper(torch.nn.Module):
        def __init__(self, model, target_edge):
            super().__init__()
            self.model = model
            self.target_edge = target_edge

        def forward(self, x, edge_index):
            return self.model(x, edge_index, self.target_edge)

    target_edge = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(_device)
    wrapped = GCNWrapper(_gcn, target_edge)

    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='raw'
        )
    )

    explanation = explainer(x=_x, edge_index=_edge_index)
    
    node_importance = None
    if explanation.node_mask is not None:
        node_importance = explanation.node_mask.sum(dim=1).cpu().numpy()
        
    return node_importance

# ── RDKit Molecule Drawing with Substructure Highlights ───────────────
def draw_molecule_with_highlights(smiles, top_bits, highlight_mode="Combined Top Features", width=340, height=280):
    if not smiles or pd.isna(smiles):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
        
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
            
        info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info)
        
        atoms_to_highlight = []
        bonds_to_highlight = []
        atom_colors = {}
        bond_colors = {}
        
        for idx, (bit, val) in enumerate(top_bits):
            if highlight_mode == "Risk-Increasing Only" and val <= 0:
                continue
            if highlight_mode == "Risk-Decreasing Only" and val >= 0:
                continue
            if "Bit " in highlight_mode and f"Bit {bit}" not in highlight_mode:
                continue
            if bit not in info:
                continue
                
            color = (0.97, 0.44, 0.44) if val > 0 else (0.2, 0.83, 0.6)
            
            for atom_idx, radius in info[bit]:
                if radius > 0:
                    try:
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                        for bond_idx in env:
                            bond = mol.GetBondWithIdx(bond_idx)
                            a1 = bond.GetBeginAtomIdx()
                            a2 = bond.GetEndAtomIdx()
                            atoms_to_highlight.extend([a1, a2])
                            bonds_to_highlight.append(bond_idx)
                            atom_colors[a1] = color
                            atom_colors[a2] = color
                            bond_colors[bond_idx] = color
                    except Exception:
                        atoms_to_highlight.append(atom_idx)
                        atom_colors[atom_idx] = color
                else:
                    atoms_to_highlight.append(atom_idx)
                    atom_colors[atom_idx] = color
                    
        atoms_to_highlight = list(set(atoms_to_highlight))
        bonds_to_highlight = list(set(bonds_to_highlight))
        
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.backgroundColour = (1.0, 1.0, 1.0, 1.0)
        
        Chem.AllChem.Compute2DCoords(mol)
        drawer.DrawMolecule(
            mol, 
            highlightAtoms=atoms_to_highlight, 
            highlightAtomColors=atom_colors,
            highlightBonds=bonds_to_highlight,
            highlightBondColors=bond_colors
        )
        drawer.FinishDrawing()
        
        return drawer.GetDrawingText()
    except Exception as e:
        print(f"RDKit drawing failed: {e}")
        return None

# ── Pyvis Interactive Local Network ───────────────────────────────────
def build_interactive_network(edge_index, idx1, idx2, drug1_name, drug2_name, idx2drug, node_importance=None):
    from pyvis.network import Network
    import tempfile
    
    mask1 = edge_index[0] == idx1
    neigh1 = set(edge_index[1][mask1].cpu().numpy())
    
    mask2 = edge_index[0] == idx2
    neigh2 = set(edge_index[1][mask2].cpu().numpy())
    
    shared = list(neigh1.intersection(neigh2))
    unique1 = list(neigh1.difference(neigh2))
    unique2 = list(neigh2.difference(neigh1))
    
    shared_to_show = shared[:10]
    unique1_to_show = unique1[:8]
    unique2_to_show = unique2[:8]
    
    nodes_to_include = {idx1, idx2}
    nodes_to_include.update(shared_to_show)
    nodes_to_include.update(unique1_to_show)
    nodes_to_include.update(unique2_to_show)
    
    net = Network(height="450px", width="100%", bgcolor="#0f172a", font_color="#f1f5f9")
    
    # Configure beautiful, overlap-free network physics and layout via vis.js options
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 11,
          "face": "Inter, sans-serif",
          "color": "#cbd5e1"
        }
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.1,
          "springLength": 160,
          "springStrength": 0.04,
          "damping": 0.5,
          "avoidOverlap": 1
        },
        "maxVelocity": 50,
        "minVelocity": 0.75,
        "solver": "barnesHut",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      }
    }
    """)
    
    for n in nodes_to_include:
        name = idx2drug[n]
        size = 15
        
        if n == idx1:
            color = "#3b82f6"
            size = 32
            group = "Target Drug 1"
            title = f"Drug 1: {name} (Selected Target)"
        elif n == idx2:
            color = "#ec4899"
            size = 32
            group = "Target Drug 2"
            title = f"Drug 2: {name} (Selected Target)"
        elif n in shared_to_show:
            color = "#a855f7"
            size = 22
            group = "Shared Neighbor"
            title = f"Shared Neighbor: {name} (Interacts with both target drugs)"
        elif n in unique1_to_show:
            color = "#60a5fa"
            size = 18
            group = "Drug 1 Neighbor"
            title = f"Neighbor of {drug1_name}: {name}"
        else:
            color = "#f472b6"
            size = 18
            group = "Drug 2 Neighbor"
            title = f"Neighbor of {drug2_name}: {name}"
            
        if node_importance is not None and len(node_importance) > n:
            importance = float(node_importance[n])
            if n not in [idx1, idx2]:
                size = max(10, int(15 + importance * 30))
                title += f" | Graph Importance: {importance:.4f}"
                
        net.add_node(
            int(n), 
            label=name, 
            title=title, 
            color=color, 
            size=size,
            shape="dot"
        )
        
    edge_pairs = edge_index.t().cpu().numpy()
    added_edges = set()
    
    for u, v in edge_pairs:
        if u in nodes_to_include and v in nodes_to_include:
            edge_key = tuple(sorted((int(u), int(v))))
            if edge_key not in added_edges:
                added_edges.add(edge_key)
                
                if (u == idx1 and v == idx2) or (u == idx2 and v == idx1):
                    net.add_edge(int(u), int(v), color="#f59e0b", width=5, title="Known BioSNAP Polypharmacy Interaction Link")
                else:
                    net.add_edge(int(u), int(v), color="rgba(255, 255, 255, 0.12)", width=1.5)
                    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        tmp.seek(0)
        html_content = tmp.read().decode('utf-8')
    try:
        os.unlink(tmp.name)
    except Exception:
        pass
        
    return html_content

# ── Main Header Banner ────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">💊 BioSNAP Drug Interaction Engine</h1>
    <p class="hero-subtitle">
        Explainable Deep Learning platform predicting adverse Drug-Drug Interactions (DDI) via Graph Neural Networks and molecular fingerprints.
    </p>
</div>
""", unsafe_allow_html=True)

# Load data and models
try:
    with st.spinner("Initializing models and loading graph weights..."):
        mlp, gcn, gat, x, edge_index, drug2idx, idx2drug, smiles_dict, device = load_everything()
    drug_list = sorted(list(drug2idx.keys()))
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.info("Ensure all requirements are installed and the files under models/ and data/ exist.")
    st.stop()

# ── Sidebar Configuration ─────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Engine Control Panel")
selected_device = st.sidebar.selectbox("Device Mode", [f"Auto-Detect ({device.type.upper()})"])
decision_threshold = st.sidebar.slider("Decision Threshold (Risk Level)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
#### **Dataset Stats**
- **Unique Drugs**: 645
- **Known Interactions**: 63,473
- **Graph Density**: 30.5%
""")

st.sidebar.markdown("""
#### **Model Types**
- **MLP Baseline**: Morgan molecular fingerprints input (2048-dim each) concatenated.
- **GCN**: Graph Convolutional Network capturing topological neighborhoods.
- **GAT**: Graph Attention Network weighing interactions dynamically.
""")

# ── Main Tabs Layout ──────────────────────────────────────────────────
tab_pred, tab_explain, tab_explore, tab_benchmarks = st.tabs([
    "🔮 Predict Interaction", 
    "🔍 Interpretation & Explanations", 
    "🕸️ Local Graph Explorer", 
    "📊 Benchmarks & Reference"
])

# ==========================================
# TAB 1: Predict Interaction
# ==========================================
with tab_pred:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Select Drug Combination</div>
        <p style="color: #9ca3af; font-size: 0.9rem;">
            Choose two therapeutic agents by their STITCH identifiers to compute the probability of a co-administration adverse side-effect.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_select1, col_select2 = st.columns(2)
    with col_select1:
        drug1 = st.selectbox("Drug 1 (STITCH ID)", drug_list, index=0, key="pred_drug_1")
    with col_select2:
        drug2 = st.selectbox("Drug 2 (STITCH ID)", drug_list, index=1, key="pred_drug_2")

    if drug1 == drug2:
        st.warning("⚠️ Warning: Select two distinct drug compounds to check for interactions.")
        st.stop()

    idx1 = drug2idx[drug1]
    idx2 = drug2idx[drug2]

    cid1 = get_pubchem_cid(drug1)
    cid2 = get_pubchem_cid(drug2)
    
    smiles1 = smiles_dict.get(drug1)
    smiles2 = smiles_dict.get(drug2)

    col_mol1, col_mol2 = st.columns(2)
    with col_mol1:
        st.markdown(f"""
        <div class="mol-panel">
            <strong>{drug1} Structural Signature</strong><br>
            <span style="font-size:0.85rem; color:#9ca3af;">PubChem CID: {cid1 if cid1 else 'N/A'}</span>
        </div>
        """, unsafe_allow_html=True)
        if cid1:
            st.markdown(f'<img src="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid1}/PNG" class="mol-image" width="180">', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid1}" target="_blank">🔗 PubChem Summary</a></p>', unsafe_allow_html=True)
        else:
            st.caption("No Pubchem Structure visual found")

    with col_mol2:
        st.markdown(f"""
        <div class="mol-panel">
            <strong>{drug2} Structural Signature</strong><br>
            <span style="font-size:0.85rem; color:#9ca3af;">PubChem CID: {cid2 if cid2 else 'N/A'}</span>
        </div>
        """, unsafe_allow_html=True)
        if cid2:
            st.markdown(f'<img src="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid2}/PNG" class="mol-image" width="180">', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid2}" target="_blank">🔗 PubChem Summary</a></p>', unsafe_allow_html=True)
        else:
            st.caption("No Pubchem Structure visual found")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔥 Analyze Interaction Risk", type="primary", use_container_width=True):
        mlp_score = predict(mlp, x, edge_index, idx1, idx2, device)
        gcn_score = predict(gcn, x, edge_index, idx1, idx2, device)
        gat_score = predict(gat, x, edge_index, idx1, idx2, device) if gat is not None else 0.0

        st.session_state['mlp_score'] = mlp_score
        st.session_state['gcn_score'] = gcn_score
        st.session_state['gat_score'] = gat_score
        st.session_state['analyzed_drugs'] = (drug1, drug2)
        
        if 'node_importance' in st.session_state:
            del st.session_state['node_importance']
        if 'shap_vals' in st.session_state:
            del st.session_state['shap_vals']

        st.markdown("### 📊 Predicted Probability Scores")

        if gat is not None:
            weights = np.array([0.9584, 0.9334, 0.9222])
            scores = np.array([mlp_score, gcn_score, gat_score])
        else:
            weights = np.array([0.9584, 0.9334])
            scores = np.array([mlp_score, gcn_score])
            
        consensus_score = np.sum(scores * weights) / np.sum(weights)
        
        if consensus_score > decision_threshold:
            consensus_color = "#f87171"
            consensus_glow = "rgba(239, 68, 68, 0.4)"
            risk_label = "🚨 CRITICAL RISK: Co-administration is strongly counter-indicated by AI model consensus."
        elif consensus_score > (decision_threshold - 0.15):
            consensus_color = "#fbbf24"
            consensus_glow = "rgba(245, 158, 11, 0.4)"
            risk_label = "⚠️ MODERATE RISK: Potential interactions detected. Clinical oversight advised."
        else:
            consensus_color = "#34d399"
            consensus_glow = "rgba(16, 185, 129, 0.4)"
            risk_label = "✅ LOW RISK: The co-administration is marked as safe based on consensus analysis."

        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.65); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 25px; text-align: center; margin-bottom: 25px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);">
            <div style="font-size: 0.9rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 8px;">Consensus Interaction Risk</div>
            <div style="font-size: 3.2rem; font-weight: 800; color: {consensus_color}; font-family: 'Outfit', sans-serif; text-shadow: 0 0 20px {consensus_glow}; line-height: 1.1;">{consensus_score*100:.1f}%</div>
            <div style="width: 100%; background-color: rgba(255,255,255,0.07); border-radius: 10px; height: 10px; margin: 18px 0; overflow: hidden; border: 1px solid rgba(255,255,255,0.03);">
                <div style="background: linear-gradient(90deg, #34d399 0%, #fbbf24 60%, #f87171 100%); width: {consensus_score*100}%; height: 100%; border-radius: 10px; box-shadow: 0 0 10px {consensus_glow};"></div>
            </div>
            <div style="font-size: 1rem; color: #f3f4f6; font-weight: 600; font-family: 'Outfit', sans-serif;">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <span class="metric-badge badge-mlp">MLP Baseline (Chemical Features)</span>
                <div class="metric-val">{mlp_score*100:.1f}%</div>
                <div class="metric-desc">Feature Accuracy ROC-AUC: <b>0.958</b></div>
                <div class="metric-desc {'risk-high' if mlp_score > decision_threshold else 'risk-low'}">
                    {'HIGH RISK' if mlp_score > decision_threshold else 'LOW RISK'}
                </div>
            </div>
            <div class="metric-card">
                <span class="metric-badge badge-gcn">GCN Graph Model</span>
                <div class="metric-val">{gcn_score*100:.1f}%</div>
                <div class="metric-desc">Topology Accuracy ROC-AUC: <b>0.933</b></div>
                <div class="metric-desc {'risk-high' if gcn_score > decision_threshold else 'risk-low'}">
                    {'HIGH RISK' if gcn_score > decision_threshold else 'LOW RISK'}
                </div>
            </div>
            <div class="metric-card">
                <span class="metric-badge badge-gat">GAT Graph Attention</span>
                <div class="metric-val">{gat_score*100:.1f}%</div>
                <div class="metric-desc">Attention Accuracy ROC-AUC: <b>0.922</b></div>
                <div class="metric-desc {'risk-high' if gat_score > decision_threshold else 'risk-low' if gat is not None else ''}">
                    {('HIGH RISK' if gat_score > decision_threshold else 'LOW RISK') if gat is not None else 'N/A'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📚 Clinical Disclaimer & How to Interpret"):
            st.markdown("""
            - **Disclaimer**: This tool is a proof-of-concept AI prediction platform based on high-confidence drug interaction mappings from Stanford's BioSNAP dataset. It should not be used as a clinical consultation system.
            - **Score Meanings**: The score predicts the likelihood that the two compounds cause a significant polypharmacy adverse event.
            - **Models Contrast**: 
              - The **MLP** represents structural chemical alignment.
              - The **GNN models (GCN/GAT)** represent how topological pathways in the biomedical graph route interactions.
            """)

# ==========================================
# TAB 2: Interpretation & Explanations
# ==========================================
with tab_explain:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">🔍 Dual Explainability Dashboard</div>
        <p style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 0;">
            This panel explains predictions at two distinct scales: **graph topological influence** (network context) and **molecular fingerprint decomposition** (substructure importance).
        </p>
    </div>
    """, unsafe_allow_html=True)

    has_analyzed = 'analyzed_drugs' in st.session_state
    if has_analyzed:
        active_drug1, active_drug2 = st.session_state['analyzed_drugs']
        st.info(f"Showing explanations for: **{active_drug1}** and **{active_drug2}**")
    else:
        active_drug1, active_drug2 = drug1, drug2
        st.warning("⚠️ Run 'Analyze Interaction Risk' in the first tab to analyze risk, or proceed with current defaults below.")

    active_idx1 = drug2idx[active_drug1]
    active_idx2 = drug2idx[active_drug2]
    
    active_smiles1 = smiles_dict.get(active_drug1)
    active_smiles2 = smiles_dict.get(active_drug2)

    col_exp_gnn, col_exp_shap = st.columns(2)

    with col_exp_gnn:
        st.markdown("### 🕸️ 1. Graph Explanations (GNNExplainer)")
        st.markdown("This identifies the neighboring drug nodes that have the highest topological impact on the GCN model's prediction.")
        
        run_gnn = st.checkbox("🚀 Run GNNExplainer analysis", key="run_gnn_check")
        if run_gnn:
            if 'node_importance' not in st.session_state:
                with st.spinner("Computing sub-graph node importance weights via GNNExplainer..."):
                    try:
                        node_importance = get_gnn_explanation(gcn, x, edge_index, active_idx1, active_idx2, device)
                        st.session_state['node_importance'] = node_importance
                    except Exception as e:
                        st.error(f"GNNExplainer failed: {e}")
                        node_importance = None
            else:
                node_importance = st.session_state['node_importance']

            if node_importance is not None:
                top_nodes = np.argsort(node_importance)[::-1]
                top_nodes_filtered = [node_idx for node_idx in top_nodes if node_idx not in [active_idx1, active_idx2]][:5]
                
                top_node_names = [idx2drug[n] for n in top_nodes_filtered]
                top_node_scores = node_importance[top_nodes_filtered]
                
                df_gnn = pd.DataFrame({
                    'Influential Drug Node': top_node_names,
                    'GNN Explainer Importance': top_node_scores
                })
                
                st.success("GNNExplainer analysis completed.")
                st.bar_chart(data=df_gnn, x='Influential Drug Node', y='GNN Explainer Importance', color='#a78bfa')
                st.dataframe(df_gnn, use_container_width=True, hide_index=True)
            else:
                st.error("No node importance scores returned.")

    with col_exp_shap:
        st.markdown("### 🔬 2. Molecular Explanations (SHAP)")
        st.markdown("This identifies which structural molecular bits of the 2048-bit Morgan Fingerprints drive the prediction.")
        
        run_shap = st.checkbox("🚀 Run SHAP Attribution analysis", key="run_shap_check")
        if run_shap:
            if 'shap_vals' not in st.session_state:
                with st.spinner("Computing molecular feature SHAP values (takes ~15s)..."):
                    try:
                        shap_vals = get_shap_values(mlp, x, active_idx1, active_idx2, device)
                        st.session_state['shap_vals'] = shap_vals
                    except Exception as e:
                        st.error(f"SHAP failed: {e}")
                        shap_vals = None
            else:
                shap_vals = st.session_state['shap_vals']

            if shap_vals is not None:
                shap_d1 = shap_vals[:2048]
                shap_d2 = shap_vals[2048:]
                
                top_bits_idx1 = np.argsort(np.abs(shap_d1))[::-1][:5]
                top_bits1 = [(bit, shap_d1[bit]) for bit in top_bits_idx1]
                
                top_bits_idx2 = np.argsort(np.abs(shap_d2))[::-1][:5]
                top_bits2 = [(bit, shap_d2[bit]) for bit in top_bits_idx2]
                
                st.success("SHAP analysis completed.")
                
                col_tab1, col_tab2 = st.columns(2)
                with col_tab1:
                    st.markdown(f"**Top Bits for {active_drug1}**")
                    df_shap1 = pd.DataFrame({
                        'Bit ID': [f"Bit {b}" for b, _ in top_bits1],
                        'SHAP': [v for _, v in top_bits1],
                        'Risk': ['↑ Increase' if v > 0 else '↓ Decrease' for _, v in top_bits1]
                    })
                    st.dataframe(df_shap1, use_container_width=True, hide_index=True)
                with col_tab2:
                    st.markdown(f"**Top Bits for {active_drug2}**")
                    df_shap2 = pd.DataFrame({
                        'Bit ID': [f"Bit {b}" for b, _ in top_bits2],
                        'SHAP': [v for _, v in top_bits2],
                        'Risk': ['↑ Increase' if v > 0 else '↓ Decrease' for _, v in top_bits2]
                    })
                    st.dataframe(df_shap2, use_container_width=True, hide_index=True)

    if run_shap and 'shap_vals' in st.session_state and st.session_state['shap_vals'] is not None:
        st.markdown("---")
        st.markdown("### 🎨 Interactive Molecular Structure Highlights (SHAP Mapped)")
        st.markdown("""
        Select how to visualize the molecular features on the chemical structure drawings. 
        <span style="color:#f87171;font-weight:bold;">Red highlights</span> represent structures increasing interaction risk; 
        <span style="color:#34d399;font-weight:bold;">Green highlights</span> represent structures decreasing risk.
        """, unsafe_allow_html=True)
        
        mode_opts = ["Combined Top Features", "Risk-Increasing Only", "Risk-Decreasing Only"]
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            bit_opts_1 = mode_opts + [f"Bit {b} ({'Increases' if v > 0 else 'Decreases'} Risk)" for b, v in top_bits1]
            highlight_mode_1 = st.selectbox(f"Highlight Mode for {active_drug1}", bit_opts_1, key="hm_1")
        with col_ctrl2:
            bit_opts_2 = mode_opts + [f"Bit {b} ({'Increases' if v > 0 else 'Decreases'} Risk)" for b, v in top_bits2]
            highlight_mode_2 = st.selectbox(f"Highlight Mode for {active_drug2}", bit_opts_2, key="hm_2")
            
        col_draw1, col_draw2 = st.columns(2)
        
        with col_draw1:
            st.markdown(f"<div style='text-align:center;'><strong>{active_drug1} Substructures</strong></div>", unsafe_allow_html=True)
            if active_smiles1:
                svg1 = draw_molecule_with_highlights(active_smiles1, top_bits1, highlight_mode_1)
                if svg1:
                    st.markdown(f'<div class="svg-container">{svg1}</div>', unsafe_allow_html=True)
                else:
                    st.caption("RDKit rendering unavailable for this compound.")
            else:
                st.caption("SMILES representation missing for this compound.")
                
        with col_draw2:
            st.markdown(f"<div style='text-align:center;'><strong>{active_drug2} Substructures</strong></div>", unsafe_allow_html=True)
            if active_smiles2:
                svg2 = draw_molecule_with_highlights(active_smiles2, top_bits2, highlight_mode_2)
                if svg2:
                    st.markdown(f'<div class="svg-container">{svg2}</div>', unsafe_allow_html=True)
                else:
                    st.caption("RDKit rendering unavailable for this compound.")
            else:
                st.caption("SMILES representation missing for this compound.")

# ==========================================
# TAB 3: Local Graph Explorer
# ==========================================
with tab_explore:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">🕸️ BioSNAP Network Neighbors</div>
        <p style="color: #9ca3af; font-size: 0.9rem;">
            This panel renders the localized biomedical knowledge graph neighborhood around the selected drug pair. 
            Drag nodes, zoom, or hover on links to explore connection pathways.
        </p>
    </div>
    """, unsafe_allow_html=True)

    active_drug1 = st.session_state.get('analyzed_drugs', (drug1, drug2))[0]
    active_drug2 = st.session_state.get('analyzed_drugs', (drug1, drug2))[1]
    
    idx1 = drug2idx[active_drug1]
    idx2 = drug2idx[active_drug2]

    node_importance = st.session_state.get('node_importance', None)
    
    with st.spinner("Generating interactive subgraph visualization..."):
        network_html = build_interactive_network(
            edge_index, 
            idx1, 
            idx2, 
            active_drug1, 
            active_drug2, 
            idx2drug,
            node_importance
        )
        
    st.components.v1.html(network_html, height=480, scrolling=False)
    
    st.markdown("""
    <div style="display:flex; justify-content:center; gap:20px; font-size:0.85rem; margin-top:5px; margin-bottom:20px;">
        <div><span style="color:#3b82f6; font-size:1.2rem;">●</span> Target Drug 1</div>
        <div><span style="color:#ec4899; font-size:1.2rem;">●</span> Target Drug 2</div>
        <div><span style="color:#a855f7; font-size:1.2rem;">●</span> Shared Neighbor Node</div>
        <div><span style="color:#60a5fa; font-size:1.2rem;">●</span> Drug 1 Neighbor</div>
        <div><span style="color:#f472b6; font-size:1.2rem;">●</span> Drug 2 Neighbor</div>
        <div><span style="color:#f59e0b; font-size:1.2rem;">▬</span> Known DDI Link</div>
    </div>
    """, unsafe_allow_html=True)

    col_net1, col_net2 = st.columns(2)
    with col_net1:
        st.markdown(f"#### **{active_drug1} Neighbors**")
        neighbors1 = get_neighbors(edge_index, idx1)
        neighbor_names1 = [idx2drug[n] for n in neighbors1]
        cids1 = [get_pubchem_cid(name) for name in neighbor_names1]
        neigh_df1 = pd.DataFrame({
            'Neighbor ID': neighbor_names1,
            'PubChem CID': [str(c) if c else 'N/A' for c in cids1]
        })
        st.dataframe(neigh_df1, use_container_width=True, hide_index=True)
        
    with col_net2:
        st.markdown(f"#### **{active_drug2} Neighbors**")
        neighbors2 = get_neighbors(edge_index, idx2)
        neighbor_names2 = [idx2drug[n] for n in neighbors2]
        cids2 = [get_pubchem_cid(name) for name in neighbor_names2]
        neigh_df2 = pd.DataFrame({
            'Neighbor ID': neighbor_names2,
            'PubChem CID': [str(c) if c else 'N/A' for c in cids2]
        })
        st.dataframe(neigh_df2, use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: Benchmarks & Reference
# ==========================================
with tab_benchmarks:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">📊 Evaluated Benchmark Scores</div>
        <p style="color: #9ca3af; font-size: 0.9rem;">
            Comparing the predictive accuracy across MLP Baseline and Graph Neural Network (GNN) paradigms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="premium-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>ROC-AUC</th>
                <th>Avg Precision</th>
                <th>Modality</th>
                <th>Verdict</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>MLP Baseline (Best)</b></td>
                <td><span style="color:#60a5fa">0.9584</span></td>
                <td>0.9602</td>
                <td>Morgan fingerprints (2048-d)</td>
                <td>🏆 Outperforms due to molecular feature density</td>
            </tr>
            <tr>
                <td><b>GCN Model</b></td>
                <td><span style="color:#a78bfa">0.9334</span></td>
                <td>0.9398</td>
                <td>Topology + Fingerprints</td>
                <td>Good representation but prone to over-smoothing</td>
            </tr>
            <tr>
                <td><b>GAT Model</b></td>
                <td><span style="color:#f472b6">0.9222</span></td>
                <td>0.9266</td>
                <td>Attention + Fingerprints</td>
                <td>Captures localized dynamic attention</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### Why does MLP beat Graph Neural Networks?
    In this dense interaction dataset, we experience **GNN over-smoothing**. 
    Because the BioSNAP TWOSIDES graph contains dense interactions (63,473 edges among just 645 drugs), 
    when stacking multiple message-passing layer steps, node embedding vectors tend to converge and become highly similar.
    Consequently, local structural signatures are diluted.
    
    **MLP baseline** sidesteps this over-smoothing since it performs predictions directly on the high-dimensional
    molecular structure descriptions (Morgan Fingerprints vectors) rather than relying on message aggregation across 
    highly connected graph vertices.
    """)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with PyTorch Geometric · RDKit · SHAP | BioSNAP-TWOSIDES dataset (645 drugs, 63,473 interactions)
</div>
""", unsafe_allow_html=True)