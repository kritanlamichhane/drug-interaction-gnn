import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import os

# Ensure root project directory and app directory are both in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
APP_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.dataset import load_twosides, build_drug_vocab
from src.model import MLPBaseline, GCNModel, GATModel

try:
    from drug_names import get_drug_name, get_display_label, extract_stitch_id
except ImportError:
    from app.drug_names import get_drug_name, get_display_label, extract_stitch_id

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioSNAP Drug Interaction Engine",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Inject Premium Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Dark Theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #171d31 0%, #0c101c 60%, #05070d 100%);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.6), 0 0 20px rgba(99, 102, 241, 0.1);
    }

    .hero-title {
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem !important;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3af;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.65);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.3);
    }

    .glass-card-title {
        font-size: 1.2rem;
        color: #f8fafc;
        margin-bottom: 14px;
        border-left: 4px solid #6366f1;
        padding-left: 10px;
        font-family: 'Outfit', sans-serif;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin: 18px 0;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.5);
        background: rgba(30, 41, 59, 0.7);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .metric-badge {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 10px;
    }

    .badge-mlp { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-gcn { background: rgba(139, 92, 246, 0.2); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.4); }
    .badge-gat { background: rgba(236, 72, 153, 0.2); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.4); }

    .metric-val {
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
        margin: 8px 0;
        font-family: 'Outfit', sans-serif;
    }

    .metric-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 6px;
    }

    /* Risk Alerts */
    .risk-alert {
        padding: 20px 24px;
        border-radius: 14px;
        margin-top: 20px;
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .risk-high {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }

    .risk-low {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #a7f3d0;
    }

    .risk-icon {
        font-size: 2.2rem;
    }

    .risk-text {
        font-size: 1.02rem;
        line-height: 1.5;
    }

    /* Molecular Container */
    .mol-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }

    .mol-header {
        font-weight: 700;
        font-size: 1.15rem;
        color: #f8fafc;
        margin-bottom: 4px;
    }

    .mol-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 12px;
    }

    .mol-img-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 8px;
        display: inline-block;
        margin: 0 auto;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* Tables */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
    }

    .custom-table th {
        background: rgba(30, 41, 59, 0.8);
        color: #f1f5f9;
        font-weight: 600;
        text-align: left;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }

    .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        color: #cbd5e1;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
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

    # Load SMILES cache if available
    smiles_map = {}
    if os.path.exists('data/processed/smiles_cache.csv'):
        try:
            cache_df = pd.read_csv('data/processed/smiles_cache.csv', index_col=0)
            smiles_map = cache_df['smiles'].dropna().to_dict()
        except Exception:
            pass

    return mlp, gcn, gat, x, edge_index, drug2idx, idx2drug, smiles_map, device

# ── Prediction & Utility Helpers ──────────────────────────────────────
def predict(model, x, edge_index, idx1, idx2, device):
    if model is None:
        return 0.0
    target = torch.tensor([[idx1], [idx2]], dtype=torch.long).to(device)
    with torch.no_grad():
        score = torch.sigmoid(model(x, edge_index, target)).item()
    return score

def get_pubchem_cid(stitch_id):
    try:
        cid = int(stitch_id.replace("CID", "").replace("CIDs", "").replace("CIDm", ""))
        if cid >= 100000000:
            cid -= 100000000
        return cid
    except Exception:
        return None

def get_neighbors(edge_index, node_idx, max_neighbors=20):
    mask = edge_index[0] == node_idx
    neighbors = edge_index[1][mask].cpu().numpy()
    return neighbors[:max_neighbors]

def get_shared_neighbors(edge_index, idx1, idx2):
    mask1 = edge_index[0] == idx1
    neigh1 = set(edge_index[1][mask1].cpu().numpy())
    mask2 = edge_index[0] == idx2
    neigh2 = set(edge_index[1][mask2].cpu().numpy())
    return list(neigh1.intersection(neigh2))

# ── RDKit 2D Molecule Image Generator ─────────────────────────────────
@st.cache_data
def render_molecule_svg(smiles_str):
    if not smiles_str or pd.isna(smiles_str) or str(smiles_str).strip() == "":
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(str(smiles_str))
        if mol is not None:
            # Generate PNG image
            img = Draw.MolToImage(mol, size=(240, 200))
            return img
    except Exception:
        pass
    return None

# ── Explainability: SHAP ──────────────────────────────────────────────
@st.cache_data
def get_shap_values(_mlp, _x, idx1, idx2, _device):
    import shap
    x_cpu = _x.cpu()
    num_nodes = x_cpu.shape[0]
    np.random.seed(42)
    bg_idx1 = np.random.randint(0, num_nodes, 40)
    bg_idx2 = np.random.randint(0, num_nodes, 40)
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
    shap_vals = explainer.shap_values(pair_feat, nsamples=40)
    
    shap_vals = np.asarray(shap_vals).squeeze()
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.reshape(-1)
        
    return shap_vals

# ── Explainability: GNNExplainer ──────────────────────────────────────
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
        algorithm=GNNExplainer(epochs=40),
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

# ── Main Application Header ───────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">💊 BioSNAP Drug Interaction Engine</h1>
    <p class="hero-subtitle">
        Explainable Graph Neural Network platform predicting adverse Drug-Drug Interactions (DDI) with molecular fingerprint attributions & network topology exploration.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Initialize Assets ─────────────────────────────────────────────────
try:
    with st.spinner("Loading models, PyG interaction graph, and chemical fingerprints..."):
        mlp, gcn, gat, x, edge_index, drug2idx, idx2drug, smiles_map, device = load_everything()
    
    # Generate list of display labels with Drug Names
    raw_drugs = sorted(list(drug2idx.keys()))
    display_options = [get_display_label(d) for d in raw_drugs]
    label_to_stitch = {get_display_label(d): d for d in raw_drugs}
except Exception as e:
    st.error(f"Error initializing system assets: {e}")
    st.info("Ensure all requirements and model weights in `models/` and `data/` exist.")
    st.stop()

# ── Sidebar Controls ──────────────────────────────────────────────────
st.sidebar.markdown("###  Engine Control Panel")

# Presets selector for easy demonstration
preset_choice = st.sidebar.selectbox(
    " Quick Load Preset Drug Pair",
    [
        "Select a preset...",
        "Ampicillin + Fentanyl (High Risk)",
        "Aspirin + Warfarin (Severe Bleeding Risk)",
        "Lisinopril + Spironolactone (Hyperkalemia Risk)",
        "Metformin + Cimetidine (Lactic Acidosis Risk)",
        "Gabapentin + Morphine (CNS Depression Risk)"
    ]
)

decision_threshold = st.sidebar.slider(
    "Clinical Risk Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Probability above which a drug combination is flagged as high interaction risk."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
####  **Dataset Metrics**
- **Drugs in Graph**: 645 unique compounds
- **Known Interactions**: 63,473 verified pairs
- **Graph Density**: 30.5%
- **Feature Vector**: 2048-bit Morgan Fingerprint
""")

st.sidebar.markdown("""
####  **Evaluated Models**
- **MLP Baseline**: 3-layer dense on Morgan fingerprints (**0.958 ROC-AUC**)
- **GCN Model**: 2-layer Graph Convolution (**0.933 ROC-AUC**)
- **GAT Model**: 2-layer Multi-Head Attention (**0.922 ROC-AUC**)
""")

# ── Main Tabs Layout ──────────────────────────────────────────────────
tab_pred, tab_explain, tab_explore, tab_benchmarks = st.tabs([
    " Predict Interaction", 
    " Dual Explainability (GNN + SHAP)", 
    " Local Graph Explorer", 
    " Benchmarks & Over-Smoothing Deep Dive"
])

# ── Helper to resolve preset indices ──────────────────────────────────
default_idx1 = 0
default_idx2 = 1

if preset_choice == "Ampicillin + Fentanyl (High Risk)":
    # CID000002173 (Ampicillin) & CID000003345 (Fentanyl)
    d1_target = get_display_label("CID000002173")
    d2_target = get_display_label("CID000003345")
    if d1_target in display_options: default_idx1 = display_options.index(d1_target)
    if d2_target in display_options: default_idx2 = display_options.index(d2_target)

# ======================================================================
# TAB 1: PREDICT INTERACTION
# ======================================================================
with tab_pred:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Select Drug Combination</div>
        <p style="color: #94a3af; font-size: 0.95rem; margin-bottom: 0;">
            Search and select two therapeutic agents by <b>generic name</b> or STITCH ID to compute interaction probabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        sel_label1 = st.selectbox(" Drug 1 (Name / STITCH ID)", display_options, index=default_idx1, key="drug_select_1")
    with col_s2:
        sel_label2 = st.selectbox(" Drug 2 (Name / STITCH ID)", display_options, index=default_idx2, key="drug_select_2")

    drug1 = label_to_stitch[sel_label1]
    drug2 = label_to_stitch[sel_label2]
    name1 = get_drug_name(drug1)
    name2 = get_drug_name(drug2)

    if drug1 == drug2:
        st.warning(" Please select two distinct drug compounds to analyze interaction risk.")
        st.stop()

    idx1 = drug2idx[drug1]
    idx2 = drug2idx[drug2]

    # Display Chemical Structures
    cid1 = get_pubchem_cid(drug1)
    cid2 = get_pubchem_cid(drug2)
    smiles1 = smiles_map.get(drug1, None)
    smiles2 = smiles_map.get(drug2, None)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="mol-box">
            <div class="mol-header">{name1}</div>
            <div class="mol-sub">STITCH ID: <code>{drug1}</code> | PubChem CID: <code>{cid1 if cid1 else 'N/A'}</code></div>
        </div>
        """, unsafe_allow_html=True)
        img1 = render_molecule_svg(smiles1)
        if img1 is not None:
            st.image(img1, caption=f"2D Structure: {name1}", use_container_width=True)
        elif cid1:
            st.markdown(f'<div style="text-align:center;"><img src="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid1}/PNG" width="200" class="mol-img-container"></div>', unsafe_allow_html=True)
        if cid1:
            st.markdown(f'<p style="text-align:center; margin-top:8px;"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid1}" target="_blank" style="color:#818cf8; text-decoration:none;">🔗 View {name1} on PubChem</a></p>', unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="mol-box">
            <div class="mol-header">{name2}</div>
            <div class="mol-sub">STITCH ID: <code>{drug2}</code> | PubChem CID: <code>{cid2 if cid2 else 'N/A'}</code></div>
        </div>
        """, unsafe_allow_html=True)
        img2 = render_molecule_svg(smiles2)
        if img2 is not None:
            st.image(img2, caption=f"2D Structure: {name2}", use_container_width=True)
        elif cid2:
            st.markdown(f'<div style="text-align:center;"><img src="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid2}/PNG" width="200" class="mol-img-container"></div>', unsafe_allow_html=True)
        if cid2:
            st.markdown(f'<p style="text-align:center; margin-top:8px;"><a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid2}" target="_blank" style="color:#818cf8; text-decoration:none;">🔗 View {name2} on PubChem</a></p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(" Run Deep Learning Interaction Analysis", type="primary", use_container_width=True):
        mlp_score = predict(mlp, x, edge_index, idx1, idx2, device)
        gcn_score = predict(gcn, x, edge_index, idx1, idx2, device)
        gat_score = predict(gat, x, edge_index, idx1, idx2, device) if gat is not None else 0.0

        st.session_state['mlp_score'] = mlp_score
        st.session_state['gcn_score'] = gcn_score
        st.session_state['gat_score'] = gat_score
        st.session_state['analyzed_drugs'] = (drug1, drug2)
        st.session_state['analyzed_names'] = (name1, name2)

        st.markdown("###  Model Predictions & Risk Consensus")

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <span class="metric-badge badge-mlp">MLP Baseline (Best Model)</span>
                <div class="metric-val">{mlp_score*100:.1f}%</div>
                <div class="metric-desc">Test ROC-AUC: <b>0.9584</b></div>
                <div class="metric-desc" style="color: {'#ef4444' if mlp_score > decision_threshold else '#10b981'}; font-weight:700;">
                    {' HIGH INTERACTION RISK' if mlp_score > decision_threshold else ' LOW INTERACTION RISK'}
                </div>
            </div>
            <div class="metric-card">
                <span class="metric-badge badge-gcn">GCN Graph Model</span>
                <div class="metric-val">{gcn_score*100:.1f}%</div>
                <div class="metric-desc">Test ROC-AUC: <b>0.9334</b></div>
                <div class="metric-desc" style="color: {'#ef4444' if gcn_score > decision_threshold else '#10b981'}; font-weight:700;">
                    {' HIGH RISK' if gcn_score > decision_threshold else 'LOW RISK'}
                </div>
            </div>
            <div class="metric-card">
                <span class="metric-badge badge-gat">GAT Attention Model</span>
                <div class="metric-val">{gat_score*100:.1f}%</div>
                <div class="metric-desc">Test ROC-AUC: <b>0.9222</b></div>
                <div class="metric-desc" style="color: {'#ef4444' if gat_score > decision_threshold else '#10b981'}; font-weight:700;">
                    {(' HIGH RISK' if gat_score > decision_threshold else 'LOW RISK') if gat is not None else 'N/A'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Consensus diagnosis banner
        if mlp_score > decision_threshold:
            st.markdown(f"""
            <div class="risk-alert risk-high">
                <span class="risk-icon">⚠️</span>
                <div class="risk-text">
                    <strong>Critical Interaction Warning: High interaction probability detected ({mlp_score*100:.1f}%).</strong><br>
                    Co-administration of <b>{name1}</b> and <b>{name2}</b> has a high likelihood of adverse polypharmacy side-effects. 
                    Clinical review and close therapeutic drug monitoring (TDM) are recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-alert risk-low">
                <span class="risk-icon">✅</span>
                <div class="risk-text">
                    <strong>Favorable Safety Profile: Low interaction probability ({mlp_score*100:.1f}%).</strong><br>
                    The consensus models predict that combining <b>{name1}</b> and <b>{name2}</b> carries relatively low risk of adverse interaction.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ======================================================================
# TAB 2: DUAL EXPLAINABILITY (GNNExplainer + SHAP)
# ======================================================================
with tab_explain:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">Dual Explainability Suite</div>
        <p style="color: #9ca3af; font-size: 0.95rem; margin-bottom: 0;">
            Interpret predictions at both the <b>network level</b> (which neighbor drugs influenced the GNN) and the <b>molecular level</b> (which fingerprint sub-structures drove the MLP).
        </p>
    </div>
    """, unsafe_allow_html=True)

    curr_drug1, curr_drug2 = st.session_state.get('analyzed_drugs', (drug1, drug2))
    curr_name1 = get_drug_name(curr_drug1)
    curr_name2 = get_drug_name(curr_drug2)
    c_idx1 = drug2idx[curr_drug1]
    c_idx2 = drug2idx[curr_drug2]

    st.info(f"Target Combination for Attribution: **{curr_name1}** (`{curr_drug1}`) + **{curr_name2}** (`{curr_drug2}`)")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("### 1. Graph Topology Explanations (GNNExplainer)")
        st.markdown("Explains the GCN prediction by identifying the most influential neighboring drugs in the interaction graph.")
        
        run_gnn = st.checkbox(" Compute GNNExplainer Graph Mask", value=False)
        if run_gnn:
            with st.spinner("Calculating subgraph node importance masks..."):
                try:
                    node_importance = get_gnn_explanation(gcn, x, edge_index, c_idx1, c_idx2, device)
                    if node_importance is not None:
                        top_nodes = np.argsort(node_importance)[::-1]
                        top_filtered = [n for n in top_nodes if n not in [c_idx1, c_idx2]][:6]
                        
                        top_drugs = [idx2drug[n] for n in top_filtered]
                        top_names = [get_drug_name(d) for d in top_drugs]
                        top_scores = node_importance[top_filtered]
                        
                        df_gnn = pd.DataFrame({
                            'Influential Context Drug': [f"{name} ({d})" for name, d in zip(top_names, top_drugs)],
                            'Graph Importance Score': top_scores
                        })
                        st.success("GNNExplainer completed!")
                        st.dataframe(df_gnn, use_container_width=True, hide_index=True)
                        st.bar_chart(data=df_gnn, x='Influential Context Drug', y='Graph Importance Score')
                except Exception as e:
                    st.error(f"GNNExplainer computation error: {e}")

    with col_exp2:
        st.markdown("###  2. Molecular Feature Attributions (SHAP)")
        st.markdown("Explains the MLP prediction by identifying which of the 2048 Morgan fingerprint bits contributed most.")
        
        run_shap = st.checkbox(" Compute SHAP Fingerprint Attributions", value=False)
        if run_shap:
            with st.spinner("Calculating SHAP kernel attribution values..."):
                try:
                    shap_vals = get_shap_values(mlp, x, c_idx1, c_idx2, device)
                    shap_d1 = shap_vals[:2048]
                    shap_d2 = shap_vals[2048:]
                    
                    top_bits1 = np.argsort(np.abs(shap_d1))[::-1][:5]
                    top_bits2 = np.argsort(np.abs(shap_d2))[::-1][:5]
                    
                    st.success("SHAP attribution completed!")
                    
                    st.markdown(f"**Top Substructure Bits for {curr_name1}:**")
                    df_s1 = pd.DataFrame({
                        'Morgan Bit': [f"Bit {b}" for b in top_bits1],
                        'SHAP Value': shap_d1[top_bits1],
                        'Effect': ['↑ Increases Risk' if v > 0 else '↓ Decreases Risk' for v in shap_d1[top_bits1]]
                    })
                    st.dataframe(df_s1, use_container_width=True, hide_index=True)

                    st.markdown(f"**Top Substructure Bits for {curr_name2}:**")
                    df_s2 = pd.DataFrame({
                        'Morgan Bit': [f"Bit {b}" for b in top_bits2],
                        'SHAP Value': shap_d2[top_bits2],
                        'Effect': ['↑ Increases Risk' if v > 0 else '↓ Decreases Risk' for v in shap_d2[top_bits2]]
                    })
                    st.dataframe(df_s2, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"SHAP attribution computation error: {e}")

# ======================================================================
# TAB 3: LOCAL GRAPH EXPLORER
# ======================================================================
with tab_explore:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title"> BioSNAP Knowledge Graph Neighborhood</div>
        <p style="color: #9ca3af; font-size: 0.95rem; margin-bottom: 0;">
            Examine known interaction partners and shared polypharmacy neighbors for the selected drug pair.
        </p>
    </div>
    """, unsafe_allow_html=True)

    curr_drug1, curr_drug2 = st.session_state.get('analyzed_drugs', (drug1, drug2))
    curr_name1 = get_drug_name(curr_drug1)
    curr_name2 = get_drug_name(curr_drug2)
    c_idx1 = drug2idx[curr_drug1]
    c_idx2 = drug2idx[curr_drug2]

    # Shared Interactors
    shared_neighbors = get_shared_neighbors(edge_index, c_idx1, c_idx2)
    if shared_neighbors:
        st.markdown(f"#### Shared Polypharmacy Partners ({len(shared_neighbors)} mutual interactors)")
        st.write(f"The following drugs interact with **both** {curr_name1} and {curr_name2}:")
        shared_names = [get_drug_name(idx2drug[n]) for n in shared_neighbors[:12]]
        shared_cids = [idx2drug[n] for n in shared_neighbors[:12]]
        
        shared_df = pd.DataFrame({
            'Shared Drug': shared_names,
            'STITCH ID': shared_cids,
            'PubChem Link': [f"https://pubchem.ncbi.nlm.nih.gov/compound/{get_pubchem_cid(c)}" for c in shared_cids]
        })
        st.dataframe(shared_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No shared mutual interactors directly between {curr_name1} and {curr_name2} in the filtered BioSNAP dataset.")

    st.markdown("---")
    col_n1, col_n2 = st.columns(2)
    
    with col_n1:
        st.markdown(f"#### Known Interactors for **{curr_name1}**")
        n1 = get_neighbors(edge_index, c_idx1, max_neighbors=15)
        n1_drugs = [idx2drug[n] for n in n1]
        n1_names = [get_drug_name(d) for d in n1_drugs]
        df_n1 = pd.DataFrame({
            'Interacting Drug': n1_names,
            'STITCH ID': n1_drugs
        })
        st.dataframe(df_n1, use_container_width=True, hide_index=True)

    with col_n2:
        st.markdown(f"#### Known Interactors for **{curr_name2}**")
        n2 = get_neighbors(edge_index, c_idx2, max_neighbors=15)
        n2_drugs = [idx2drug[n] for n in n2]
        n2_names = [get_drug_name(d) for d in n2_drugs]
        df_n2 = pd.DataFrame({
            'Interacting Drug': n2_names,
            'STITCH ID': n2_drugs
        })
        st.dataframe(df_n2, use_container_width=True, hide_index=True)

# ======================================================================
# TAB 4: BENCHMARKS & OVER-SMOOTHING DEEP DIVE
# ======================================================================
with tab_benchmarks:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title"> Evaluated Benchmark Metrics</div>
        <p style="color: #9ca3af; font-size: 0.95rem; margin-bottom: 0;">
            Comprehensive performance evaluation on held-out test splits (10% test edges) across model architectures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <table class="custom-table">
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>ROC-AUC</th>
                <th>Average Precision (AP)</th>
                <th>Modality</th>
                <th>Key Characteristic</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>MLP Baseline (Best Model)</b></td>
                <td><span style="color:#60a5fa; font-weight:700;">0.9584</span></td>
                <td><span style="color:#60a5fa; font-weight:700;">0.9602</span></td>
                <td>2048-bit Morgan Fingerprints</td>
                <td>Outperforms due to high molecular expressiveness</td>
            </tr>
            <tr>
                <td><b>GCN Model</b></td>
                <td><span style="color:#a78bfa; font-weight:700;">0.9334</span></td>
                <td><span style="color:#a78bfa; font-weight:700;">0.9398</span></td>
                <td>Graph Topology + Fingerprints</td>
                <td>2-layer spectral convolution; prone to over-smoothing</td>
            </tr>
            <tr>
                <td><b>GAT Model</b></td>
                <td><span style="color:#f472b6; font-weight:700;">0.9222</span></td>
                <td><span style="color:#f472b6; font-weight:700;">0.9266</span></td>
                <td>Graph Attention (4 heads)</td>
                <td>Dynamic edge weighting; captures localized subgraphs</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### Why does the MLP Baseline outperform GNNs?
    
    A critical finding in this research is that the **MLP baseline surpasses both GCN and GAT architectures**. This occurs due to two primary biological and computational factors:

    1. **Graph Over-Smoothing on Dense Networks**:
       - The BioSNAP-TWOSIDES graph has **63,473 edges among only 645 drugs**, resulting in a high graph density of ~30.5%.
       - When stacking Graph Convolution or Attention layers, message aggregation averages feature vectors across densely connected neighbors. Over multiple hops, the node representations converge and become excessively similar, washing out distinguishing molecular signatures.

    2. **High Expressiveness of Morgan Chemical Fingerprints**:
       - The 2048-bit circular Morgan Fingerprints (radius=2) directly encode detailed sub-molecular fragments, functional groups, and pharmacophores.
       - The MLP processes the concatenated fingerprint pairs directly without message-passing noise, allowing it to preserve the subtle chemical features responsible for drug reactivity and adverse interactions.
    """)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    BioSNAP Drug Interaction Engine · PyTorch Geometric · RDKit · SHAP · GNNExplainer | BioSNAP-TWOSIDES Dataset
</div>
""", unsafe_allow_html=True)
