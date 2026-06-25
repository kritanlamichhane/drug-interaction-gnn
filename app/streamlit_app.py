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

    return mlp, gcn, gat, x, edge_index, drug2idx, idx2drug, device

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
    
    # Safely convert to a flat 1D array of 4096 values
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
        algorithm=GNNExplainer(epochs=50), # Fast epochs for web performance
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
    
    # Process node importance sum
    node_importance = None
    if explanation.node_mask is not None:
        node_importance = explanation.node_mask.sum(dim=1).cpu().numpy()
        
    return node_importance

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
        mlp, gcn, gat, x, edge_index, drug2idx, idx2drug, device = load_everything()
    # Create the sorted drug list for selectboxes
    drug_list = sorted(list(drug2idx.keys()))
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.info("Ensure all requirements are installed and the files under models/ and data/ exist.")
    st.stop()

# ── Sidebar Configuration ─────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Engine Control Panel")
selected_device = st.sidebar.selectbox("Device Mode", ["Auto-Detect", "CPU Only", "GPU Only"])
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

    # Show chemical structure if available
    cid1 = get_pubchem_cid(drug1)
    cid2 = get_pubchem_cid(drug2)

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

        # Save scores to session state for explainability tab
        st.session_state['mlp_score'] = mlp_score
        st.session_state['gcn_score'] = gcn_score
        st.session_state['gat_score'] = gat_score
        st.session_state['analyzed_drugs'] = (drug1, drug2)

        st.markdown("### 📊 Predicted Probability Scores")

        # Metric grid
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <span class="metric-badge badge-mlp">MLP Baseline (Aromatic + Morgan)</span>
                <div class="metric-val">{mlp_score*100:.1f}%</div>
                <div class="metric-desc">Best Model ROC-AUC: <b>0.958</b></div>
                <div class="metric-desc {'risk-high' if mlp_score > decision_threshold else 'risk-low'}">
                    {'HIGH RISK' if mlp_score > decision_threshold else 'LOW RISK'}
                </div>
            </div>
            <div class="metric-card">
                <span class="metric-badge badge-gcn">GCN Graph Model</span>
                <div class="metric-val">{gcn_score*100:.1f}%</div>
                <div class="metric-desc">Topology ROC-AUC: <b>0.933</b></div>
                <div class="metric-desc {'risk-high' if gcn_score > decision_threshold else 'risk-low'}">
                    {'HIGH RISK' if gcn_score > decision_threshold else 'LOW RISK'}
                </div>
            </div>
            <div class="metric-card">
                <span class="metric-badge badge-gat">GAT Graph Attention</span>
                <div class="metric-val">{gat_score*100:.1f}%</div>
                <div class="metric-desc">Attention ROC-AUC: <b>0.922</b></div>
                <div class="metric-desc {'risk-high' if gat_score > decision_threshold else 'risk-low' if gat is not None else ''}">
                    {('HIGH RISK' if gat_score > decision_threshold else 'LOW RISK') if gat is not None else 'N/A'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Safety Diagnosis banner
        if mlp_score > decision_threshold:
            st.markdown(f"""
            <div class="custom-alert alert-danger">
                <span class="alert-icon">⚠️</span>
                <div class="alert-text">
                    <strong>Critical Alert: High adverse interaction probability detected ({mlp_score*100:.1f}%).</strong><br>
                    Combining <b>{drug1}</b> and <b>{drug2}</b> could yield high risks of adverse clinical side-effects. 
                    Monitor patient metrics closely if combined administration is required.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-alert alert-success">
                <span class="alert-icon">✅</span>
                <div class="alert-text">
                    <strong>Safe Match: Low adverse interaction probability ({mlp_score*100:.1f}%).</strong><br>
                    The prediction model marks the combination of <b>{drug1}</b> and <b>{drug2}</b> as relatively safe. 
                    Ensure checking other therapeutic counter-indications.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <p style="text-align: center; color: #6b7280; margin-top: 10px; font-size: 0.85rem;">
            Note: Predictions are powered by features compiled from 2048-bit Morgan chemical fingerprint descriptors.
        </p>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: Interpretation & Explanations (Dual)
# ==========================================
with tab_explain:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">🔍 Dual Explainability Dashboard</div>
        <p style="color: #9ca3af; font-size: 0.9rem;">
            We offer two distinct paradigms of explainability:
            <ol>
                <li><b>GNNExplainer on GCN Graph Model</b>: Identifies influential neighboring context drugs in the interaction graph topology.</li>
                <li><b>SHAP on MLP Model</b>: Identifies structural molecular bits (Morgan Fingerprints) driving the prediction.</li>
            </ol>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check if a pair has been analyzed
    has_analyzed = 'analyzed_drugs' in st.session_state
    if has_analyzed:
        active_drug1, active_drug2 = st.session_state['analyzed_drugs']
        st.info(f"Currently explaining: **{active_drug1}** and **{active_drug2}**")
    else:
        active_drug1, active_drug2 = drug1, drug2
        st.warning("Please click 'Analyze Interaction Risk' in the first tab to pre-load specific scores, or use the current selection below.")

    active_idx1 = drug2idx[active_drug1]
    active_idx2 = drug2idx[active_drug2]

    # Create Columns for the two explainers
    col_exp_gnn, col_exp_shap = st.columns(2)

    with col_exp_gnn:
        st.markdown("### 🕸️ 1. Graph Explanations (GNNExplainer)")
        st.markdown("Identifies the neighboring drug nodes that have the highest topological impact on the GCN's prediction.")
        
        run_gnn = st.checkbox("🚀 Run GNNExplainer Analysis (Takes ~10 seconds)")
        if run_gnn:
            with st.spinner("Computing sub-graph node importance using GNNExplainer..."):
                try:
                    node_importance = get_gnn_explanation(gcn, x, edge_index, active_idx1, active_idx2, device)
                    
                    if node_importance is not None:
                        # Find indices of top nodes excluding the target nodes themselves
                        top_nodes = np.argsort(node_importance)[::-1]
                        top_nodes_filtered = [node_idx for node_idx in top_nodes if node_idx not in [active_idx1, active_idx2]][:5]
                        
                        top_node_names = [idx2drug[n] for n in top_nodes_filtered]
                        top_node_scores = node_importance[top_nodes_filtered]
                        
                        df_gnn = pd.DataFrame({
                            'Influential Drug Node': top_node_names,
                            'Importance Score': top_node_scores
                        })
                        
                        st.success("GNNExplainer completed successfully!")
                        st.dataframe(df_gnn, use_container_width=True, hide_index=True)
                        st.bar_chart(data=df_gnn, x='Influential Drug Node', y='Importance Score')
                    else:
                        st.error("No node importance attributes returned by GNNExplainer.")
                except Exception as e:
                    st.error(f"Failed to generate GNNExplainer visualization: {e}")

    with col_exp_shap:
        st.markdown("### 🔬 2. Molecular Explanations (SHAP)")
        st.markdown("Identifies which structural molecular bits of the Morgan Fingerprints drive the MLP predictions.")
        
        run_shap = st.checkbox("🚀 Run SHAP Attribution Analysis (Takes ~15 seconds)")
        if run_shap:
            with st.spinner("Attributing molecular features with SHAP KernelExplainer..."):
                try:
                    shap_vals = get_shap_values(mlp, x, active_idx1, active_idx2, device)
                    
                    shap_d1 = shap_vals[:2048]
                    shap_d2 = shap_vals[2048:]
                    
                    # Fetch top 5 bits for each drug
                    top_bits1 = np.argsort(np.abs(shap_d1))[::-1][:5]
                    top_bits2 = np.argsort(np.abs(shap_d2))[::-1][:5]
                    
                    st.success("SHAP analysis completed successfully!")
                    
                    st.markdown(f"**Top Attributes for {active_drug1}:**")
                    df_shap1 = pd.DataFrame({
                        'Fingerprint Bit ID': [f"Bit {bit}" for bit in top_bits1],
                        'SHAP Importance': shap_d1[top_bits1],
                        'Direction': ['↑ Increases Risk' if v > 0 else '↓ Decreases Risk' for v in shap_d1[top_bits1]]
                    })
                    st.dataframe(df_shap1, use_container_width=True, hide_index=True)
                    
                    st.markdown(f"**Top Attributes for {active_drug2}:**")
                    df_shap2 = pd.DataFrame({
                        'Fingerprint Bit ID': [f"Bit {bit}" for bit in top_bits2],
                        'SHAP Importance': shap_d2[top_bits2],
                        'Direction': ['↑ Increases Risk' if v > 0 else '↓ Decreases Risk' for v in shap_d2[top_bits2]]
                    })
                    st.dataframe(df_shap2, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Failed to generate SHAP attributes: {e}")

# ==========================================
# TAB 3: Local Graph Explorer
# ==========================================
with tab_explore:
    st.markdown("""
    <div class="glass-card">
        <div class="glass-card-title">🕸️ BioSNAP Network Neighbors</div>
        <p style="color: #9ca3af; font-size: 0.9rem;">
            GNN models make predictions by querying neighboring entities in the drug interaction graph. 
            Below are the confirmed interaction neighbors loaded from the TWOSIDES database.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_net1, col_net2 = st.columns(2)
    
    # We display details for the currently active drugs
    active_drug1 = st.session_state.get('analyzed_drugs', (drug1, drug2))[0]
    active_drug2 = st.session_state.get('analyzed_drugs', (drug1, drug2))[1]
    
    idx1 = drug2idx[active_drug1]
    idx2 = drug2idx[active_drug2]

    with col_net1:
        st.markdown(f"#### **{active_drug1} Neighbors**")
        neighbors1 = get_neighbors(edge_index, idx1)
        st.write(f"Connected to **{len(neighbors1)}** visible neighbor nodes in BioSNAP graph:")
        
        neighbor_names1 = [idx2drug[n] for n in neighbors1]
        cids1 = [get_pubchem_cid(name) for name in neighbor_names1]
        
        neigh_df1 = pd.DataFrame({
            'Neighbor ID': neighbor_names1,
            'PubChem Link': [f"https://pubchem.ncbi.nlm.nih.gov/compound/{c}" if c else 'N/A' for c in cids1]
        })
        st.dataframe(neigh_df1, use_container_width=True, hide_index=True)
        
    with col_net2:
        st.markdown(f"#### **{active_drug2} Neighbors**")
        neighbors2 = get_neighbors(edge_index, idx2)
        st.write(f"Connected to **{len(neighbors2)}** visible neighbor nodes in BioSNAP graph:")
        
        neighbor_names2 = [idx2drug[n] for n in neighbors2]
        cids2 = [get_pubchem_cid(name) for name in neighbor_names2]
        
        neigh_df2 = pd.DataFrame({
            'Neighbor ID': neighbor_names2,
            'PubChem Link': [f"https://pubchem.ncbi.nlm.nih.gov/compound/{c}" if c else 'N/A' for c in cids2]
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

    # Benchmark table
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