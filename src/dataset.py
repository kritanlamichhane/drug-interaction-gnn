import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import os
import requests
import time

# Load raw data 
def load_twosides(path):
    df = pd.read_csv(path)
    df.columns = ['drug1', 'drug2', 'side_effect_id', 'side_effect_name']
    # Keep only unique drug pairs — we don't care which side effect for now
    pairs = df[['drug1', 'drug2']].drop_duplicates().reset_index(drop=True)
    print(f"Loaded {len(pairs)} unique drug pairs")
    return pairs

# Build drug vocabulary 
def build_drug_vocab(pairs):
    # Collect all unique drugs and assign each an integer ID
    all_drugs = pd.concat([pairs['drug1'], pairs['drug2']]).unique()
    drug2idx = {drug: idx for idx, drug in enumerate(all_drugs)}
    print(f"Vocabulary: {len(drug2idx)} unique drugs")
    return drug2idx

# Build edge index 
def build_edge_index(pairs, drug2idx):
    # Converting drug ID strings to integer indices
    src = [drug2idx[d] for d in pairs['drug1']]
    dst = [drug2idx[d] for d in pairs['drug2']]
    # Make edges bidirectional (undirected graph)
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    print(f"Edge index shape: {edge_index.shape}")
    return edge_index





def stitch_to_pubchem_cid(stitch_id):
    # STITCH IDs are PubChem CIDs with leading zeros and offset
    # CID1 format: removing 'CID' prefix and converting to int
    cid = int(stitch_id.replace('CID', ''))
    # STITCH adds 100000000 to stereo compounds, subtracting if needed
    if cid > 100000000:
        cid = cid - 100000000
    return cid

def fetch_smiles(pubchem_cid, retries=3):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/IsomericSMILES/JSON"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                props = data['PropertyTable']['Properties'][0]
                smiles = props.get('IsomericSMILES') or props.get('CanonicalSMILES') or props.get('SMILES')
                return smiles
        except Exception:
            time.sleep(1)
    return None  # Return None if fetch fails

def build_node_features(drug2idx, cache_path="data/processed/smiles_cache.csv"):
    # Load cache if exists (avoid re-fetching)
    if os.path.exists(cache_path):
        print("Loading SMILES from cache...")
        cache = pd.read_csv(cache_path, index_col=0)['smiles'].to_dict()
    else:
        cache = {}

    smiles_dict = {}
    drugs = list(drug2idx.keys())

    print(f"Fetching SMILES for {len(drugs)} drugs...")
    for i, stitch_id in enumerate(drugs):
        if stitch_id in cache:
            smiles_dict[stitch_id] = cache[stitch_id]
        else:
            cid = stitch_to_pubchem_cid(stitch_id)
            smiles = fetch_smiles(cid)
            smiles_dict[stitch_id] = smiles
            time.sleep(0.2)  

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(drugs)} done...")
            os.makedirs("data/processed", exist_ok=True)
            pd.DataFrame.from_dict(smiles_dict, orient='index', columns=['smiles']).to_csv(cache_path)

    # Save cache
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame.from_dict(smiles_dict, orient='index', columns=['smiles']).to_csv(cache_path)
    print(f"SMILES fetched. Cache saved to {cache_path}")

    # Compute Morgan fingerprints
    node_features = []
    failed = 0
    for stitch_id in drugs:
        smiles = smiles_dict.get(stitch_id)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                node_features.append(list(fp))
            else:
                node_features.append([0] * 2048)
                failed += 1
        else:
            node_features.append([0] * 2048)
            failed += 1

    print(f"Node features built. Failed: {failed}/{len(drugs)}")
    x = torch.tensor(node_features, dtype=torch.float)
    print(f"Node feature matrix shape: {x.shape}")
    return x

# Main 
if __name__ == "__main__":
    data_path = "data/raw/TWOSIDES.csv"
    pairs = load_twosides(data_path)
    drug2idx = build_drug_vocab(pairs)
    edge_index = build_edge_index(pairs, drug2idx)
    x = build_node_features(drug2idx)
    print(f"\nGraph summary:")
    print(f"  Nodes: {x.shape[0]}, Node feature dim: {x.shape[1]}")
    print(f"  Edges: {edge_index.shape[1]}")