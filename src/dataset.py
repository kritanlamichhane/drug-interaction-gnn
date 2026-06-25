import os
import time
import requests
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem


def load_twosides(path):
    df = pd.read_csv(path)

    df.columns = [
        "drug1",
        "drug2",
        "side_effect_id",
        "side_effect_name"
    ]

    pairs = (
        df[["drug1", "drug2"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Loaded {len(pairs)} unique drug pairs")

    return pairs



def build_drug_vocab(pairs):
    all_drugs = pd.concat(
        [pairs["drug1"], pairs["drug2"]]
    ).unique()

    drug2idx = {
        drug: idx
        for idx, drug in enumerate(all_drugs)
    }

    print(f"Vocabulary: {len(drug2idx)} unique drugs")

    return drug2idx




def build_edge_index(pairs, drug2idx):

    src = [
        drug2idx[d]
        for d in pairs["drug1"]
    ]

    dst = [
        drug2idx[d]
        for d in pairs["drug2"]
    ]

    edge_index = torch.tensor(
        [src + dst, dst + src],
        dtype=torch.long
    )

    print(f"Edge index shape: {edge_index.shape}")

    return edge_index




def stitch_to_pubchem_cid(stitch_id):
    """
    Examples:
        CID000002173 -> 2173
        CID100002173 -> 2173
    """

    cid = int(stitch_id.replace("CID", ""))

    if cid >= 100000000:
        cid -= 100000000

    return cid



def fetch_smiles_batch(cids, batch_size=10):

    results = {}

    cid_list = list(cids.items())

    for i in range(0, len(cid_list), batch_size):

        batch = cid_list[i:i + batch_size]

        cid_str = ",".join(
            str(cid)
            for _, cid in batch
        )

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            f"compound/cid/{cid_str}/property/"
            "ConnectivitySMILES/JSON"
        )

        try:

            response = requests.get(
                url,
                timeout=30
            )

            if response.status_code != 200:

                print(
                    f"HTTP error: {response.status_code}"
                )

                for stitch_id, _ in batch:
                    results[stitch_id] = None

                continue

            data = response.json()

            props = (
                data.get("PropertyTable", {})
                .get("Properties", [])
            )

            props_map = {
                p["CID"]: p
                for p in props
            }

            for stitch_id, cid in batch:

                if cid not in props_map:
                    results[stitch_id] = None
                    continue

                smiles = props_map[cid].get(
                    "ConnectivitySMILES"
                )

                results[stitch_id] = smiles

        except Exception as e:

            print(
                f"Batch exception: {e}"
            )

            for stitch_id, _ in batch:
                results[stitch_id] = None

        print(
            f"{min(i + batch_size, len(cid_list))}/{len(cid_list)} done..."
        )

        time.sleep(0.2)

    return results




def build_node_features(
    drug2idx,
    cache_path="data/processed/smiles_cache.csv"
):

    if os.path.exists(cache_path):

        print("Loading SMILES cache...")

        cache_df = pd.read_csv(
            cache_path,
            index_col=0
        )

        cache = cache_df["smiles"].to_dict()

    else:
        cache = {}

    drugs = list(drug2idx.keys())

    smiles_dict = {}

    uncached_cids = {}

    for stitch_id in drugs:

        cached_smiles = cache.get(stitch_id)

        if (
            cached_smiles is not None
            and pd.notna(cached_smiles)
            and str(cached_smiles).strip() != ""
        ):

            smiles_dict[stitch_id] = cached_smiles

        else:

            uncached_cids[stitch_id] = (
                stitch_to_pubchem_cid(
                    stitch_id
                )
            )

    print(
        f"Cached: {len(smiles_dict)}"
    )

    print(
        f"Need to fetch: {len(uncached_cids)}"
    )

    if len(uncached_cids) > 0:

        new_smiles = fetch_smiles_batch(
            uncached_cids,
            batch_size=10
        )

        smiles_dict.update(new_smiles)

    os.makedirs(
        os.path.dirname(cache_path),
        exist_ok=True
    )

    pd.DataFrame.from_dict(
        smiles_dict,
        orient="index",
        columns=["smiles"]
    ).to_csv(cache_path)

    print(
        f"Cache saved to {cache_path}"
    )

    node_features = []

    failed = 0

    valid_smiles = 0

    for stitch_id in drugs:

        smiles = smiles_dict.get(
            stitch_id
        )

        if (
            smiles is None
            or pd.isna(smiles)
            or str(smiles).strip() == ""
        ):

            node_features.append(
                [0] * 2048
            )

            failed += 1

            continue

        mol = Chem.MolFromSmiles(
            str(smiles)
        )

        if mol is None:

            node_features.append(
                [0] * 2048
            )

            failed += 1

            continue

        fp = (
            AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=2,
                nBits=2048
            )
        )

        node_features.append(
            list(fp)
        )

        valid_smiles += 1

    print(
        f"Valid SMILES: {valid_smiles}"
    )

    print(
        f"Failed: {failed}/{len(drugs)}"
    )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    print(
        f"Node feature matrix shape: {x.shape}"
    )

    return x




if __name__ == "__main__":

    data_path = "data/raw/TWOSIDES.csv"

    # Delete bad cache if needed
    cache_path = "data/processed/smiles_cache.csv"

    if os.path.exists(cache_path):

        cache_df = pd.read_csv(
            cache_path
        )

        if "smiles" in cache_df.columns:

            non_null = (
                cache_df["smiles"]
                .notna()
                .sum()
            )

            if non_null == 0:

                print(
                    "Removing invalid cache..."
                )

                os.remove(cache_path)

    pairs = load_twosides(data_path)

    drug2idx = build_drug_vocab(
        pairs
    )

    edge_index = build_edge_index(
        pairs,
        drug2idx
    )

    x = build_node_features(
        drug2idx
    )

    print("\nGraph Summary")
    print("-" * 40)
    print(f"Nodes: {x.shape[0]}")
    print(f"Feature Dim: {x.shape[1]}")
    print(f"Edges: {edge_index.shape[1]}")
    
    # Save PyG Data object
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, 'data/processed/ddi_graph.pt')
    print("Graph saved to data/processed/ddi_graph.pt")