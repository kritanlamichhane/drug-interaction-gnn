import torch
import torch_geometric
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import sklearn
import networkx as nx

# Test PyG Data object
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(f"PyG graph: {data.num_nodes} nodes, {data.num_edges} edges")

# Test RDKit fingerprint
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
print(f"Morgan fingerprint: {fp.GetNumBits()} bits")

# Test GPU
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n✓ All dependencies working")