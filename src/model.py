import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

# ── 1. MLP Baseline ───────────────────────────────────────────────────
class MLPBaseline(nn.Module):
    def __init__(self, input_dim=2048):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        src, dst = edge_index
        # Concatenate features of both drugs in each pair
        pair_features = torch.cat([x[src], x[dst]], dim=1)
        h = F.relu(self.fc1(pair_features))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h).squeeze()

# ── 2. GCN Model ──────────────────────────────────────────────────────
class GCNModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=64):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def encode(self, x, edge_index):
        # Layer 1 — message passing
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        # Layer 2 — message passing
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edge_index):
        src, dst = edge_index
        # Dot product between drug embeddings
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x, edge_index, target_edges):
        z = self.encode(x, edge_index)
        return self.decode(z, target_edges)

# ── 3. GAT Model (final model) ────────────────────────────────────────
class GATModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=64, heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.3)
        self.dropout = nn.Dropout(p=0.3)

    def encode(self, x, edge_index):
        # Layer 1 — attention-weighted message passing
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        # Layer 2
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x, edge_index, target_edges):
        z = self.encode(x, edge_index)
        return self.decode(z, target_edges)