import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


# =====================================================
# MLP Baseline
# =====================================================
class MLPBaseline(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()

        self.fc1 = nn.Linear(input_dim * 2, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, graph_edge_index, target_edges):
        """
        graph_edge_index is ignored.
        Added only so all models share the same interface.
        """

        src, dst = target_edges

        pair_features = torch.cat(
            [x[src], x[dst]],
            dim=1
        )

        h = F.relu(self.fc1(pair_features))
        h = self.dropout(h)

        h = F.relu(self.fc2(h))
        h = self.dropout(h)

        return self.fc3(h).squeeze(-1)


# =====================================================
# GCN Model
# =====================================================
class GCNModel(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 hidden_dim=256,
                 output_dim=64):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.3)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)

        return h

    def decode(self, z, edge_pairs):
        src, dst = edge_pairs
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x, edge_index, target_edges):
        z = self.encode(x, edge_index)
        return self.decode(z, target_edges)


# =====================================================
# GAT Model
# =====================================================
class GATModel(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 hidden_dim=256,
                 output_dim=64,
                 heads=4):
        super().__init__()

        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=0.3
        )

        self.conv2 = GATConv(
            hidden_dim * heads,
            output_dim,
            heads=1,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.3)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)

        return h

    def decode(self, z, edge_pairs):
        src, dst = edge_pairs
        return (z[src] * z[dst]).sum(dim=1)

    def forward(self, x, edge_index, target_edges):
        z = self.encode(x, edge_index)
        return self.decode(z, target_edges)