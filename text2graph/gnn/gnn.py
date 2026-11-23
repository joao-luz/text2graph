import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, HANConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            conv = GCNConv(in_c, out_c)
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels if i == 0 else hidden_channels
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            lin = nn.Linear(in_c, out_c)
            self.lins.append(lin)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.relu(x)
        return x


class HeteroGNN(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            out_c = out_channels if i == num_layers - 1 else hidden_channels
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), out_c)
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict


class HAN(nn.Module):
    """Heterogeneous GNN using HANConv (node-level + semantic-level attention).

    Every node type is first linearly projected into hidden_channels, so that
    node types with no incoming edges at some (or all) layers -- which pass
    through HANConv unchanged -- stay dimensionally consistent with node
    types that do get updated, instead of drifting back to their raw input
    width.
    """
    def __init__(self, metadata, hidden_channels, out_channels, num_layers=2, heads=8, dropout=0.0):
        super().__init__()

        if hidden_channels % heads != 0:
            raise ValueError(
                f'hidden_channels ({hidden_channels}) must be divisible by heads ({heads}) -- '
                f'HANConv splits its output into `heads` chunks internally.'
            )

        node_types = metadata[0]

        self.proj_dict = nn.ModuleDict({
            node_type: nn.LazyLinear(hidden_channels)
            for node_type in node_types
        })

        self.convs = nn.ModuleList([
            HANConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=metadata, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.lin_dict = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels, out_channels)
            for node_type in node_types
        })

    def forward(self, x_dict, edge_index_dict, return_semantic_attention_weights=False):
        x_dict = {node_type: F.relu(self.proj_dict[node_type](x)) for node_type, x in x_dict.items()}

        attention_weights = [] if return_semantic_attention_weights else None

        for conv in self.convs:
            if return_semantic_attention_weights:
                out_dict, attn_dict = conv(x_dict, edge_index_dict, return_semantic_attention_weights=True)
                attention_weights.append(attn_dict)
            else:
                out_dict = conv(x_dict, edge_index_dict)

            x_dict = {
                node_type: (F.relu(out_dict[node_type]) if out_dict.get(node_type) is not None else x_dict[node_type])
                for node_type in x_dict
            }

        out_dict = {node_type: self.lin_dict[node_type](x) for node_type, x in x_dict.items()}

        if return_semantic_attention_weights:
            return out_dict, attention_weights
        return out_dict