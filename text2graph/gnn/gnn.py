import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
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