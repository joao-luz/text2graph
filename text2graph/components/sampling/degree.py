from ..component import Component
from ..component_registry import register_component
from ...utils import get_same_type_edges

import torch

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree


@register_component('degree_sampler')
class DegreeSampler(Component):
    def __init__(self, n=100, node_type='documents', mask_name='to_label', label_attribute='pseudo_y', edge_link_name='similar'):
        self.n = n
        self.node_type = node_type
        self.mask_name = mask_name
        self.label_attribute = label_attribute
        self.edge_link_name = edge_link_name

        self.str_parameters = {
            'n': n,
            'node_type': node_type,
            'mask_name': mask_name,
            'label_attribute': label_attribute,
            'edge_link_name': edge_link_name,
        }

    def sample_nodes(self, edge_index, num_nodes, mask):
        n = self.n
        if isinstance(n, float):
            n = int(num_nodes * n)

        degrees = degree(edge_index[1], num_nodes)
        candidates = torch.nonzero(mask).flatten()

        order = torch.sort(degrees[candidates], descending=True, stable=True).indices

        return candidates[order[:n]]

    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')

        type_data = data[self.node_type] if isinstance(data, HeteroData) else data
        edge_index = get_same_type_edges(data, self.node_type, self.edge_link_name).edge_index

        if type_data.get(self.label_attribute) is not None:
            mask = type_data[self.label_attribute] == -1
        else:
            mask = torch.ones(type_data.num_nodes, dtype=torch.bool)

        sampled_indices = self.sample_nodes(edge_index, type_data.num_nodes, mask)
        mask = torch.zeros(type_data.num_nodes, dtype=torch.bool)
        mask[sampled_indices] = True

        context[self.mask_name] = mask

        return context
