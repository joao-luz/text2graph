from ..component import Component
from ..component_registry import register_component

import torch
import random

from torch_geometric.data import HeteroData


@register_component('random_sampler')
class RandomSampler(Component):
    def __init__(self, n=100, node_type='documents', mask_name='to_label'):
        self.n = n
        self.node_type = node_type
        self.mask_name = mask_name

        self.str_parameters = {
            'n': n,
            'node_type': node_type,
            'mask_name': mask_name,
        }

    def sample_nodes(self, indices):        
        shuffled_indices = torch.randperm(len(indices))
        sampled_indices = indices[shuffled_indices[:self.n]]

        return sampled_indices

    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')
        
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        if type_data.get(self.label_attribute) is not None:
            mask = type_data[self.label_attribute] == -1
        else:
            mask = torch.ones(type_data.num_nodes, dtype=torch.bool)

        indices = torch.tensor(type_data.num_nodes)[mask]

        sampled_indices = self.sample_nodes(indices)
        mask = torch.zeros(type_data.num_nodes, dtype=torch.bool)
        mask[sampled_indices] = True

        context[self.mask_name] = mask

        return context
