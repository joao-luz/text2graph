from ..component import Component
from ..component_registry import register_component

import torch
import random

from torch_geometric.data import HeteroData


@register_component('random_sampler')
class RandomSampler(Component):
    """Picks `n` random unlabeled nodes, e.g. as the documents to send to an LLM labeler.

    Stores the result in `context[mask_name]`, a boolean mask over the nodes of
    `node_type`. Only nodes whose `label_attribute` is -1 can be picked (all
    nodes, if the attribute doesn't exist yet).

    Args:
        n: Number of nodes to pick, or a float for a fraction of all the node
            type's nodes. If fewer nodes are unlabeled, all of them are picked.
        node_type: Node type to sample from.
        mask_name: Context key the mask is stored under.
        label_attribute: Node attribute telling labeled nodes (not -1) apart.

    Raises:
        ValueError: If `node_type` isn't in the graph.
    """

    def __init__(self, n=100, node_type='documents', mask_name='to_label', label_attribute='pseudo_y'):
        self.n = n
        self.node_type = node_type
        self.mask_name = mask_name
        self.label_attribute = label_attribute

        self.str_parameters = {
            'n': n,
            'node_type': node_type,
            'mask_name': mask_name,
            'label_attribute': label_attribute,
        }

    def _sample_nodes(self, data, indices):
        n = self.n
        if isinstance(n, float):
            n = int(data.num_nodes*n)

        shuffled_indices = torch.randperm(len(indices))
        sampled_indices = indices[shuffled_indices[:n]]

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

        indices = torch.arange(type_data.num_nodes)[mask]

        sampled_indices = self._sample_nodes(type_data, indices)
        mask = torch.zeros(type_data.num_nodes, dtype=torch.bool)
        mask[sampled_indices] = True

        context[self.mask_name] = mask

        return context
