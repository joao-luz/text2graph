from ..component import Component
from ..component_registry import register_component
from ...utils import get_same_type_edges

import torch

from torch_geometric.data import HeteroData
from torch_geometric.utils import degree


@register_component('degree_sampler')
class DegreeSampler(Component):
    """Picks the `n` unlabeled nodes with the most incoming edges.

    Well-connected nodes pass their labels on to many neighbors during
    propagation, so labeling them first can spread labels furthest. Degrees are
    counted over the edges between nodes of `node_type` (see
    `get_same_type_edges`). Ties go to the node with the lower id, so the sample
    is deterministic.

    Stores the result in `context[mask_name]`, a boolean mask over the nodes of
    `node_type`. Only nodes whose `label_attribute` is -1 can be picked (all
    nodes, if the attribute doesn't exist yet).

    Args:
        n: Number of nodes to pick, or a float for a fraction of all the node
            type's nodes. If fewer nodes are unlabeled, all of them are picked.
        node_type: Node type to sample from.
        mask_name: Context key the mask is stored under.
        label_attribute: Node attribute telling labeled nodes (not -1) apart.
        edge_link_name: Relation name of the edges to count, on `HeteroData` graphs.

    Raises:
        ValueError: If `node_type` isn't in the graph, or (on `HeteroData`) has no
            edges between its own nodes.
    """

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

    def _sample_nodes(self, edge_index, num_nodes, mask):
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

        sampled_indices = self._sample_nodes(edge_index, type_data.num_nodes, mask)
        mask = torch.zeros(type_data.num_nodes, dtype=torch.bool)
        mask[sampled_indices] = True

        context[self.mask_name] = mask

        return context
