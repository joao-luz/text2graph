from ..component import Component
from ..component_registry import register_component

import torch

from torch_geometric.data import HeteroData


@register_component('knn_edge_creator')
class KNNEdgeCreator(Component):
    def __init__(self, k, representation_attr='x', node_type='documents'):
        self.k = k
        self.representation_attr = representation_attr
        self.node_type = node_type

        self.str_parameters = {
            'k': k,
            'representation_attr': representation_attr,
            'node_type': node_type
        }

    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')

        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        embeddings = type_data[self.representation_attr]

        sim_matrix = torch.matmul(embeddings, embeddings.T)

        n = len(embeddings)
        _,indices = torch.topk(sim_matrix, k=self.k+1, dim=1)

        edge_index_list = []
        for i in range(n):
            for j in indices[i][1:]:
                edge_index_list.append([i, j.item()])

        if len(edge_index_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

        type_data.edge_index = edge_index

        return context