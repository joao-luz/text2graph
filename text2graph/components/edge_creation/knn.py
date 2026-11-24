from ..component import Component
from ..component_registry import register_component

import torch

from torch_geometric.data import HeteroData


@register_component('knn_edge_creator')
class KNNEdgeCreator(Component):
    """Connects each node to its `k` most similar nodes of the same type.

    Similarity is the dot product of the node features, which is the cosine
    similarity for normalized features such as `SentenceEmbeddingExtractor`'s.
    Edges go from each node to its neighbors and aren't made symmetric. They have
    no weights, and replace any existing edges of the node type.

    On a homogeneous `Data` graph they're stored as the graph's `edge_index`; on a
    `HeteroData` graph, as the edge type `(node_type, link_name, node_type)`.

    Args:
        k: Number of neighbors per node.
        representation_attr: Node attribute holding the features to compare.
        node_type: Node type to connect.
        link_name: Relation name of the edge type, on `HeteroData` graphs.

    Raises:
        ValueError: If `node_type` isn't in the graph.
    """

    def __init__(self, k, representation_attr='x', node_type='documents', link_name='similar'):
        self.k = k
        self.representation_attr = representation_attr
        self.node_type = node_type
        self.link_name = link_name

        self.str_parameters = {
            'k': k,
            'representation_attr': representation_attr,
            'node_type': node_type,
            'link_name': link_name
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

        if isinstance(data, HeteroData):
            data[self.node_type, self.link_name, self.node_type].edge_index = edge_index
        else:
            data.edge_index = edge_index

        return context