from ..component import Component
from ..component_registry import register_component

import numpy as np
import torch

from sklearn.cluster import KMeans
from torch_geometric.data import HeteroData


def kmeans_sample(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings.numpy())
            
    sampled_indices = []
    for center in kmeans.cluster_centers_:
        dist = np.linalg.norm(embeddings - center, axis=1)
        sampled_indices.append(np.argmin(dist))

    sampled_indices = list(set(sampled_indices))

    return torch.tensor(sampled_indices)

@register_component('kmeans_sampler')
class KMeansSampler(Component):
    def __init__(self, n=100, node_type='documents', representation_attr='x', mask_name='to_label', label_attribute='pseudo_y'):
        self.node_type = node_type
        self.representation_attr = representation_attr
        self.n = n
        self.mask_name = mask_name
        self.label_attribute = label_attribute

        self.str_parameters = {
            'n': n,
            'representation_attr': representation_attr,
            'node_type': node_type,
            'mask_name': mask_name,
        }

    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')
        
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        if type_data.get(self.label_attribute) is not None:
            mask = type_data[self.label_attribute] == -1
        else:
            mask = torch.ones(type_data.num_nodes, dtype=torch.bool)

        embeddings = type_data[self.representation_attr][mask]

        print(type_data.num_nodes)

        sampled_indices = kmeans_sample(embeddings, self.n)
        mask = torch.zeros(type_data.num_nodes, dtype=torch.bool)
        mask[sampled_indices] = True

        context[self.mask_name] = mask

        return context
