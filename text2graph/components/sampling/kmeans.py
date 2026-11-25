from ..component import Component
from ..component_registry import register_component

import numpy as np
import torch

from sklearn.cluster import KMeans
from torch_geometric.data import HeteroData


def _kmeans_sample(embeddings, k, random_state=42):
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(embeddings.numpy())
            
    sampled_indices = []
    for center in kmeans.cluster_centers_:
        dist = np.linalg.norm(embeddings - center, axis=1)
        sampled_indices.append(np.argmin(dist))

    sampled_indices = list(set(sampled_indices))

    return torch.tensor(sampled_indices)

@register_component('kmeans_sampler')
class KMeansSampler(Component):
    """Picks `n` representative unlabeled nodes by clustering their features.

    The unlabeled nodes (`label_attribute` -1, or all nodes if the attribute
    doesn't exist yet) are split into `n` k-means clusters, and the node closest to
    each cluster center is picked (see `kmeans_sample`). This spreads the sample
    over the feature space, so it covers the data better than random picks. The
    clustering is deterministic: the same features and `random_state` always give
    the same sample.

    Stores the result in `context[mask_name]`, a boolean mask over the nodes of
    `node_type`.

    Args:
        n: Number of clusters, and so the most nodes picked; an int. Fewer are
            picked when two centers share their closest node.
        node_type: Node type to sample from.
        representation_attr: Node attribute holding the features to cluster.
        mask_name: Context key the mask is stored under.
        label_attribute: Node attribute telling labeled nodes (not -1) apart.
        random_state: Seed for k-means' initialization; change it to get a
            different (but still deterministic) sample, e.g. one per run.

    Raises:
        ValueError: If `node_type` isn't in the graph.
    """

    def __init__(self, n=100, node_type='documents', representation_attr='x', mask_name='to_label', label_attribute='pseudo_y', random_state=42):
        self.node_type = node_type
        self.representation_attr = representation_attr
        self.n = n
        self.mask_name = mask_name
        self.label_attribute = label_attribute
        self.random_state = random_state

        self.str_parameters = {
            'n': n,
            'representation_attr': representation_attr,
            'node_type': node_type,
            'mask_name': mask_name,
            'random_state': random_state,
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

        candidate_ids = torch.nonzero(mask).flatten()
        embeddings = type_data[self.representation_attr][candidate_ids]

        sampled_indices = candidate_ids[_kmeans_sample(embeddings, self.n, random_state=self.random_state)]
        mask = torch.zeros(type_data.num_nodes, dtype=torch.bool)
        mask[sampled_indices] = True

        context[self.mask_name] = mask

        return context
