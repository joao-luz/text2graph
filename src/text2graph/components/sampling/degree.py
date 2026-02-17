from .sampling import NodeSampler

import torch
from torch_geometric.utils import degree

class DegreeSampler(NodeSampler):
    def __init__(self, n):
        super().__init__(n, 'degree_sampler')

    def sample_nodes(self, data, n=None):
        n = n or self.n
        if isinstance(n, float):
            n = int(data.num_nodes*n)

        degrees = degree(data.edge_index, data.num_nodes)
        _,indices = torch.sort(degrees, descending=True)
        nodes = indices[:n]

        sample_mask = torch.zeros(data.num_nodes)
        sample_mask[nodes] = True

        return sample_mask
    
    def __call__(self, data, n=None):
        data = data.clone()

        sample_mask = self.sample_nodes(data, n)

        data.sample_mask = sample_mask

        return data
        