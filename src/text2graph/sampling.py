from .component import Component

import random
import torch
from torch_geometric.utils import degree

class NodeSampler(Component):
    def __init__(self, n, name=''):
        self.n = n
        self.name = name

        self.str_parameters = {
            'n': n
        }

    def sample_nodes(self, data, n):
        pass

class RandomSampler(NodeSampler):
    def __init__(self, n):
        super().__init__(n, 'random_sampler')

    def sample_nodes(self, data, n=None):
        n = n or self.n
        if isinstance(n, float):
            n = int(data.num_nodes*n)
        
        nodes = random.sample(range(data.num_nodes), k=n)
        nodes = torch.tensor(nodes)

        return nodes

class DegreeSampler(NodeSampler):
    def __init__(self, n):
        super().__init__(n, 'degree_sampler')

    def sample_nodes(self, data, n=None):
        n = n or self.n
        if isinstance(n, float):
            n = int(data.num_nodes*n)

        degrees = degree(data.edge_index, data.num_nodes)
        _,indices = torch.sort(degrees, descending=True)

        return indices[:n]