from .sampling import NodeSampler

import torch
import random

class RandomSampler(NodeSampler):
    def __init__(self, n):
        super().__init__(n, 'random_sampler')

    def sample_nodes(self, data, n=None):
        data = data.clone()

        n = n or self.n
        if isinstance(n, float):
            n = int(data.num_nodes*n)
        
        nodes = random.sample(range(data.num_nodes), k=n)
        nodes = torch.tensor(nodes)

        sample_mask = torch.zeros(data.num_nodes)
        sample_mask[nodes] = True

        return sample_mask
    
    def __call__(self, data, n=None):
        data = data.clone()

        sample_mask = self.sample_nodes(data, n)

        data.sample_mask = sample_mask

        return data