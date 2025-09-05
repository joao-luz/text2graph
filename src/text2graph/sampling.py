import random

from .component import Component

class NodeSampler(Component):
    def __init__(self, n, name=''):
        self.n = n
        self.name = name

        self.str_parameters = {
            'n': n
        }

    def sample_nodes(self, G, n):
        pass

class RandomSampler(NodeSampler):
    def __init__(self, n):
        super().__init__(n, 'random_sampler')

    def sample_nodes(self, G, n=None):
        n = n or self.n
        if isinstance(n, float):
            n = int(len(G.nodes)*n)
        
        nodes = random.sample(range(len(G.nodes)), k=n)

        return nodes

class DegreeSampler(NodeSampler):
    def __init__(self, n):
        super().__init__(n, 'degree_sampler')

    def sample_nodes(self, G, n=None):
        n = n or self.n
        if isinstance(n, float):
            n = int(len(G.nodes)*n)

        node_degrees = list(G.degree(G.nodes))
        node_degrees.sort(key=lambda t: t[1], reverse=True)
        nodes = [t[0] for t in node_degrees[:n]]

        return nodes