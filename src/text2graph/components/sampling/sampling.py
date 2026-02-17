from ..component import Component

class NodeSampler(Component):
    def __init__(self, n, name=''):
        self.n = n
        self.name = name

        self.str_parameters = {
            'n': n
        }

    def sample_nodes(self, data, n):
        pass
