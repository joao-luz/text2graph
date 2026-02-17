from ..component import Component

class FeatureExtractor(Component):
    def __init__(self, name):
        super().__init__(name)

    def compute_representations(self, docs):
        pass

    def __call__(self):
        pass

