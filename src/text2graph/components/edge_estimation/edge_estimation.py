from ..component import Component


class SimilarityEstimator(Component):
    def __init__(self, threshold, name):
        super().__init__(name)
        self.threshold = threshold

    def compute_similarities(self, docs):
        pass

    def __call__(self, G):
        pass


