from .component import Component

import torch
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph

class SimilarityEstimator(Component):
    def __init__(self, threshold, name):
        super().__init__(name)
        self.threshold = threshold

    def compute_similarities(self, docs):
        pass

    def __call__(self, G):
        pass

def _compute_minimum_threshold(similarities):
    mask = ~torch.eye(similarities.size(0), dtype=bool)
    rows_without_diag = similarities[mask].view(similarities.size(0), -1)
    row_max = rows_without_diag.max(dim=1).values
    minimum_threshold = row_max.min()

    return minimum_threshold

class EmbeddingSimilarity(SimilarityEstimator):
    def __init__(self, model=None, embedding_feature=None, threshold=None, k=None, mode='mst'):
        super().__init__(threshold, 'embedding_similarity')

        assert model or embedding_feature, 'Must pass either a model or the node feature to obtain the embedding from.'
        assert mode in ['mst', 'threshold', 'knn']

        self.model = model
        self.embedding_feature = embedding_feature
        self.mode = mode
        self.k = k

        self.str_parameters = {
            'mode': self.mode
        }
        if self.mode == 'knn':
            self.str_parameters['k'] = self.k
        else:
            self.str_parameters['threshold'] = self.threshold
    
    def _get_embeddings(self, G):
        if self.embedding_feature:
            return torch.stack([G.nodes[n][self.embedding_feature] for n in G.nodes])
        else:
            texts = [G.nodes[n]['text'] for n in G.nodes]
            return self.model.encode(texts, convert_to_tensor=True)

    def _compute_similarities(self, embeddings):
        normalized = F.normalize(embeddings, 2, dim=1)
        similarities = torch.matmul(normalized, normalized.T)

        return similarities
    
    def __call__(self, G):
        embeddings = torch.stack([G.nodes[n][self.embedding_feature] for n in G.nodes])

        if self.mode == 'threshold' or self.mode == 'mst':
            similarities = self._compute_similarities(embeddings)
            threshold = self.threshold or _compute_minimum_threshold(similarities)

            mask = similarities >= threshold
            edges = similarities.numpy()
            edges[~mask] = 0.0

            if self.mode == 'mst':
                distance_matrix = 1.0 - edges
                graph = csr_matrix(distance_matrix)
                mst = minimum_spanning_tree(graph)
                mst_dense = mst.toarray()
                edges = mst_dense
            
        elif self.mode == 'knn':
            edges = kneighbors_graph(embeddings, n_neighbors=self.k, mode='distance', n_jobs=-1)
            edges = edges.toarray()
        
        print(f'Will create {len(edges.nonzero()[0])} edges')

        for i,j in zip(*edges.nonzero()):
            if i < j:
                G.add_edge(i, j, weight=edges[i][j].item())
        
        return G
