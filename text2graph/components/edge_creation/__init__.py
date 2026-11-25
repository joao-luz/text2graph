"""Components that connect nodes.

- `KNNEdgeCreator` (`knn_edge_creator`): each node to its most similar nodes of
  the same type.
- `HeteroKNNEdgeCreator` (`hetero_knn_edge_creator`): each node of one type to
  its most similar nodes of another (or the same) type, with similarities as
  edge weights.
"""

from .knn import KNNEdgeCreator
from .hetero_knn import HeteroKNNEdgeCreator

__all__ = [
    'KNNEdgeCreator',
    'HeteroKNNEdgeCreator',
]
