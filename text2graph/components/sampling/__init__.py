"""Components that pick which unlabeled nodes to label next.

Each stores a boolean mask in the context (by default under `to_label`), which
labelers read.

- `RandomSampler` (`random_sampler`): random nodes.
- `KMeansSampler` (`kmeans_sampler`): one representative node per k-means cluster.
- `DegreeSampler` (`degree_sampler`): the best-connected nodes.
- `DMASampler` (`dma_sampler`): nodes that cover the graph and are likely to be
  labeled reliably, picked batch by batch.
"""

from .kmeans import KMeansSampler
from .random import RandomSampler
from .degree import DegreeSampler
from .dma import DMASampler

__all__ = [
    'KMeansSampler',
    'RandomSampler',
    'DegreeSampler',
    'DMASampler'
]
