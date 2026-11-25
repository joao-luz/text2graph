"""The graph models trained by the GNN propagators: `GCN`, `MLP`, `HeteroGNN` and `HAN`."""

from .gnn import GCN, MLP, HeteroGNN, HAN

__all__ = [
    'GCN',
    'MLP',
    'HeteroGNN',
    'HAN',
]
