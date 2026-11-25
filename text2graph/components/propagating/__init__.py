"""Components that label the remaining nodes from the labeled ones.

Graph neural networks, trained on the labeled nodes:

- `GCNPropagator` (`gcn_propagator`) and `MLPPropagator` (`mlp_propagator`): one
  node type (the MLP ignores edges).
- `HeteroGNNPropagator` (`hetero_gnn_propagator`) and `HANPropagator`
  (`han_propagator`): several node types.

Others:

- `LabelPropagator` (`label_propagator`): classic label propagation, no training.
- `LMPropagator` (`lm_propagator`): a text classifier fine-tuned on the labeled
  nodes' text.

To add a GNN propagator, subclass `HomogeneousPropagator` or
`HeterogeneousPropagator` (both built on `GNNPropagator`).
"""

from .base import GNNPropagator, HomogeneousPropagator, HeterogeneousPropagator
from .gcn import GCNPropagator
from .mlp import MLPPropagator
from .language_model import LMPropagator
from .hetero_gnn import HeteroGNNPropagator
from .han import HANPropagator
from .label_propagation import LabelPropagator

__all__ = [
    'GNNPropagator',
    'HomogeneousPropagator',
    'HeterogeneousPropagator',

    'GCNPropagator',
    'MLPPropagator',
    'LMPropagator',
    'HeteroGNNPropagator',
    'HANPropagator',
    'LabelPropagator',
]
