from ..component_registry import register_component
from ...gnn import MLP
from .base import HomogeneousPropagator


@register_component('mlp_propagator')
class MLPPropagator(HomogeneousPropagator):
    """Propagates labels with an MLP on the node features of `node_type` (edges are ignored).

    Args: see `HomogeneousPropagator` and `GNNPropagator`.
    """

    source = 'mlp_propagator'
    requires_edges = False

    def _build_model(self, data, num_classes):
        return MLP(
            in_channels=data[self.graph_embedding_attribute].shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=num_classes,
            num_layers=self.num_layers
        )

    def _forward(self, model, data):
        return model(data[self.graph_embedding_attribute])
