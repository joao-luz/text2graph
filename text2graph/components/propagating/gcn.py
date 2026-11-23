from ..component_registry import register_component
from ...gnn import GCN
from .base import HomogeneousPropagator


@register_component('gcn_propagator')
class GCNPropagator(HomogeneousPropagator):
    """Propagates labels with a GCN trained on the labeled nodes of `node_type`.

    Args: see `HomogeneousPropagator` and `GNNPropagator`.
    """

    source = 'gcn_propagator'

    def _build_model(self, data, num_classes):
        return GCN(
            in_channels=data[self.graph_embedding_attribute].shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=num_classes,
            num_layers=self.num_layers
        )

    def _forward(self, model, data):
        return model(data[self.graph_embedding_attribute], data.edge_index)
