from ..component_registry import register_component
from ...gnn import HeteroGNN
from .base import HeterogeneousPropagator


@register_component('hetero_gnn_propagator')
class HeteroGNNPropagator(HeterogeneousPropagator):
    """Propagates labels to `target_node_type` with a heterogeneous GraphSAGE
    (one SAGEConv per edge type, summed) over the selected node types.

    Args: see `HeterogeneousPropagator` and `GNNPropagator`.
    """

    source = 'hetero_gnn_propagator'

    def _build_model(self, inputs, num_classes):
        _, edge_types = self._metadata(inputs)
        return HeteroGNN(
            edge_types=edge_types,
            hidden_channels=self.hidden_channels,
            out_channels=num_classes,
            num_layers=self.num_layers
        )
