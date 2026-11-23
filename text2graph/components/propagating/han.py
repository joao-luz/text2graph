import torch

from ..component_registry import register_component
from ...gnn import HAN
from .base import HeterogeneousPropagator


@register_component('han_propagator')
class HANPropagator(HeterogeneousPropagator):
    """Propagates labels to `target_node_type` with a Heterogeneous Attention Network.

    Args:
        heads: Attention heads per HANConv layer (`hidden_channels` must be divisible by it).
        dropout: Attention dropout.
        store_attention: If True, stores the trained model's semantic attention
            weights (one dict per layer) in `context[attention_attribute]`.
        attention_attribute: Context key for the attention weights.
        **kwargs: See `HeterogeneousPropagator` and `GNNPropagator`.
    """

    source = 'han_propagator'

    def __init__(self, heads=8, dropout=0.0, store_attention=False, attention_attribute='type_attention', **kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.dropout = dropout
        self.store_attention = store_attention
        self.attention_attribute = attention_attribute
        self.str_parameters.update({
            'heads': heads,
            'dropout': dropout,
            'store_attention': store_attention
        })

    def _build_model(self, inputs, num_classes):
        return HAN(
            metadata=self._metadata(inputs),
            hidden_channels=self.hidden_channels,
            out_channels=num_classes,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout
        )

    def _after_training(self, model, inputs, context):
        if not self.store_attention:
            return

        x_dict, edge_index_dict = inputs
        with torch.no_grad():
            _, attention_weights = model(x_dict, edge_index_dict, return_semantic_attention_weights=True)
        context[self.attention_attribute] = attention_weights
