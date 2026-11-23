import logging

from ..component import Component

import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


class GNNPropagator(Component):
    """Base class for propagators that train a neural model on the labeled nodes
    and use it to label the remaining ones.

    Subclasses implement `_prepare`, `_build_model` and `_forward`; this class
    handles the training loop (with optional early stopping), prediction and
    writing the predicted labels back into the graph and `context['label_info']`.

    Args:
        hidden_channels: Width of the hidden layers.
        epochs: Maximum number of training epochs.
        patience: Stop after this many epochs without the training loss improving
            (None disables early stopping).
        lr: Adam learning rate.
        num_layers: Number of layers in the model.
        label_attribute: Node attribute holding the labels; -1 marks unlabeled nodes.
            Predictions are written back into it.
        graph_embedding_attribute: Node attribute holding the input features.
        propagating_mask: Optional boolean mask over the target nodes; only unlabeled
            nodes inside it receive a predicted label (None means all of them).
    """

    source = None

    def __init__(
            self,
            hidden_channels=16,
            epochs=200,
            patience=None,
            lr=1e-3,
            num_layers=2,
            label_attribute='pseudo_y',
            graph_embedding_attribute='x',
            propagating_mask=None
        ):
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.num_layers = num_layers
        self.label_attribute = label_attribute
        self.graph_embedding_attribute = graph_embedding_attribute
        self.propagating_mask = propagating_mask

        self.str_parameters = {
            'hidden_channels': hidden_channels,
            'epochs': epochs,
            'patience': patience,
            'lr': lr,
            'num_layers': num_layers,
            'label_attribute': label_attribute,
            'graph_embedding_attribute': graph_embedding_attribute
        }

    def _prepare(self, context):
        """Validate the graph and return `(inputs, target_data)`: whatever `_forward`
        needs, and the node storage whose labels are being propagated."""
        raise NotImplementedError

    def _build_model(self, inputs, num_classes):
        """Return a fresh, untrained model."""
        raise NotImplementedError

    def _forward(self, model, inputs):
        """Return the logits for the target nodes, shape [num_target_nodes, num_classes]."""
        raise NotImplementedError

    def _after_training(self, model, inputs, context):
        """Called with the trained model, after labels are written. No-op by default."""

    def _train(self, model, inputs, train_mask, targets):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        min_loss = None
        curr_patience = 0
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            out = self._forward(model, inputs)
            loss = F.cross_entropy(out[train_mask], targets)

            if self.patience is not None and curr_patience == self.patience:
                logger.info('Early stopping due to patience = %s', self.patience)
                break

            patience_str = ''
            if min_loss is None or loss.item() < min_loss:
                curr_patience = 0
                min_loss = loss.item()
            else:
                curr_patience += 1
                patience_str = f', patience at {curr_patience}'

            logger.debug('Epoch %d: Loss = %s%s', epoch, loss.item(), patience_str)
            loss.backward()
            optimizer.step()

    def propagate(self, inputs, target_data, train_mask):
        """Train a model on the labeled target nodes and predict every target node.

        Returns:
            `(preds, probs, model)`: predicted labels (in the original label space)
            and their softmax probabilities for all target nodes, and the trained model.
        """
        unique_labels, targets = torch.unique(target_data[self.label_attribute][train_mask], sorted=True, return_inverse=True)

        model = self._build_model(inputs, len(unique_labels))
        self._train(model, inputs, train_mask, targets)

        model.eval()
        with torch.no_grad():
            out = self._forward(model, inputs)
            probs, preds = torch.softmax(out, 1).max(dim=1)
            preds = unique_labels[preds]

        return preds, probs, model

    def run(self, context):
        inputs, target_data = self._prepare(context)
        num_nodes = target_data.num_nodes

        train_mask = target_data[self.label_attribute] != -1
        if not train_mask.any():
            raise ValueError(f'No labeled nodes to train on ("{self.label_attribute}" is -1 everywhere)')

        preds, probs, model = self.propagate(inputs, target_data, train_mask)

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(num_nodes)]

        propagating_mask = self.propagating_mask if self.propagating_mask is not None else torch.ones(num_nodes, dtype=torch.bool)
        propagating_mask = propagating_mask & ~train_mask

        for node_id in torch.nonzero(propagating_mask).flatten().tolist():
            context['label_info'][node_id] = {'source': self.source, 'prob': probs[node_id].item()}
            target_data[self.label_attribute][node_id] = preds[node_id]

        self._after_training(model, inputs, context)

        return context


class HomogeneousPropagator(GNNPropagator):
    """GNNPropagator over a single node type.

    Works on a homogeneous graph, or on one node type of a HeteroData graph
    (only that type's features and its own `edge_index` are used).

    Args:
        node_type: Node type whose labels are propagated.
        **kwargs: See `GNNPropagator`.
    """

    def __init__(self, node_type='documents', **kwargs):
        super().__init__(**kwargs)
        self.node_type = node_type
        self.str_parameters['node_type'] = node_type

    def _prepare(self, context):
        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')

        data = context['graph']
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        return type_data, type_data


class HeterogeneousPropagator(GNNPropagator):
    """GNNPropagator over several node types of a HeteroData graph.

    Every edge type whose source and destination are both among `node_types` is used.
    Inputs are `(x_dict, edge_index_dict)`; `_metadata(inputs)` gives the
    `(node_types, edge_types)` pair PyG models expect.

    Args:
        node_types: Node types to include (None means all of the graph's types).
        target_node_type: Node type whose labels are propagated.
        **kwargs: See `GNNPropagator`.
    """

    def __init__(self, node_types=None, target_node_type='documents', **kwargs):
        super().__init__(**kwargs)
        self.node_types = node_types
        self.target_node_type = target_node_type
        self.str_parameters['node_types'] = node_types
        self.str_parameters['target_node_type'] = target_node_type

    def _prepare(self, context):
        data = context['graph']

        if not isinstance(data, HeteroData):
            raise ValueError(f'{type(self).__name__} requires a heterogeneous graph')

        node_types = list(self.node_types or data.node_types)

        if self.target_node_type not in node_types:
            raise ValueError(f'"{self.target_node_type}" not among the selected node_types {node_types}')

        for node_type in node_types:
            if node_type not in context['node_types']:
                raise ValueError(f'"{node_type}" not a valid type, only {context["node_types"]}')

        edge_types = [
            edge_type for edge_type in data.edge_types
            if edge_type[0] in node_types and edge_type[-1] in node_types
        ]
        if not edge_types:
            raise ValueError(f'No edges found connecting node_types {node_types}')

        x_dict = {node_type: data[node_type][self.graph_embedding_attribute] for node_type in node_types}
        edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in edge_types}

        return (x_dict, edge_index_dict), data[self.target_node_type]

    @staticmethod
    def _metadata(inputs):
        x_dict, edge_index_dict = inputs
        return list(x_dict), list(edge_index_dict)

    def _forward(self, model, inputs):
        x_dict, edge_index_dict = inputs
        return model(x_dict, edge_index_dict)[self.target_node_type]
