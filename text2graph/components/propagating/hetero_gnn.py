from ..component import Component
from ..component_registry import register_component

import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData


@register_component('hetero_gnn_propagator')
class HeteroGNNPropagator(Component):
    def __init__(
            self,
            hidden_channels=16,
            epochs=200,
            patience=None,
            lr=1e-3,
            num_layers=2,
            node_types=None,
            target_node_type='documents',
            label_attribute='pseudo_y',
            graph_embedding_attribute='x',
            propagating_mask=None
        ):
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.num_layers = num_layers
        self.node_types = node_types
        self.target_node_type = target_node_type
        self.label_attribute = label_attribute
        self.graph_embedding_attribute = graph_embedding_attribute
        self.propagating_mask = propagating_mask

        self.str_parameters = {
            'hidden_channels': hidden_channels,
            'epochs': epochs,
            'patience': patience,
            'lr': lr,
            'num_layers': num_layers,
            'node_types': node_types,
            'target_node_type': target_node_type,
            'label_attribute': label_attribute,
            'graph_embedding_attribute': graph_embedding_attribute
        }

    def _select_edge_types(self, data, node_types):
        return [
            edge_type for edge_type in data.edge_types
            if edge_type[0] in node_types and edge_type[-1] in node_types
        ]

    def propagate(self, data, node_types, edge_types, train_mask):
        target_data = data[self.target_node_type]

        unique_labels, new_labels = torch.unique(target_data[self.label_attribute][train_mask], sorted=True, return_inverse=True)
        num_classes = len(unique_labels)

        x_dict = {node_type: data[node_type][self.graph_embedding_attribute] for node_type in node_types}
        edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in edge_types}

        model = HeteroGNN(
            edge_types=edge_types,
            hidden_channels=self.hidden_channels,
            out_channels=num_classes,
            num_layers=self.num_layers
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        min_loss = None
        curr_patience = 0
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            out_dict = model(x_dict, edge_index_dict)
            out = out_dict[self.target_node_type]
            loss = F.cross_entropy(out[train_mask], new_labels)

            if self.patience is not None and curr_patience == self.patience:
                print(f'Early stopping due to patience = {self.patience}')
                break

            patience_str = ''
            if min_loss is None or loss < min_loss:
                curr_patience = 0
                min_loss = loss
            else:
                curr_patience += 1
                patience_str = f', patience at {curr_patience}'

            print(f'Epoch {epoch}: Loss = {loss}{patience_str}')
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out_dict = model(x_dict, edge_index_dict)
            out = out_dict[self.target_node_type]
            probs, preds = torch.softmax(out, 1).max(dim=1)
            preds = unique_labels[preds]

        return preds, probs

    def run(self, context):
        data = context['graph']

        if not isinstance(data, HeteroData):
            raise ValueError('HeteroGCNPropagator requires a heterogeneous graph')

        node_types = self.node_types or data.node_types

        if self.target_node_type not in node_types:
            raise ValueError(f'"{self.target_node_type}" not among the selected node_types {node_types}')

        for node_type in node_types:
            if node_type not in context['node_types']:
                raise ValueError(f'"{node_type}" not a valid type, only {context["node_types"]}')

        edge_types = self._select_edge_types(data, node_types)
        if not edge_types:
            raise ValueError(f'No edges found connecting node_types {node_types}')

        target_data = data[self.target_node_type]

        train_mask = target_data[self.label_attribute] != -1
        unlabeled_node_ids = torch.arange(target_data.num_nodes)[~train_mask]

        print(f'{train_mask.sum()} nodes with pseudo-labels')
        print(f'{target_data[self.label_attribute][train_mask].unique(return_counts=True)}')

        preds, probs = self.propagate(data, node_types, edge_types, train_mask)

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(target_data.num_nodes)]

        propagating_mask = self.propagating_mask if self.propagating_mask is not None else torch.ones(target_data.num_nodes, dtype=torch.bool)
        propagating_mask = (propagating_mask & (~train_mask))
        for node_id, pred, prob in zip(unlabeled_node_ids, preds[propagating_mask], probs[propagating_mask]):
            context['label_info'][node_id] = {'source': 'hetero_gcn_propagator', 'prob': prob.item()}
            target_data[self.label_attribute][node_id] = pred

        return context