from ..component import Component
from ..component_registry import register_component
from ...gnn.gcn import GCN

import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData


@register_component('gcn_propagator')
class GCNPropagator(Component):
    def __init__(
            self, 
            hidden_channels=16, 
            epochs=200, 
            patience=None, 
            lr=1e-3, 
            node_type='documents', 
            label_attribute='pseudo_y',
            graph_embedding_attribute='x'
        ):
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.node_type = node_type
        self.label_attribute = label_attribute
        self.graph_embedding_attribute = graph_embedding_attribute

        self.str_parameters = {
            'hidden_channels': hidden_channels,
            'epochs': epochs,
            'patience': patience,
            'lr': lr,
            'node_type': node_type,
            'label_attribute': label_attribute,
            'graph_embedding_attribute': graph_embedding_attribute
        }

    def propagate(self, data, train_mask):
        unique_labels, new_labels = torch.unique(data[self.label_attribute][train_mask], sorted=True, return_inverse=True)

        num_classes = len(unique_labels)

        model = GCN(
            in_channels=data[self.graph_embedding_attribute].shape[1], 
            hidden_channels=self.hidden_channels, 
            out_channels=num_classes
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        min_loss = None
        curr_patience = 0
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(data[self.graph_embedding_attribute], data.edge_index, data.edge_weight)
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
            out = model(data[self.graph_embedding_attribute], data.edge_index, data.edge_weight)
            probs, preds = torch.softmax(out, 1).max(dim=1)
            preds = unique_labels[preds]

        return preds, probs
    
    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')
        
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        train_mask = type_data[self.label_attribute] != -1
        unlabeled_node_ids = torch.arange(type_data.num_nodes)[~train_mask]

        print(f'{train_mask.sum()} nodes with pseudo-labels')
        print(f'{data[self.label_attribute][train_mask].unique(return_counts=True)}')

        preds, probs = self.propagate(type_data, train_mask)

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(type_data.num_nodes)]

        for node_id,pred,prob in zip(unlabeled_node_ids, preds[~train_mask], probs[~train_mask]):
            context['label_info'][node_id] = {'source': 'lm_propagator', 'prob': prob.item()}
            type_data[self.label_attribute][node_id] = pred
        
        return context