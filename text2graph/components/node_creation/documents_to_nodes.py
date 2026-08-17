from ..component import Component
from ..component_registry import register_component

import torch
from torch_geometric.data import Data, HeteroData


@register_component('documents_to_nodes')
class DocumentsToNodes(Component):
    def __init__(self, source='documents', node_type='documents', labels=None, labels_attribute='y'):
        self.source = source
        self.node_type = node_type
        self.labels = labels
        self.labels_attribute = labels_attribute

        self.str_parameters = {
            'source': source,
            'node_type': node_type,
        }
        if labels is not None:
            self.str_parameters |= {
                'labels': labels,
                'labels_attribute': labels_attribute,
            }

    def run(self, context):
        documents = context[self.source]

        graph = context.get('graph')

        if graph is None:
            graph = Data()
            node_store = graph
            context['node_types'] = [self.node_type]

        elif isinstance(graph, Data):
            existing_node_type = context['node_types'][0]
            graph = self._data_to_hetero(graph, node_type=existing_node_type)
            node_store = graph[self.node_type]
            context['node_types'].append(self.node_type)

        elif isinstance(graph, HeteroData):
            node_store = graph[self.node_type]
            context['node_types'].append(self.node_type)

        else:
            raise TypeError(f'Expected graph to be torch_geometric.data.Data, HeteroData, or None, got {type(graph).__name__}')

        node_store.text = documents

        if self.labels is not None:
            if self.labels not in context:
                raise KeyError(f'Labels key "{self.labels}" was not found in context.')

            labels = context[self.labels]

            if torch.is_tensor(labels):
                node_store[self.labels_attribute] = labels
            else:
                node_store[self.labels_attribute] = torch.as_tensor(labels)

        context['graph'] = graph
        
        return context

    @staticmethod
    def _data_to_hetero(data, node_type='documents'):
        hetero = HeteroData()

        node_store = hetero[node_type]

        for key, value in data.items():
            if key in {'edge_index', 'edge_attr', 'edge_weight'}:
                continue

            node_store[key] = value

        if data.edge_index is not None:
            edge_type = (node_type, 'to', node_type)

            hetero[edge_type].edge_index = data.edge_index

            if getattr(data, 'edge_attr', None) is not None:
                hetero[edge_type].edge_attr = data.edge_attr

            if getattr(data, 'edge_weight', None) is not None:
                hetero[edge_type].edge_weight = data.edge_weight

        return hetero