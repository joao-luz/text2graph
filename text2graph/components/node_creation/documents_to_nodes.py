from ..component import Component
from ..component_registry import register_component
from ...utils import data_to_hetero

import torch
from torch_geometric.data import Data, HeteroData


@register_component('documents_to_nodes')
class DocumentsToNodes(Component):
    """Adds one node per document, usually as a pipeline's first step.

    Each node stores its document in the `text` attribute. If the context has no
    graph yet, this creates a homogeneous `Data` graph. Otherwise the documents are
    added as a new node type, and a homogeneous graph is first converted to
    `HeteroData` (its nodes keep their node type, and its edges become
    `(type, 'to', type)` edges).

    Adds `node_type` to `context['node_types']` and sets `context['graph']`.

    Args:
        source: Context key holding the list of document strings.
        node_type: Name of the new node type.
        labels: Optional context key holding one label per document (a list or a
            tensor), stored on the nodes as `labels_attribute`. Usually the ground
            truth, for evaluation or for `GroundTruthLabeler`.
        labels_attribute: Node attribute the labels are stored in.

    Raises:
        KeyError: If `source`, or `labels` when given, isn't in the context.
        TypeError: If `context['graph']` is neither `None`, `Data` nor `HeteroData`.
    """

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
            graph = data_to_hetero(graph, node_type=existing_node_type)
            node_store = graph[self.node_type]
            context['node_types'].append(self.node_type)

        elif isinstance(graph, HeteroData):
            node_store = graph[self.node_type]
            context['node_types'].append(self.node_type)

        else:
            raise TypeError(f'Expected graph to be torch_geometric.data.Data, HeteroData, or None, got {type(graph).__name__}')

        node_store.text = documents
        node_store.num_nodes = len(documents)

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