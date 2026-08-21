from ..component import Component
from ..component_registry import register_component

import torch

from torch_geometric.data import HeteroData


@register_component('ground_truth_labeler')
class GroundTruthLabeler(Component):
    def __init__(self, sample_mask_name='to_label', node_type='documents', true_label_source='labels', label_attribute='pseudo_y'):
        super().__init__()

        self.sample_mask_name = sample_mask_name
        self.node_type = node_type
        self.true_label_source = true_label_source
        self.label_attribute = label_attribute

        self.str_parameters = {
            'sample_mask_name': sample_mask_name,
            'node_type': node_type,
            'label_attribute': label_attribute
        }

        if isinstance(true_label_source, str):
            self.str_parameters |= { 'true_label_source': true_label_source }

    def extract_labels(self, data, node_ids):
        ground_truths = data[self.label_attribute]

        labels = torch.full((data.num_nodes, ), -1)
        for node_id in node_ids:
            labels[node_id] = ground_truths[node_id]

        return labels
    
    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')
        
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        label_mask = context[self.sample_mask_name]
        node_ids = torch.nonzero(label_mask).flatten().tolist()
        labels = torch.full((type_data.num_nodes,), -1)

        true_labels = context[self.true_label_source] if isinstance(self.true_label_source, str) else self.true_label_source
        labels[label_mask] = true_labels[label_mask]
        
        if not context.get('pseudo_label_info'):
            context['pseudo_label_info'] = [{} for _ in range(type_data.num_nodes)]

        for id in node_ids:
            context['pseudo_label_info'][id] = {'source': 'ground_truth', 'prob': 1.0}
        
        type_data[self.label_attribute] = labels

        return context