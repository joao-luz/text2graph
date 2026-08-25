from ..component import Component
from ..component_registry import register_component

import torch
import torch.nn.functional as F

from torch_geometric.data import HeteroData


@register_component('knn_labeler')
class KNNLabeler(Component):
    def __init__(self,
        k,
        target_node_type,
        source_node_types=None,
        label_attribute='pseudo_y',
        source_label_attribute='y',
        representation_attr='x',
        threshold=0.5,
        mask_name=None
    ):
        self.k = k
        self.target_node_type = target_node_type
        self.source_node_types = source_node_types
        self.label_attribute = label_attribute
        self.source_label_attribute = source_label_attribute
        self.representation_attr = representation_attr
        self.threshold = threshold
        self.mask_name = mask_name

        self.str_parameters = {
            'k': k,
            'target_node_type': target_node_type,
            'source_node_types': source_node_types,
            'label_attribute': label_attribute,
            'source_label_attribute': source_label_attribute,
            'threshold': threshold
        }

    def compute_labels(self, target_data, target_ids, source_reps, source_labels):
        x_target = getattr(target_data, self.representation_attr)[target_ids]

        target_norm = F.normalize(x_target, p=2, dim=-1)
        source_norm = F.normalize(source_reps, p=2, dim=-1)

        sim = target_norm @ source_norm.t()  # [n_target, n_source]

        k = min(self.k, source_reps.size(0))
        _, topk_idx = sim.topk(k, dim=-1, largest=True)  # [n_target, k]

        neighbor_labels = source_labels[topk_idx]  # [n_target, k]

        preds = torch.full((len(target_ids),), -1, dtype=torch.long)
        probs = torch.zeros(len(target_ids))

        for i in range(len(target_ids)):
            labels, counts = torch.unique(neighbor_labels[i], return_counts=True)
            majority_idx = torch.argmax(counts)
            majority_label = labels[majority_idx]
            agreement = counts[majority_idx].item() / k

            if agreement >= self.threshold:
                preds[i] = majority_label
                probs[i] = agreement

        return preds, probs

    def run(self, context):
        data = context['graph']

        if self.target_node_type not in context['node_types']:
            raise ValueError(f'"{self.target_node_type}" not a valid type, only {context["node_types"]}')

        source_node_types = self.source_node_types or [t for t in context['node_types'] if t != self.target_node_type]

        for node_type in source_node_types:
            if node_type not in context['node_types']:
                raise ValueError(f'"{node_type}" not a valid type, only {context["node_types"]}')

        target_data = data[self.target_node_type] if isinstance(data, HeteroData) else data

        if target_data.get(self.label_attribute) is None:
            target_data[self.label_attribute] = torch.full((target_data.num_nodes,), -1, dtype=torch.long)

        if self.mask_name:
            target_mask = context[self.mask_name]
        else:
            target_mask = target_data[self.label_attribute] == -1

        target_ids = torch.nonzero(target_mask).flatten().tolist()

        source_reps, source_labels = [], []
        for node_type in source_node_types:
            source_data = data[node_type] if isinstance(data, HeteroData) else data
            labels = source_data[self.source_label_attribute]
            labeled_mask = labels != -1

            source_reps.append(getattr(source_data, self.representation_attr)[labeled_mask])
            source_labels.append(labels[labeled_mask])

        source_reps = torch.cat(source_reps, dim=0)
        source_labels = torch.cat(source_labels, dim=0)

        if source_reps.size(0) == 0:
            raise ValueError('No labeled source nodes found to compute KNN labels from.')

        preds, probs = self.compute_labels(target_data, target_ids, source_reps, source_labels)

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(target_data.num_nodes)]

        for i, node_id in enumerate(target_ids):
            if preds[i] != -1:
                target_data[self.label_attribute][node_id] = preds[i]
                context['label_info'][node_id] = {'source': 'knn_labeler', 'prob': probs[i].item()}

        context['graph'] = data

        return context