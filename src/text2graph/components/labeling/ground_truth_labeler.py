from .labeling import Labeler

import torch


class GroundTruthLabeler(Labeler):
    def __init__(self, sample_mask_attribute='sample_mask', label_attribute='true_label'):
        super().__init__('ground_truth_labeler', sample_mask_attribute)

        self.label_attribute = label_attribute

    def extract_labels(self, data, node_ids):
        ground_truths = data[self.label_attribute]

        labels = torch.full((data.num_nodes, ), -1)
        for node_id in node_ids:
            labels[node_id] = ground_truths[node_id]

        return labels
    
    def __call__(self, data):
        data = data.clone()

        label_mask = data[self.sample_mask_attribute]
        node_ids = torch.nonzero(label_mask).flatten().tolist()
        labels = self.extract_labels(data, node_ids)
        
        if not data.get('label_info'):
            data.label_info = [{} for _ in range(data.num_nodes)]

        for node_id in node_ids:
            data.label_info[node_id] = {'source': 'ground_truth', 'prob': 1.0}
        
        data.y = labels

        return data