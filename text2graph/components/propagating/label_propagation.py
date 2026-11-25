import logging

from ..component import Component
from ..component_registry import register_component
from ...utils import get_same_type_edges

import torch

from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


@register_component('label_propagator')
class LabelPropagator(Component):
    """Classic Label Propagation (Zhu & Ghahramani, 2002): iteratively spreads
    label distributions from labeled to unlabeled nodes along graph edges,
    re-clamping the labeled nodes' distributions back to their true one-hot
    label after every iteration, until convergence or `max_iter` is reached.

    Set `alpha` (in (0, 1)) to use the softer Label Spreading variant instead
    (Zhou et al., 2004), which blends propagated and original labels at every
    step rather than hard-clamping labeled nodes back exactly.

    Each unlabeled node gets its most likely label, with that probability in its
    `label_info` entry (`{'source': 'label_propagator', 'prob': ...}`). A node no
    label reaches, e.g. one without outgoing edges, stays unlabeled (-1) and gets
    no `label_info` entry.

    Args:
        node_type: Node type to label.
        label_attribute: Node attribute holding the labels (-1 for unlabeled);
            predictions are written into it.
        edge_link_name: Relation name of the edges to use, on `HeteroData` graphs.
        edge_attr_attribute: Edge attribute holding edge weights; edges without it
            have weight 1.
        max_iter: Maximum number of propagation steps. A warning is logged if the
            labels haven't converged by then.
        tol: Convergence threshold: stop once no label probability changes by more
            than this in a step.
        alpha: `None` for Label Propagation; a value in (0, 1) for Label Spreading,
            the weight of the propagated labels against the original ones.
        propagating_mask: Optional boolean mask over the nodes; only unlabeled nodes
            inside it receive a predicted label (None means all of them).

    Raises:
        ValueError: If `node_type` isn't in the graph, or (on `HeteroData`) has no
            edges between its own nodes.
    """

    def __init__(
        self,
        node_type='documents',
        label_attribute='pseudo_y',
        edge_link_name='similar',
        edge_attr_attribute='edge_attr',
        max_iter=1000,
        tol=1e-6,
        alpha=None,
        propagating_mask=None
    ):
        self.node_type = node_type
        self.label_attribute = label_attribute
        self.edge_link_name = edge_link_name
        self.edge_attr_attribute = edge_attr_attribute
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.propagating_mask = propagating_mask

        self.str_parameters = {
            'node_type': node_type,
            'label_attribute': label_attribute,
            'max_iter': max_iter,
            'tol': tol,
            'alpha': alpha,
        }

    def _get_edge_index_and_weight(self, data, node_type):
        edge_store = get_same_type_edges(data, node_type, self.edge_link_name)

        edge_index = edge_store.edge_index
        edge_weight = edge_store.get(self.edge_attr_attribute)

        if edge_weight is not None:
            edge_weight = edge_weight.reshape(-1).clamp(min=0)
        else:
            edge_weight = torch.ones(edge_index.size(1))

        return edge_index, edge_weight

    def _propagate(self, data, type_data, train_mask):
        num_nodes = type_data.num_nodes

        edge_index, edge_weight = self._get_edge_index_and_weight(data, self.node_type)
        src, dst = edge_index[0], edge_index[1]

        unique_labels, new_labels = torch.unique(type_data[self.label_attribute][train_mask], sorted=True, return_inverse=True)
        num_classes = len(unique_labels)

        labeled_ids = torch.nonzero(train_mask).flatten()

        Y0 = torch.zeros(num_nodes, num_classes)
        Y0[labeled_ids, new_labels] = 1.0

        deg = torch.zeros(num_nodes).scatter_add_(0, src, edge_weight).clamp(min=1e-12)

        Y = Y0.clone()
        diff = float('inf')
        for it in range(self.max_iter):
            messages = edge_weight.unsqueeze(-1) * Y[dst]
            Y_new = torch.zeros_like(Y)
            Y_new.scatter_add_(0, src.unsqueeze(-1).expand(-1, num_classes), messages)
            Y_new = Y_new / deg.unsqueeze(-1)

            if self.alpha is not None:
                Y_new = self.alpha * Y_new + (1 - self.alpha) * Y0
            else:
                Y_new[labeled_ids] = Y0[labeled_ids]

            diff = (Y_new - Y).abs().max().item()
            Y = Y_new

            if diff < self.tol:
                logger.info('label_propagator converged after %d iterations (max change %.2e)', it + 1, diff)
                break
        else:
            logger.warning('label_propagator did not converge within %d iterations (max change %.2e)', self.max_iter, diff)

        row_sum = Y.sum(dim=1)
        reached = row_sum > 0
        probs, preds = (Y / row_sum.clamp(min=1e-12).unsqueeze(-1)).max(dim=1)
        preds = unique_labels[preds]
        preds[~reached] = -1
        probs[~reached] = 0.0

        return preds, probs

    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')

        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        train_mask = type_data[self.label_attribute] != -1

        preds, probs = self._propagate(data, type_data, train_mask)

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(type_data.num_nodes)]

        propagating_mask = self.propagating_mask if self.propagating_mask is not None else torch.ones(type_data.num_nodes, dtype=torch.bool)
        propagating_mask = propagating_mask & ~train_mask & (preds != -1)
        for node_id in torch.nonzero(propagating_mask).flatten().tolist():
            context['label_info'][node_id] = {'source': 'label_propagator', 'prob': probs[node_id].item()}
            type_data[self.label_attribute][node_id] = preds[node_id]

        return context
