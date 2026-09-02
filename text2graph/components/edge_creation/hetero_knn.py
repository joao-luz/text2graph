from ..component import Component
from ..component_registry import register_component

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, HeteroData


@register_component('hetero_knn_edge_creator')
class HeteroKNNEdgeCreator(Component):
    def __init__(self, k, source_node_type, destine_node_type, link_name='similar', representation_attr='x', rev_link=True):
        self.k = k
        self.node_type_pair = (source_node_type, destine_node_type)
        self.link_name = link_name
        self.representation_attr = representation_attr
        self.rev_link = rev_link

        self.str_parameters = {
            'k': k,
            'node_type_pair': (source_node_type, destine_node_type),
            'link_name': link_name,
            'representation_attr': representation_attr,
            'rev_link': rev_link
        }

    def compute_knn(self, x_src, x_dst, same_type):
        n_src = x_src.size(0)
        n_dst = x_dst.size(0)

        k = min(self.k, n_dst if not same_type else n_dst - 1)
        if k <= 0:
            raise ValueError(
                f"Not enough destination nodes to form {self.k} neighbors "
                f"(available: {n_dst})."
            )

        # Cosine similarity between every src node and every dst node
        src_norm = F.normalize(x_src, p=2, dim=-1)
        dst_norm = F.normalize(x_dst, p=2, dim=-1)
        sim = src_norm @ dst_norm.t()  # [n_src, n_dst]

        if same_type:
            # avoid connecting a node to itself
            sim.fill_diagonal_(float('-inf'))

        # top-k most similar dst nodes for each src node
        topk_sim, topk_idx = sim.topk(k, dim=-1, largest=True)  # [n_src, k]

        src_index = torch.arange(n_src, device=x_src.device).view(-1, 1).expand(-1, k).reshape(-1)
        dst_index = topk_idx.reshape(-1)

        edge_index = torch.stack([src_index, dst_index], dim=0)

        return edge_index, topk_sim.reshape(-1, 1)

    def run(self, context):
        data = context['graph']

        src_type, dst_type = self.node_type_pair

        if src_type not in context['node_types']:
            raise ValueError(f'"{src_type}" not a valid type, only {context["node_types"]}')

        if dst_type not in context['node_types']:
            raise ValueError(f'"{dst_type}" not a valid type, only {context["node_types"]}')
        
        if isinstance(data, HeteroData):
            x_src = getattr(data[src_type], self.representation_attr)
            x_dst = getattr(data[dst_type], self.representation_attr)

            edge_index, edge_attr = self.compute_knn(x_src, x_dst, same_type=(src_type == dst_type))

            data[src_type, self.link_name, dst_type].edge_index = edge_index
            data[src_type, self.link_name, dst_type].edge_attr = edge_attr

            if self.rev_link:
                data[dst_type, self.link_name, src_type].edge_index = edge_index.flip(0)
                data[dst_type, self.link_name, src_type].edge_attr = edge_attr

        else:
            # homogeneous graph: single implicit node type, just add to edge_index
            x_src = getattr(data, self.representation_attr)
            x_dst = x_src

            edge_index, edge_attr = self.compute_knn(x_src, x_dst, same_type=True)

            if self.rev_link:
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

            existing_edge_index = data.edge_index if data.get('edge_index') is not None else torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
            data.edge_index = torch.cat([existing_edge_index, edge_index], dim=1)

            if data.get('edge_attr') is not None:
                existing_edge_attr = data.edge_attr
                # pad existing edge_attr if it has a different width than the new similarity column
                if existing_edge_attr.dim() == 1:
                    existing_edge_attr = existing_edge_attr.view(-1, 1)
                if existing_edge_attr.size(1) != edge_attr.size(1):
                    raise ValueError(
                        f'edge_attr width mismatch: existing edges have width {existing_edge_attr.size(1)}, '
                        f'new similarity edges have width {edge_attr.size(1)}.'
                    )
                data.edge_attr = torch.cat([existing_edge_attr, edge_attr], dim=0)
            else:
                data.edge_attr = edge_attr

        context['graph'] = data

        return context