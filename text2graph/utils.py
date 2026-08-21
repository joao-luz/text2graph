from torch_geometric.data import HeteroData


def data_to_hetero(data, node_type='documents'):
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