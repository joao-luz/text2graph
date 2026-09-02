import json

from pathlib import Path


def load_json(path):
    if not Path(path).is_file():
        return None

    def keystoint(pairs):
        return {int(k) if k.isdigit() else k: v for k, v in pairs}
    
    with open(path, 'r') as f:
        return json.load(f, object_pairs_hook=keystoint)


def save_json(data, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent='\t', ensure_ascii=False)


def load_llm_responses_from_cache(cache_path, keys):
    responses = {}
    unprocessed_keys = []

    cache = load_json(cache_path)

    if cache is not None:
        print(f'Loading prompt responses from "{cache_path}".')
        for key in keys:
            if key in cache:
                responses[key] = cache[key]
            else:
                unprocessed_keys.append(key)

        print(f'Loaded {len(responses)}/{len(keys)} responses')
    else:
        print(f'Cache path "{cache_path}" doesn\'t exist. Will process promts')
        unprocessed_keys = keys

    return responses, unprocessed_keys


def save_llm_responses_to_cache(cache_path, responses):
    current_cache = load_json(cache_path) or {}
    
    current_cache |= responses

    save_json(current_cache, cache_path)


def data_to_hetero(data, node_type='documents'):
    from torch_geometric.data import HeteroData

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