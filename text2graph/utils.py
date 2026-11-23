import json
import logging

from pathlib import Path


logger = logging.getLogger(__name__)


__all__ = [
    'load_json',
    'save_json',
    'load_llm_responses_from_cache',
    'save_llm_responses_to_cache',
    'load_embeddings_from_cache',
    'save_embeddings_to_cache',
    'data_to_hetero',
]


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
        logger.info('Loading prompt responses from "%s".', cache_path)
        for key in keys:
            if key in cache:
                responses[key] = cache[key]
            else:
                unprocessed_keys.append(key)

        logger.info('Loaded %d/%d responses', len(responses), len(keys))
    else:
        logger.info("Cache path \"%s\" doesn't exist. Will process prompts", cache_path)
        unprocessed_keys = keys

    return responses, unprocessed_keys


def save_llm_responses_to_cache(cache_path, responses):
    current_cache = load_json(cache_path) or {}
    
    current_cache |= responses

    save_json(current_cache, cache_path)


def load_embeddings_from_cache(cache_path):
    import torch

    if not Path(cache_path).is_file():
        logger.info("Cache path \"%s\" doesn't exist. Will compute embeddings", cache_path)
        return None

    logger.info('Loading embeddings from "%s".', cache_path)

    return torch.load(cache_path, weights_only=True)


def save_embeddings_to_cache(cache_path, embeddings):
    import torch

    Path(cache_path).parent.mkdir(exist_ok=True, parents=True)

    torch.save(embeddings, cache_path)


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