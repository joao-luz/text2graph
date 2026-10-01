from ..component import Component
from ..component_registry import register_component
from ...utils import load_embeddings_from_cache, save_embeddings_to_cache

import torch
import gc

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data, HeteroData


@register_component('sentence_embedding_extractor')
class SentenceEmbeddingExtractor(Component):
    def __init__(
            self,
            model=None,
            model_path=None,
            graph_embedding_attribute='x',
            texts_attribute='text',
            prompt=None,
            node_types=['documents'],
            unload_model=True,
            max_seq_length=5000,
            cache_dir='cache',
            cache_file=None,
            load_from_cache=False
        ):
        super().__init__()

        assert model or model_path, 'Either pass a model or a model path'

        self.graph_embedding_attribute = graph_embedding_attribute
        self.texts_attribute = texts_attribute
        self.prompt = prompt
        self.node_types = node_types if isinstance(node_types, list) else [node_types]
        self.unload_model=unload_model
        self.max_seq_length = max_seq_length

        if model:
            self.model = model
            self.model_path = model.model_card_data.base_model
        else:
            self.model = None
            self.model_path = model_path

        self.sanitized_model_name = self.model_path.replace('/', '--')

        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.load_from_cache = load_from_cache

        if self.load_from_cache is None and self.cache_dir:
            print('load_from_cache is set to True but cache_dir is None. Won\'t load from cache')
            self.load_from_cache = False

        self.str_parameters = {
            'model_path': self.model_path,
            'graph_embedding_attribute': graph_embedding_attribute,
            'texts_attribute': texts_attribute,
            'node_types': self.node_types,
            'cache_dir': cache_dir,
            'load_from_cache': load_from_cache
        }

    def compute_representations(self, texts):
        return self.model.encode(
            texts, 
            convert_to_tensor=True, 
            prompt=self.prompt, 
            normalize_embeddings=True, 
            batch_size=8, 
            show_progress_bar=True
        ).to(torch.float).cpu().clone()

    def compute_cached_representations(self, node_type, texts):
        cache_path = f'{self.cache_dir}/{self.sanitized_model_name}/{self.cache_file or node_type}.pt'

        if self.load_from_cache:
            embeddings = load_embeddings_from_cache(cache_path)

            if embeddings is not None:
                if embeddings.shape[0] != len(texts):
                    raise ValueError(
                        f'Cached embeddings at "{cache_path}" have {embeddings.shape[0]} rows, '
                        f'but {len(texts)} texts were given.'
                    )

                return embeddings

        self.model = self.model or SentenceTransformer(self.model_path)

        # Never raise max_seq_length past what the model actually supports -- some models
        # (e.g. small BERT-based encoders) silently error out on sequences past their real
        # position embedding limit, even though max_seq_length itself just controls truncation.
        native_max_seq_length = self.model.max_seq_length
        self.model.max_seq_length = min(self.max_seq_length, native_max_seq_length)

        embeddings = self.compute_representations(texts)

        if self.cache_dir is not None:
            save_embeddings_to_cache(cache_path, embeddings)

        return embeddings

    def run(self, context):
        data = context['graph']

        for node_type in self.node_types:
            if node_type not in context['node_types']:
                raise ValueError(f'"{node_type}" not a valid type, only {context["node_types"]}')

            if isinstance(data, HeteroData):
                type_data = data[node_type]
            else:
                type_data = data

            type_data[self.graph_embedding_attribute] = self.compute_cached_representations(node_type, type_data[self.texts_attribute])

        if self.unload_model and self.model is not None:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()

            self.model = None

        return context