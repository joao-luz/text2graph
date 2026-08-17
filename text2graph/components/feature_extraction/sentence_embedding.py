from ..component import Component
from ..component_registry import register_component

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
            texts_attribute='documents', 
            prompt=None,
            node_types=['documents'],
            unload_model=True
        ):
        super().__init__()

        assert model or model_path, 'Either pass a model or a model path'

        self.graph_embedding_attribute = graph_embedding_attribute
        self.texts_attribute = texts_attribute
        self.prompt = prompt
        self.node_types = node_types if isinstance(node_types, list) else [node_types]
        self.unload_model=unload_model
        
        if model:
            self.model = model
            self.model_path = model.model_card_data.base_model
        else:
            self.model = None
            self.model_path = model_path

        self.str_parameters = {
            'model_path': self.model_path,
            'graph_embedding_attribute': graph_embedding_attribute,
            'texts_attribute': texts_attribute,
            'node_types': self.node_types
        }

    def compute_representations(self, texts):
        return self.model.encode(texts, convert_to_tensor=True, prompt=self.prompt).cpu()
    
    def run(self, context):
        data = context['graph']
        
        self.model = self.model or SentenceTransformer(self.model_path)

        for node_type in self.node_types:
            if node_type not in context['node_types']:
                raise ValueError(f'"{node_type}" not a valid type, only {context["node_types"]}')
            
            if isinstance(data, HeteroData):
                type_data = data[node_type]
            else:
                type_data = data

            type_data[self.graph_embedding_attribute] = self.compute_representations(context[self.texts_attribute])

        if self.unload_model:
            del self.model
            torch.cuda.empty_cache()
            gc.collect()

            self.model = None

        return context