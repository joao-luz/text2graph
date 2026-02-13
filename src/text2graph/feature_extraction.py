from .component import Component

import torch
import gc

from sentence_transformers import SentenceTransformer

class FeatureExtractor(Component):
    def __init__(self, name):
        super().__init__(name)

    def compute_representations(self, docs):
        pass

    def __call__(self):
        pass

class SentenceEmbedding(FeatureExtractor):
    def __init__(self, model=None, model_path=None):
        super().__init__('sentence_embedding')

        assert model or model_path, 'Either pass a model or a model path'
        
        if model:
            self.model = model
        else:
            self.model = SentenceTransformer(model_path)

        self.str_parameters = {
            'model': self.model
        }

    def compute_representations(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).cpu()
    
    def __call__(self, data):
        data = data.clone()

        data.x = self.compute_representations(data.text)

        del self.model
        torch.cuda.empty_cache()
        gc.collect()

        return data
