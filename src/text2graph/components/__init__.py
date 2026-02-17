from .component import Component
from .component_factory import register_component, get_component, component_from_config

from .edge_estimation import EmbeddingSimilarity
from .feature_extraction import *
from .labeling import *
from .propagating import *
from .sampling import *
from .visualizing import *


register_component('embedding_similarity', EmbeddingSimilarity)

register_component('sentence_embedding', SentenceEmbedding)

register_component('ground_truth_labeler', GroundTruthLabeler)
register_component('llm_labeler', LLMLabeler)
register_component('llm_ensemble_labeler', LLMEnsembleLabeler)

register_component('gcn_propagator', GCNPropagator)
register_component('lm_propagator', LMPropagator)

register_component('degree_sampler', DegreeSampler)
register_component('random_sampler', RandomSampler)
register_component('dma_sampler', DMASampler)

register_component('graph_visualizer', GraphVisualizer)
register_component('figure_visualizer', FigureVisualizer)
register_component('gml_visualizer', GMLVisualizer)
register_component('pickle_visualizer', PickleVisualizer)


__all__ = [
    'Component',
    'register_component',
    'get_component',
    'component_from_config'
]