import torch

from torch_geometric.data import Data

from .feature_extraction import SentenceEmbedding
from .edge_estimation import EmbeddingSimilarity
from .sampling import NodeSampler, RandomSampler, DegreeSampler
from .labeling import LLMLabeler, LLMEnsembleLabeler, GroundTruthLabeler
from .propagating import GCNPropagator, LMPropagator
from .visualize import GraphVisualizer, FigureVisualizer, GMLVisualizer, PickleVisualizer

class Text2Graph:
    component_map = {
        'sentence_embedding': SentenceEmbedding,
        'embedding_similarity': EmbeddingSimilarity,

        'random_sampler': RandomSampler,
        'degree_sampler': DegreeSampler,

        'llm_labeler': LLMLabeler,
        'llm_ensemble_labeler': LLMEnsembleLabeler,
        'ground_truth_labeler': GroundTruthLabeler,

        'gcn_propagator': GCNPropagator,
        'lm_propagator': LMPropagator,

        'figure_visualizer': FigureVisualizer,
        'gml_visualizer': GMLVisualizer,
        'pickle_visualizer': PickleVisualizer,
    }

    def _replace_sampler(self, parameters):
        for key,values in parameters.items():
            if key == 'node_sampler' and not isinstance(values, NodeSampler):
                name = parameters[key]['name']
                sampler = Text2Graph.component_map[name](**values['parameters'])
                parameters[key] = sampler

    def _load_steps_from_config(self, config):
        steps = []
        for component_config in config['pipeline']['components']:
            name = component_config['name']
            parameters = component_config.get('parameters') or {}

            self._replace_sampler(parameters)

            component = Text2Graph.component_map[name](**parameters)
            steps.append(component)

        return steps
    
    def __init__(self, steps=None, config=None, name='', skip_visualization=False, output_dir='.'):
        assert steps or config, 'Either pass the pipeline steps or a config dict'

        self.name = name
        self.skip_visualization = skip_visualization

        if steps:
            self.steps = steps
        else:
            self.steps = self._load_steps_from_config(config)

        self.output_dir = output_dir
    
    def _build_data(self, texts, true_labels=None, id2label=None):
        label2id = {v: k for k,v in id2label.items()}
        data = Data(text=texts, true_label=torch.tensor(true_labels), label2id=label2id, id2label=id2label, num_nodes=len(texts))

        return data
    
    def __str__(self):
        return 'Text2Graph(steps=[\n\t' + ',\n\t'.join([str(step) for step in self.steps]) + '\n])'
    
    def __call__(self, texts, true_labels=None, id2label=None):
        data = self._build_data(texts, true_labels, id2label)

        for step in self.steps:
            if isinstance(step, GraphVisualizer):
                if self.skip_visualization:
                    continue
                
                step.set_output_dir(self.output_dir)

            string = str(step)

            print(f'Running {string}...')
            data = step(data)

        return data