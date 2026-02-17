from . import components
from .components.visualizing import GraphVisualizer

import torch

from torch_geometric.data import Data


def load_steps_from_config(config):
    steps = []
    for component_config in config['pipeline']['components']:
        component = components.component_from_config(component_config)
        steps.append(component)

    return steps


class Text2Graph:
    def __init__(self, steps=None, config=None, name='', skip_visualization=False, output_dir='.'):
        assert steps or config, 'Either pass the pipeline steps or a config dict'

        self.name = name
        self.skip_visualization = skip_visualization

        if steps:
            self.steps = steps
        else:
            self.steps = load_steps_from_config(config)

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