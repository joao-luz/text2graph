from .component import Component
from .llm import LLM
from .sampling import RandomSampler, DegreeSampler

import regex as re

node_sampler_map = {
    'random_sampler': RandomSampler,
    'degree_sampler': DegreeSampler
}

class Labeler(Component):
    def __init__(self, name, node_sampler):
        super().__init__(name)

        self.node_sampler = node_sampler

    def label(self, inputs):
        pass

    def __call__(self):
        pass

class GroundTruthLabeler(Labeler):
    def __init__(self, node_sampler, label_attribute='ground_truth'):
        super().__init__('ground_truth_labeler', node_sampler)

        self.label_attribute = label_attribute

    def extract_labels(self, nodes):
        return [node[self.label_attribute] for node in nodes]
    
    def __call__(self, G):
        nodes = self.node_sampler.sample_nodes(G)
        labels = self.extract_labels([G.nodes[n] for n in nodes])

        for n,label in zip(nodes, labels):
            G.nodes[n]['class'] = {'label': label, 'source': 'ground_truth', 'prob': 1.0}

        return G

class LLMLabeler(Labeler):
    def __init__(self, prompt_template, node_sampler, input_builder=None, response_parser=None, model=None, model_path=None, parser_args={}, temperature=0.0):
        super().__init__(f'llm_labeler', node_sampler=node_sampler)

        assert model or model_path, 'Either pass a model or a model path'

        if model:
            self.model = model
        else:
            self.model = LLM(model_path)

        def default_input_builder(node):
            index = sum(len(token) for token in node['text'].split()[:1200]) + 1200
            text = node['text'][:index]
            return {'text': text}

        def default_parser(response, options):
            m = re.search(r'([0-9]+)', response)
            if m: option = int(m[1])
            else: option = 0

            return options[option]

        self.prompt_template = prompt_template
        self.node_sampler = node_sampler
        self.input_builder = input_builder or default_input_builder
        self.response_parser = response_parser or default_parser
        self.parser_args = parser_args
        self.temperature = temperature

        if isinstance(node_sampler, str):
            self.node_sampler = node_sampler_map[node_sampler]
        else:
            self.node_sampler = node_sampler

        self.str_parameters = {
            'model': self.model.name,
            'sampler': self.node_sampler
        }

    def set_prompt_template(self, new_template):
        self.prompt_template = new_template

    def set_parser_args(self, new_args):
        self.parser_args = new_args

    def extract_labels(self, nodes):
        inputs = [self.input_builder(node) for node in nodes]
        prompts = [self.prompt_template.format(**input) for input in inputs]
        outputs = self.model.query(prompts, self.temperature)
        parsed = [self.response_parser(output, **self.parser_args) for output in outputs]

        return parsed
    
    def __call__(self, G):
        nodes = self.node_sampler.sample_nodes(G)
        labels = self.extract_labels([G.nodes[n] for n in nodes])

        for n,label in zip(nodes, labels):
            G.nodes[n]['class'] = {'label': label, 'source': self.model.name, 'prob': 1.0}

        return G