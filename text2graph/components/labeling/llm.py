from ..component import Component
from ..component_registry import register_component
from ...llm import LLM

import regex as re
import torch

from torch_geometric.data import Data, HeteroData


@register_component('llm_labeler')
class LLMLabeler(Component):
    def __init__(self,
        prompt_template,
        label_map,
        model=None,
        model_path=None,
        mask_name='to_label',
        node_type='documents',
        label_attribute='pseudo_y',
        input_builder=None,
        response_parser=None, 
        parser_args={}, 
        temperature=0.0,
        unload_model=True
    ):
        assert model or model_path, 'Either pass a model or a model path'

        if model:
            self.model = model
        else:
            self.model = LLM(model_path)

        def default_input_builder(context, node_type, node_id, cap=1200):
            data = context['graph'] if isinstance(context['graph'], Data) else context['graph'][node_type]
            text = data.text[node_id]
            index = sum(len(token) for token in text.split()[:cap]) + cap
            text = text[:index]
            return {'text': text}

        def default_parser(response, options):
            m = re.search(r'([0-9]+)', response)
            if m: option = int(m[1])
            else: option = 0

            if option >= len(options):
                option = 0

            return option
        
        self.input_builder = input_builder or default_input_builder
        self.response_parser = response_parser or default_parser
        self.parser_args = parser_args

        self.prompt_template = prompt_template
        self.label_map = label_map
        self.mask_name = mask_name
        self.node_type = node_type
        self.label_attribute = label_attribute
        self.temperature = temperature
        self.unload_model = unload_model

        self.str_parameters = {
            'model': self.model.model_name,
            'mask_name': mask_name,
            'node_type': node_type,
            'label_attribute': label_attribute,
            'temperature': temperature
        }

    def set_prompt_template(self, new_template):
        self.prompt_template = new_template

    def set_parser_args(self, new_args):
        self.parser_args = new_args

    def extract_labels(self, context, type_data, node_ids):        
        inputs = [self.input_builder(context, self.node_type, node_id) for node_id in node_ids]
        prompts = [self.prompt_template.format(**input) for input in inputs]
        outputs = self.model.invoke(prompts, self.temperature)
        parsed = [self.response_parser(output, **self.parser_args) for output in outputs]

        labels = torch.full((type_data.num_nodes, ), -1)
        for i,node_id in enumerate(node_ids):
            labels[node_id] = parsed[i]

        return labels
    
    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')
        
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        self.parser_args['options'] = self.label_map

        label_mask = context[self.mask_name]
        node_ids = torch.nonzero(label_mask).flatten().tolist()
        labels = self.extract_labels(context, type_data, node_ids)

        type_data[self.label_attribute] = labels

        if not type_data.get('label_info'):
            context['label_info'] = [{} for _ in range(type_data.num_nodes)]

        for node_id in node_ids:
            context['label_info'][node_id] = {'source': self.model.model_name, 'prob': 1.0}

        if self.unload_model:
            self.model.unload_model()

        return context
