from ..component import Component
from ..component_registry import register_component
from ...llm import LLM
from ...utils import data_to_hetero, load_llm_responses_from_cache, save_llm_responses_to_cache

import regex as re
import torch

from torch_geometric.data import Data, HeteroData


@register_component('llm_node_generator')
class LLMNodeGenerator(Component):
    def __init__(self,
        prompt_template,
        node_type,
        num_generations=1,
        model=None,
        model_path=None,
        text_attribute='text',
        label_attribute='pseudo_y',
        generated_mask_name='is_generated',
        inputs_builder=None,
        response_parser=None,
        temperature=0.7,
        unload_model=True,
        true_labels=None,
        cache_dir='cache',
        cache_file=None,
        load_from_cache=False
    ):
        assert model or model_path, 'Either pass a model or a model path'

        if model:
            self.model = model
        else:
            self.model = LLM(model_path)

        def build_inputs(context):
            inputs = []
            for class_label in context['classes'].values():
                inputs.append({'class_label': class_label, 'key': class_label})

            return inputs

        def single_response_parser(response):
            return response

        def multi_response_parser(response):
            pattern = r'\n?\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|\Z)'
            matches = list(re.finditer(pattern, response))

            if not matches:
                return []

            generated_texts = []
            for match in matches:
                text = match.group(2).strip()
                generated_texts.append(text)

            return generated_texts

        self.inputs_builder = inputs_builder or build_inputs

        if response_parser == 'single' or response_parser is None:
            self.response_parser = single_response_parser
        elif response_parser == 'multi':
            self.response_parser = multi_response_parser
        else:
            self.response_parser = response_parser

        self.prompt_template = prompt_template
        self.node_type = node_type
        self.num_generations = num_generations
        self.text_attribute = text_attribute
        self.label_attribute = label_attribute
        self.generated_mask_name = generated_mask_name
        self.temperature = temperature
        self.unload_model = unload_model
        self.true_labels = true_labels

        self.cache_dir = cache_dir
        self.load_from_cache = load_from_cache
        self.cache_file = cache_file or node_type

        if self.load_from_cache is None and self.cache_dir:
            print('load_from_cache is set to True but cache_dir is None. Won\'t load from cache')
            self.load_from_cache = False

        self.str_parameters = {
            'model': self.model.model_name,
            'node_type': node_type,
            'generated_mask_name': generated_mask_name,
            'temperature': temperature
        }

    def generate_nodes(self, context):
        inputs = self.inputs_builder(context)
        unprocessed_keys = [input['key'] for input in inputs]
        responses = []

        cache_path = f'{self.cache_dir}/{self.model.sanitized_model_name}/{self.cache_file}.json'

        if self.load_from_cache:
            cache_keys = [input['key'] for input in inputs]
            responses, unprocessed_keys = load_llm_responses_from_cache(cache_path, cache_keys)

        prompts = [self.prompt_template.format(**input) for input in inputs if input['key'] in unprocessed_keys]
        responses_list = self.model.invoke(prompts, self.temperature)
        responses |= {key: response for key,response in zip(unprocessed_keys, responses_list)}

        if self.cache_dir is not None:
            save_llm_responses_to_cache(cache_path, responses)

        texts = []
        labels = []
        for input,response in zip(inputs, responses.values()):
            parsed = self.response_parser(response)
            if isinstance(parsed, str):
                parsed = [parsed]

            label = next((k for k, v in context['classes'].items() if v == input['class_label']), 0)
            labels += [label] * len(parsed)
            texts += parsed

        return texts, labels

    def run(self, context):
        data = context['graph']

        is_new_layer = self.node_type not in context['node_types']

        if is_new_layer and not isinstance(data, HeteroData):
            homogeneous_node_type = context['node_types'][0]

            data = data_to_hetero(data, homogeneous_node_type)
            context['node_types'] = [homogeneous_node_type]

        generated_texts, generated_labels = self.generate_nodes(context)
        n_generated = len(generated_texts)

        if is_new_layer:
            target_data = data[self.node_type]
            target_data.num_nodes = n_generated
            target_data[self.text_attribute] = generated_texts
            target_data[self.label_attribute] = torch.tensor(generated_labels)
            target_data[self.generated_mask_name] = torch.ones(n_generated, dtype=torch.bool)
            offset = 0
            context['node_types'].append(self.node_type)
        else:
            target_data = data[self.node_type] if isinstance(data, HeteroData) else data
            offset = len(target_data[self.text_attribute])

            existing_text = list(target_data[self.text_attribute]) if target_data.get(self.text_attribute) is not None else [None] * offset
            target_data[self.text_attribute] = existing_text + generated_texts

            existing_labels = target_data[self.label_attribute] if target_data.get(self.label_attribute) is not None else torch.full((offset,), -1)
            target_data[self.label_attribute] = torch.concat((existing_labels, torch.tensor(generated_labels)))

            if self.true_labels is not None:
                context[self.true_labels] = torch.concat((context[self.true_labels], torch.tensor(generated_labels)))

            existing_mask = target_data.get(self.generated_mask_name)
            if existing_mask is None:
                existing_mask = torch.zeros(offset, dtype=torch.bool)
            target_data[self.generated_mask_name] = torch.cat([existing_mask, torch.ones(n_generated, dtype=torch.bool)])

            target_data.num_nodes = offset + n_generated

        context['graph'] = data

        if self.unload_model:
            self.model.unload_model()

        print(data)

        return context