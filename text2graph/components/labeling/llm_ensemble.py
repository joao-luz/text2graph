from ..component import Component
from ..component_registry import register_component
from ...llm import LLM
from ...utils import load_llm_responses_from_cache, save_llm_responses_to_cache

import regex as re
import torch

from torch_geometric.data import Data, HeteroData


@register_component('llm_ensemble_labeler')
class LLMEnsembleLabeler(Component):
    def __init__(self,
        prompt_template,
        label_map,
        models=None,
        model_paths=None,
        mask_name='to_label',
        node_type='documents',
        label_attribute='pseudo_y',
        resolver='majority_vote',
        threshold=None,
        concatenate_decisions_to_x=None,
        decision_features_attribute=None,
        input_builder=None,
        response_parser=None,
        parser_args={},
        temperature=0.0,
        unload_model=True,
        cache_dir='cache',
        cache_file='labels',
        load_from_cache=False
    ):
        assert model_paths is not None or models is not None, 'Either pass a list of models or model paths'

        if models is not None:
            self.models = models
            self.model_paths = [model.model_name for model in models]

        else:
            self.model_paths = model_paths
            self.models = [LLM(path) for path in model_paths]
        
        def default_input_builder(context, node_type, node_id, cap=1200):
            data = context['graph'] if isinstance(context['graph'], Data) else context['graph'][node_type]
            text = data.text[node_id]
            index = sum(len(token) for token in text.split()[:cap]) + cap
            text = text[:index]
            return {'text': text, 'key': node_id}

        def default_parser(response, options):
            m = re.search(r'([0-9]+)', response)
            if m: option = int(m[1])
            else: option = 0

            if option >= len(options):
                option = 0

            return option

        self.threshold = threshold
        def majority_vote_resolver(decisions, threshold=threshold):
            votes = torch.stack(list(decisions.values()))
            majority_votes, _ = torch.mode(votes, dim=0)
            probs = torch.sum(votes == majority_votes, dim=0) / votes.shape[0]

            if threshold:
                majority_votes[probs < threshold] = -1

            return majority_votes, probs

        self.resolver = majority_vote_resolver if resolver == 'majority_vote' else resolver

        self.decision_features_attribute = decision_features_attribute
        self.concatenate_decisions_to_x = concatenate_decisions_to_x
        self.prompt_template = prompt_template
        self.label_map = label_map
        self.mask_name = mask_name
        self.node_type = node_type
        self.label_attribute = label_attribute
        self.input_builder = input_builder or default_input_builder
        self.response_parser = response_parser or default_parser
        self.parser_args = parser_args
        self.temperature = temperature
        self.unload_model = unload_model

        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.load_from_cache = load_from_cache

        if self.load_from_cache is None and self.cache_dir:
            print('load_from_cache is set to True but cache_dir is None. Won\'t load from cache')
            self.load_from_cache = False

        self.str_parameters = {
            'models': self.model_paths,
            'mask_name': mask_name,
            'node_type': node_type,
            'label_attribute': label_attribute,
            'resolver': resolver,
            'threshold': threshold,
            'temperature': temperature
        }

    def set_prompt_template(self, new_template):
        self.prompt_template = new_template

    def set_parser_args(self, new_args):
        self.parser_args = new_args

    def extract_labels(self, context, type_data, node_ids):        
            inputs = [self.input_builder(context, self.node_type, node_id) for node_id in node_ids]
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
    
            parsed = [self.response_parser(response, **self.parser_args) for response in responses.values()]
    
            labels = torch.full((type_data.num_nodes, ), -1)
            for i,node_id in enumerate(node_ids):
                labels[node_id] = parsed[i]
    
            return labels

    def extract_labels(self, context, type_data, node_ids):
        inputs = [self.input_builder(context, self.node_type, node_id) for node_id in node_ids]

        decisions = {}
        for model,model_path in zip(self.models, self.model_paths):
            unprocessed_keys = [input['key'] for input in inputs]
            responses = []

            cache_path = f'{self.cache_dir}/{model.sanitized_model_name}/{self.cache_file}.json'

            if self.load_from_cache:
                cache_keys = [input['key'] for input in inputs]
                responses, unprocessed_keys = load_llm_responses_from_cache(cache_path, cache_keys)

            prompts = [self.prompt_template.format(**input) for input in inputs if input['key'] in unprocessed_keys]

            print(f'Voting with {model_path}...')
            responses_list = model.invoke(prompts, self.temperature)
            responses |= {key: response for key,response in zip(unprocessed_keys, responses_list)}

            if self.cache_dir is not None:
                save_llm_responses_to_cache(cache_path, responses)

            parsed = [self.response_parser(response, **self.parser_args) for response in responses.values()]

            decisions[model_path] = torch.tensor(parsed)

            if self.unload_model:
                model.unload_model()

        preds, probs = self.resolver(decisions)

        preds_full = torch.full((type_data.num_nodes, ), -1)
        probs_full = torch.zeros((type_data.num_nodes, ))
        for i, node_id in enumerate(node_ids):
            preds_full[node_id] = preds[i]
            probs_full[node_id] = probs[i]

        if not torch.any(preds_full != -1):
            raise ValueError(f'No nodes labeled with at least {self.threshold:.2f} agreement between ensemble models.')

        decisions_full = {}
        for model_path in self.model_paths:
            full = torch.full((type_data.num_nodes, ), -1)
            for i, node_id in enumerate(node_ids):
                full[node_id] = decisions[model_path][i]
            decisions_full[model_path] = full

        return preds_full, probs_full, decisions_full

    def run(self, context):
        data = context['graph']

        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')

        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        self.parser_args['options'] = self.label_map

        label_mask = context[self.mask_name]
        node_ids = torch.nonzero(label_mask).flatten().tolist()
        preds, probs, decisions = self.extract_labels(context, type_data, node_ids)

        if type_data.get(self.label_attribute) is None:
            type_data[self.label_attribute] = torch.full((type_data.num_nodes,), -1)

        type_data[self.label_attribute][label_mask] = preds[label_mask]

        if not type_data.get('label_info'):
            context['label_info'] = [{} for _ in range(type_data.num_nodes)]

        for node_id in node_ids:
            context['label_info'][node_id] = {
                'source': 'llm_ensemble (' + ', '.join(sorted(self.model_paths)) + ')' + (f' @ {self.threshold:.2f}' if self.threshold else ''),
                'decisions': {model_path: decisions[model_path][node_id].item() for model_path in self.model_paths},
                'prob': probs[node_id].item()
            }

        context['graph'] = data

        return context