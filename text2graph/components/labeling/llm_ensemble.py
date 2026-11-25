import logging

from ..component import Component
from ..component_registry import register_component
from ...llm import load_llm
from ...utils import load_llm_responses_from_cache, save_llm_responses_to_cache

import regex as re
import torch

from torch_geometric.data import Data, HeteroData


logger = logging.getLogger(__name__)


@register_component('llm_ensemble_labeler')
class LLMEnsembleLabeler(Component):
    """Labels the sampled nodes by asking several LLMs and combining their votes.

    Each model labels every sampled node like `LLMLabeler` does (same prompts,
    parsing and caching, with one cache file per model), one model at a time.
    `resolver` then combines the votes into one label per node.

    Only the sampled nodes' labels change; other nodes keep theirs. Each sampled
    node gets a `label_info` entry with `source` (the ensemble's models and
    threshold), `decisions` (each model's vote, by model name) and `prob` (from
    the resolver; for majority vote, the share of models that agreed).

    Args:
        prompt_template: See `LLMLabeler`.
        label_map: See `LLMLabeler`.
        models: List of `LLM` instances.
        model_paths: Used instead of `models`: list of anything `load_llm` accepts.
            Pass either `models` or `model_paths`.
        mask_name: See `LLMLabeler`.
        node_type: See `LLMLabeler`.
        label_attribute: Node attribute the labels are written to (created, all
            -1, if missing).
        resolver: How the votes are combined:

            - `'majority_vote'` (default): the most common vote (ties go to the
              smallest class id), with the share of models that cast it as `prob`.
            - A function `resolver(decisions) -> (preds, probs)`, where `decisions`
              maps each model name to a tensor of its votes on the sampled nodes,
              and `preds` uses -1 for nodes left unlabeled.

        threshold: For `'majority_vote'`: nodes whose majority has a smaller share
            of the votes are left unlabeled (-1). `None` labels every node.
        concatenate_decisions_to_x: Not used yet.
        decision_features_attribute: Not used yet.
        input_builder: See `LLMLabeler`.
        response_parser: See `LLMLabeler`.
        parser_args: See `LLMLabeler`.
        temperature: See `LLMLabeler`.
        unload_model: Unload each model right after it has voted, so only one is in
            GPU memory at a time.
        cache_dir: See `LLMLabeler`; each model has its own subdirectory.
        cache_file: See `LLMLabeler`.
        load_from_cache: See `LLMLabeler`.

    Raises:
        ValueError: If neither `models` nor `model_paths` is given; when running, if
            `node_type` isn't in the graph, or no node is labeled (e.g. because
            no majority reaches `threshold`).
    """

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
        if models is None and model_paths is None:
            raise ValueError('Either pass a list of models or model paths')

        if models is not None:
            self.models = models
            self.model_paths = [model.model_name for model in models]

        else:
            self.models = [load_llm(path) for path in model_paths]
            self.model_paths = [model.model_name for model in self.models]
        
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

        if self.load_from_cache and self.cache_dir is None:
            logger.warning("load_from_cache is set to True but cache_dir is None. Won't load from cache")
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
        """Replace the prompt template used for the next runs."""
        self.prompt_template = new_template

    def set_parser_args(self, new_args):
        """Replace the extra keyword arguments passed to `response_parser`."""
        self.parser_args = new_args

    def _extract_labels(self, context, type_data, node_ids):
        inputs = [self.input_builder(context, self.node_type, node_id) for node_id in node_ids]

        decisions = {}
        for model,model_path in zip(self.models, self.model_paths):
            responses = {}

            cache_path = f'{self.cache_dir}/{model.sanitized_model_name}/{self.cache_file}.json'

            if self.load_from_cache:
                responses, _ = load_llm_responses_from_cache(cache_path, [input['key'] for input in inputs])

            logger.info('Voting with %s...', model_path)
            pending = [input for input in inputs if input['key'] not in responses]
            if pending:
                prompts = [self.prompt_template.format(**input) for input in pending]
                responses_list = model.invoke(prompts, self.temperature)
                responses |= {input['key']: response for input,response in zip(pending, responses_list)}

            if self.cache_dir is not None:
                save_llm_responses_to_cache(cache_path, responses)

            parsed = [self.response_parser(responses[input['key']], **self.parser_args) for input in inputs]

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
            if self.threshold:
                raise ValueError(f'No nodes labeled with at least {self.threshold:.2f} agreement between ensemble models.')
            raise ValueError(f'The resolver left all {len(node_ids)} sampled nodes unlabeled (-1).')

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
        preds, probs, decisions = self._extract_labels(context, type_data, node_ids)

        if type_data.get(self.label_attribute) is None:
            type_data[self.label_attribute] = torch.full((type_data.num_nodes,), -1)

        type_data[self.label_attribute][label_mask] = preds[label_mask]

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(type_data.num_nodes)]

        for node_id in node_ids:
            context['label_info'][node_id] = {
                'source': 'llm_ensemble (' + ', '.join(sorted(self.model_paths)) + ')' + (f' @ {self.threshold:.2f}' if self.threshold else ''),
                'decisions': {model_path: decisions[model_path][node_id].item() for model_path in self.model_paths},
                'prob': probs[node_id].item()
            }

        context['graph'] = data

        return context