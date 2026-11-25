import logging

from ..component import Component
from ..component_registry import register_component
from ...llm import load_llm
from ...utils import load_llm_responses_from_cache, save_llm_responses_to_cache

import regex as re
import torch

from torch_geometric.data import Data, HeteroData


logger = logging.getLogger(__name__)


@register_component('llm_labeler')
class LLMLabeler(Component):
    """Labels the sampled nodes by asking an LLM to classify each one.

    For each node in the sample mask, `input_builder` makes an input dict, the
    prompt is `prompt_template.format(**input)`, and `response_parser` turns the
    LLM's response into a class id.

    Only the sampled nodes' labels change; other nodes keep theirs, e.g. labels
    from earlier steps. Each sampled node gets a `label_info` entry
    `{'source': model name, 'prob': 1.0}`.

    Args:
        prompt_template: Prompt with `str.format` fields filled from each input dict
            (by default, `{text}`). Literal braces must be doubled: `{{` and `}}`.
        label_map: Mapping of class ids to class names. Passed to
            `response_parser` as `options`.
        model: `LLM` instance to label with.
        model_path: Used instead of `model`: anything `load_llm` accepts, e.g. a
            model name. Pass either `model` or `model_path`.
        mask_name: Context key of the boolean mask of nodes to label, e.g. set by
            a sampler.
        node_type: Node type to label.
        label_attribute: Node attribute the labels are written to (created, all
            -1, if missing).
        input_builder: Function `input_builder(context, node_type, node_id) -> dict`.
            The dict fills `prompt_template` and needs a `key`, unique per node, used
            as its cache key. The default gives `{'text': ..., 'key': node_id}`, with
            the node's text cut to about its first 1200 words.
        response_parser: Function `response_parser(response, options, **parser_args)`
            returning a class id. The default takes the first integer in the
            response, and falls back to 0 when there is none or it's not below
            `len(options)`; a response it can't parse silently becomes class 0.
        parser_args: Extra keyword arguments for `response_parser`.
        temperature: Sampling temperature passed to `LLM.invoke`.
        unload_model: Unload the model after this step, to free GPU memory for the
            next one.
        cache_dir: Directory for cached responses, saved to
            `{cache_dir}/{model.sanitized_model_name}/{cache_file}.json`, keyed by
            each input's `key`, whenever this is set. `None` disables caching.
            With the default keys (node ids), use a separate directory per dataset.
        cache_file: Cache file name, without `.json`.
        load_from_cache: Reuse cached responses for nodes whose `key` is in the
            cache, and only prompt the LLM for the rest. A cached response is
            reused even if the prompt has changed since it was saved.

    Raises:
        ValueError: If neither `model` nor `model_path` is given; when running, if
            `node_type` isn't in the graph.
    """

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
        unload_model=True,
        cache_dir='cache',
        cache_file='labels',
        load_from_cache=False
    ):
        if model is None and model_path is None:
            raise ValueError('Either pass a model or a model path')

        if model:
            self.model = model
        else:
            self.model = load_llm(model_path)

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

        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.load_from_cache = load_from_cache

        if self.load_from_cache and self.cache_dir is None:
            logger.warning("load_from_cache is set to True but cache_dir is None. Won't load from cache")
            self.load_from_cache = False

        self.str_parameters = {
            'model': self.model.model_name,
            'mask_name': mask_name,
            'node_type': node_type,
            'label_attribute': label_attribute,
            'temperature': temperature
        }

    def _extract_labels(self, context, type_data, node_ids):
        inputs = [self.input_builder(context, self.node_type, node_id) for node_id in node_ids]
        responses = {}

        cache_path = f'{self.cache_dir}/{self.model.sanitized_model_name}/{self.cache_file}.json'

        if self.load_from_cache:
            responses, _ = load_llm_responses_from_cache(cache_path, [input['key'] for input in inputs])

        pending = [input for input in inputs if input['key'] not in responses]
        if pending:
            prompts = [self.prompt_template.format(**input) for input in pending]
            responses_list = self.model.invoke(prompts, self.temperature)
            responses |= {input['key']: response for input,response in zip(pending, responses_list)}

        if self.cache_dir is not None:
            save_llm_responses_to_cache(cache_path, responses)

        parsed = [self.response_parser(responses[input['key']], **self.parser_args) for input in inputs]

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
        labels = self._extract_labels(context, type_data, node_ids)

        if type_data.get(self.label_attribute) is None:
            type_data[self.label_attribute] = torch.full((type_data.num_nodes,), -1)

        type_data[self.label_attribute][label_mask] = labels[label_mask]

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(type_data.num_nodes)]

        for node_id in node_ids:
            context['label_info'][node_id] = {'source': self.model.model_name, 'prob': 1.0}

        if self.unload_model:
            self.model.unload_model()

        return context
