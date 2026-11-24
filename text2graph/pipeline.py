import logging

from collections.abc import Mapping
from string import Formatter

from .components import RepeatComponent
from .components.component_registry import get_component


logger = logging.getLogger(__name__)


__all__ = [
    'Pipeline',
]


def _available_keys(context):
    return ', '.join(sorted(context)) or '(none)'


def _resolve_str_to_object(obj_str, context):
    clean_path = obj_str.strip('${}')

    formatter = Formatter()
    try:
        obj, _ = formatter.get_field(clean_path, (), context)
        return obj
    except (KeyError, IndexError, AttributeError, TypeError) as e:
        raise ValueError(
            f'Could not resolve "{obj_str}" ({type(e).__name__}: {e}). '
            f'Available Pipeline arguments: {_available_keys(context)}'
        ) from e


def _is_obj_str(obj_str):
    return obj_str.startswith('${') and obj_str.endswith('}')


def _convert_parameter(val_str, context):
    if _is_obj_str(val_str):
        val = _resolve_str_to_object(val_str, context)
    else:
        try:
            val = val_str.format(**context)
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            raise ValueError(
                f'Could not format "{val_str}" ({type(e).__name__}: {e}). '
                f'Available Pipeline arguments: {_available_keys(context)}. '
                'Literal braces in config strings must be doubled: {{ and }}'
            ) from e

    return val


def _load_component_from_config(component_config, context):
    if 'name' not in component_config:
        raise ValueError(f'Pipeline step is missing a "name": {component_config}')

    cls = get_component(component_config['name'])

    parameters = dict(component_config.get('parameters') or {})

    if cls == RepeatComponent:
        steps = parameters.get('steps') or []
        parameters['steps'] = _load_steps_from_config(steps, context)

    else:
        for param,val in parameters.items():
            if isinstance(val, list):
                val = [_convert_parameter(v, context) if isinstance(v, str) else v for v in val]
            elif isinstance(val, str):
                val = _convert_parameter(val, context)

            parameters[param] = val

    component = cls(**parameters)

    return component


def _load_steps_from_config(component_configs, context):
    steps = []

    for component_config in component_configs:
        component = _load_component_from_config(component_config, context)
        steps.append(component)

    return steps


class Pipeline:
    """Runs a sequence of components that build and label a graph.

    Every step receives a shared `context` dict and returns it, usually after adding
    to it or changing the graph in it (see `Component`). The context starts out
    as the keyword arguments given here and to `run`::

        pipeline = Pipeline([DocumentsToNodes(), SentenceEmbeddingExtractor(model_path='...'), ...])
        context = pipeline.run(documents=texts)

    The steps can also be built from a config dict, by passing `config` instead
    of `steps` (or with `Pipeline.from_config`, which documents the format)::

        pipeline = Pipeline(config=config, llms=llms, dataset='r8')

    Args:
        steps: List of components to run, in order.
        config: Mapping with a `steps` list describing the components; see
            `Pipeline.from_config`. Pass either `steps` or `config`, not both.
        verbose: Log each step at INFO level before running it, through the
            `text2graph.pipeline` logger (shown only if logging is configured,
            e.g. with `logging.basicConfig(level=logging.INFO)`).
        **context_kwargs: Initial context, e.g. `documents`. With `config`, config
            strings can also refer to these values (e.g. `dataset`, `llms`).

    Attributes:
        steps: The components, in order.
        context: The shared context; after `run`, it holds everything the steps produced.

    Raises:
        ValueError: If both or neither of `steps` and `config` are given, or the
            config is invalid (see `Pipeline.from_config`).
    """

    def __init__(self, steps=None, config=None, verbose=True, **context_kwargs):
        if (steps is None) == (config is None):
            raise ValueError('Pass exactly one of `steps` (a list of components) or `config` (a dict with a "steps" list)')

        if config is not None:
            if not isinstance(config, Mapping) or not isinstance(config.get('steps'), list):
                raise ValueError(f'Pipeline config must be a mapping with a "steps" list, got {config!r}')
            steps = _load_steps_from_config(config['steps'], context_kwargs)

        self.steps = list(steps)
        self.context = context_kwargs
        self.verbose = verbose

    @classmethod
    def from_config(cls, config, verbose=True, **context_kwargs):
        """Build a pipeline whose steps are described by a config dict.

        Same as `Pipeline(config=config, ...)`. Each step names a registered component (see `register_component`) and
        the keyword arguments to build it with::

            config = {'steps': [
                {'name': 'documents_to_nodes',
                 'parameters': {'node_type': 'documents'}},
                {'name': 'llm_labeler',
                 'parameters': {
                     'model': '${llms[0]}',                   # an object from context_kwargs
                     'cache_dir': 'cache/{dataset}',          # a string with context_kwargs filled in
                     'prompt_template': '${labeling_prompt_template}',
                 }},
            ]}
            pipeline = Pipeline.from_config(config, llms=llms, dataset='r8', labeling_prompt_template=prompt)

        String parameters are resolved against `context_kwargs` when the pipeline
        is built:

        - `${path}` (the whole string) is replaced by the object at `path`, which can
          use attribute access and indexing: `${llms[0]}`, `${llms[0].model_name}`.
        - Any other string is filled in with `str.format`: `cache/{dataset}`. Literal
          braces, e.g. JSON in a prompt, must be doubled: `{{` and `}}`.
        - Strings inside a list are resolved one by one. Other values, including
          dicts and anything nested in them, are passed to the component unchanged.
        - The `steps` of a `repeat_component` are themselves loaded as a list of
          steps in this format.

        The config itself is never modified, so it can be reused to build several pipelines.

        Args:
            config: Mapping with a `steps` list, in the format above.
            verbose: See `Pipeline`.
            **context_kwargs: Initial context, as for `Pipeline`. Config strings can
                refer to any of these values (e.g. `dataset`, `llms`).

        Returns:
            The `Pipeline`, with the built components as its steps.

        Raises:
            ValueError: If `config` has no `steps` list, a step has no `name` or an
                unknown one, or a config string refers to something missing from
                `context_kwargs`.
        """
        return cls(config=config, verbose=verbose, **context_kwargs)

    def __str__(self):
        steps_str = '\n'.join(['\t' + line for line in ',\n'.join([str(step) for step in self.steps]).split('\n')])
        return 'Pipeline(steps=[\n' + steps_str + '\n])'

    def run(self, **context_kwargs):
        """Run every step in order, passing the context from one to the next.

        The context is kept between calls: a second `run` starts from what the
        first one left (including its graph), not from a fresh context.

        Args:
            **context_kwargs: Added to the context before the first step, replacing
                existing keys with the same name (e.g. `documents`, `labels`).

        Returns:
            The final context dict. The graph is in `context['graph']`.
        """
        self.context |= context_kwargs

        for step in self.steps:
            if isinstance(step, RepeatComponent):
                step.verbose = self.verbose
                
            if self.verbose:
                logger.info('%s...', step)
            self.context = step(self.context)

        return self.context