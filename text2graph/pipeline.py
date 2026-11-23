import logging

from string import Formatter

from .components import RepeatComponent
from .components.component_registry import get_component


logger = logging.getLogger(__name__)


__all__ = [
    'Pipeline',
]


def _resolve_str_to_object(obj_str, context):
    clean_path = obj_str.strip('${}')

    formatter = Formatter()
    try:
        obj, _ = formatter.get_field(clean_path, (), context)
        return obj
    except (KeyError, IndexError, AttributeError):
        return None


def _is_obj_str(obj_str):
    return obj_str.startswith('${') and obj_str.endswith('}')


def _convert_parameter(val_str, context):
    if _is_obj_str(val_str):
        val = _resolve_str_to_object(val_str, context)
    else:
        val = val_str.format(**context)

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
    def __init__(self, steps=None, config=None, verbose=True, **context_kwargs):
        if (steps is None) == (config is None):
            raise ValueError('Pass exactly one of `steps` (a list of components) or `config` (a dict with a "steps" list)')

        self.steps = steps
        self.context = context_kwargs
        self.verbose = verbose

        if config is not None:
            self.steps = _load_steps_from_config(config['steps'], self.context)

    def __str__(self):
        steps_str = '\n'.join(['\t' + line for line in ',\n'.join([str(step) for step in self.steps]).split('\n')])
        return 'Pipeline(steps=[\n' + steps_str + '\n])'

    def run(self, **context_kwargs):
        self.context |= context_kwargs

        for step in self.steps:
            if isinstance(step, RepeatComponent):
                step.verbose = self.verbose
                
            if self.verbose:
                logger.info('%s...', step)
            self.context = step(self.context)

        return self.context