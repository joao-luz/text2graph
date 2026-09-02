from .components import RepeatComponent
from .components.component_registry import COMPONENTS


def resolve_str_to_object(str, context):
    from string import Formatter

    clean_path = str.strip('${}')
    
    formatter = Formatter()
    try:
        obj, _ = formatter.get_field(clean_path, (), context)
        return obj
    except (KeyError, IndexError, AttributeError):
        return None


def is_obj_str(str):
    return str.startswith('${') and str.endswith('}')


def convert_parameter(val_str, context):
    if is_obj_str(val_str):
        val = resolve_str_to_object(val_str, context)
    else:
        val = val_str.format(**context)

    return val


def load_component_from_config(component_config, context):
    name = component_config['name']
    cls = COMPONENTS[name]

    if cls == RepeatComponent:
        parameters = component_config.get('parameters') or {}
        steps = parameters.get('steps') or {}
        parameters['steps'] = load_steps_from_config(steps, context)

    else:
        parameters = component_config.get('parameters') or {}
        for param,val in parameters.items():
            if isinstance(val, list):
                val = [convert_parameter(v, context) for v in val]
            elif isinstance(val, str):
                val = convert_parameter(val, context)

            parameters[param] = val

    component = cls(**parameters)

    return component


def load_steps_from_config(component_configs, context):
    steps = []

    for component_config in component_configs:
        component = load_component_from_config(component_config, context)
        steps.append(component)

    return steps


class Pipeline:
    def __init__(self, steps=None, config=None, verbose=True, **context_kwargs):
        assert (steps is not None) ^ (config is not None), 'Either pass a list of steps for the pipeline or a config dict.'

        self.steps = steps
        self.context = context_kwargs
        self.verbose = verbose

        if config is not None:
            self.steps = load_steps_from_config(config['steps'], self.context)

    def __str__(self):
        steps_str = '\n'.join(['\t' + line for line in ',\n'.join([str(step) for step in self.steps]).split('\n')])
        return 'Pipeline(steps=[\n' + steps_str + '\n])'

    def run(self, **context_kwargs):
        self.context |= context_kwargs

        for step in self.steps:
            if isinstance(step, RepeatComponent):
                step.verbose = self.verbose
                
            if self.verbose:
                print(str(step) + '...')
            self.context = step(self.context)

        return self.context