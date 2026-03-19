_registry = {}


def register_component(name, cls):
    if name in _registry:
        raise ValueError(f'Component "{name}" already registered.')
    _registry[name] = cls


def get_component(name):
    if name not in _registry:
        raise ValueError(f'Unknown component "{name}".')
    return _registry[name]


def component_from_config(config, **kwargs):
    name = config['name']

    cls = get_component(name)
    params = config.get('parameters', {})

    return cls.from_config(params, **kwargs)