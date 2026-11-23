import difflib
import logging


logger = logging.getLogger(__name__)


COMPONENTS = {}


def _qualified_name(cls):
    return f'{cls.__module__}.{cls.__qualname__}'


def register_component(name):
    def wrapper(cls):
        existing = COMPONENTS.get(name)
        if existing is not None and _qualified_name(existing) != _qualified_name(cls):
            logger.warning(
                'Component name "%s" was already registered by %s; it now refers to %s',
                name, _qualified_name(existing), _qualified_name(cls)
            )

        COMPONENTS[name] = cls
        return cls

    return wrapper


def get_component(name):
    """Return the component class registered under `name`.

    Raises:
        ValueError: If no component is registered under `name`. The message suggests
            close matches and lists every registered name.
    """
    try:
        return COMPONENTS[name]
    except KeyError:
        pass

    message = f'Unknown component "{name}".'
    suggestions = difflib.get_close_matches(name, COMPONENTS, n=3)
    if suggestions:
        message += ' Did you mean ' + ' or '.join(f'"{s}"' for s in suggestions) + '?'
    message += (
        f' Registered components: {", ".join(sorted(COMPONENTS))}.'
        ' Custom components must be imported (so @register_component runs) before the pipeline is built.'
    )
    raise ValueError(message)
