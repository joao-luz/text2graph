import difflib
import logging


logger = logging.getLogger(__name__)


COMPONENTS = {}


def _qualified_name(cls):
    return f'{cls.__module__}.{cls.__qualname__}'


def register_component(name):
    """Class decorator that makes a component usable in pipeline configs under `name`.

    The class is registered when its module is imported, so import custom
    components before building a `Pipeline` from a config::

        @register_component('my_labeler')
        class MyLabeler(Component):
            ...

    If `name` is already taken by a different class, a warning is logged and the
    new class replaces the old one.

    Args:
        name: Name used as a step's `name` in configs.

    Returns:
        The decorator, which returns the class unchanged.
    """
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

    Args:
        name: Name the class was registered under with `register_component`.

    Returns:
        The component class (not an instance).

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
