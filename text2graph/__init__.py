from importlib.metadata import PackageNotFoundError, version

from .pipeline import Pipeline
from .llm import LLM, load_llm

try:
    __version__ = version('text2graph')
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = '0.0.0+unknown'

__all__ = [
    'Pipeline',
    'LLM',
    'load_llm',
    '__version__',
]
