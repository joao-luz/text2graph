"""Build graphs from text and label their nodes with LLMs and graph models.

A `Pipeline` runs a sequence of components over a shared context dict. A typical
pipeline:

1. Node creation: turn documents into nodes, and optionally add LLM-generated
   nodes such as class descriptions (`text2graph.components.node_creation`).
2. Feature extraction: embed each node's text (`...feature_extraction`).
3. Edge creation: connect similar nodes (`...edge_creation`).
4. Sampling: pick the nodes worth labeling (`...sampling`).
5. Labeling: label them, e.g. with one or several LLMs (`...labeling`).
6. Propagation: label the remaining nodes from the labeled ones, e.g. with a
   GNN (`...propagating`).

`LLM` (and `load_llm`) give the components one interface to local vLLM models
and to the OpenAI and Anthropic APIs.
"""

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
