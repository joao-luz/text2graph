"""Components that add nodes to the graph.

- `DocumentsToNodes` (`documents_to_nodes`): one node per document.
- `LLMNodeGenerator` (`llm_node_generator`): nodes whose text an LLM writes,
  e.g. class descriptions or synthetic examples, labeled with their class.
"""

from .documents_to_nodes import DocumentsToNodes
from .llm_node_generation import LLMNodeGenerator

__all__ = [
    'DocumentsToNodes',
    'LLMNodeGenerator',
]
