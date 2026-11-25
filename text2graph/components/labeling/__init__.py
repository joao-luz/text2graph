"""Components that label nodes, usually the ones a sampler picked.

Each records how it labeled a node in `context['label_info']`.

- `LLMLabeler` (`llm_labeler`): asks an LLM to classify each node.
- `LLMEnsembleLabeler` (`llm_ensemble_labeler`): asks several LLMs and combines
  their votes.
- `KNNLabeler` (`knn_labeler`): a majority vote of the most similar labeled nodes,
  e.g. LLM-generated class descriptions.
- `GroundTruthLabeler` (`ground_truth_labeler`): the true labels, as an oracle
  baseline.
"""

from .ground_truth import GroundTruthLabeler
from .llm import LLMLabeler
from .llm_ensemble import LLMEnsembleLabeler
from .knn import KNNLabeler

__all__ = [
    'GroundTruthLabeler',
    'LLMLabeler',
    'LLMEnsembleLabeler',
    'KNNLabeler',
]
