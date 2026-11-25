"""Components that compute node features.

- `SentenceEmbeddingExtractor` (`sentence_embedding_extractor`): embeds each
  node's text with a sentence-transformers model.
"""

from .sentence_embedding import SentenceEmbeddingExtractor

__all__ = [
    'SentenceEmbeddingExtractor',
]
