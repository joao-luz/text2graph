"""One interface to language models, run locally or through an API.

`LLM(name)` picks the backend from a prefix on the model name: none or `vllm:`
for a local vLLM engine (`VLLM`), `openai:` for the OpenAI API or any
OpenAI-compatible server (`OpenAILLM`), `anthropic:` for the Anthropic API
(`AnthropicLLM`). `load_llm` also accepts an existing `LLM` or a config dict.
"""

from .llm import LLM, VLLM, OpenAILLM, AnthropicLLM, load_llm

__all__ = [
    'LLM',
    'VLLM',
    'OpenAILLM',
    'AnthropicLLM',
    'load_llm',
]
