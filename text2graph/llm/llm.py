"""LLM backends used by the text2graph components.

All backends share the same interface (``invoke``, ``embed``, ``load_model``,
``unload_model``, ``set_default_chat_kwargs``, ``model_name``,
``sanitized_model_name``), so components don't need to know where the model runs.

The backend is picked from a prefix on the model name, or from ``backend=``:

    LLM('Qwen/Qwen3-8B')                    # local vLLM engine (default)
    LLM('openai:gpt-5-mini')                # OpenAI API
    LLM('openai:Qwen/Qwen3-8B',             # any OpenAI-compatible server
        base_url='http://localhost:8000/v1')   # (vllm serve, OpenRouter, ...)
    LLM('anthropic:claude-opus-5-5')        # Anthropic API

Because the prefix is part of the name, configs that only pass ``model_path``
can use an API model without code changes. API keys and base URLs are read from
the SDKs' usual environment variables (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``,
``ANTHROPIC_API_KEY``, ...) unless passed explicitly.
"""

import gc
import importlib
import os
import re
import warnings

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


_EXTRAS = {'vllm': 'vllm', 'ray': 'vllm', 'openai': 'openai', 'anthropic': 'anthropic'}


def _require(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        extra = _EXTRAS.get(module_name, module_name)
        raise ImportError(
            f'This LLM backend requires "{module_name}", which is part of the optional "{extra}" extra. '
            f'Install it with `pip install "text2graph[{extra}]"`.'
        ) from e


def _strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL).strip()


def _split_backend(model_name, backend):
    prefix, sep, rest = model_name.partition(':')
    if sep and prefix in _BACKENDS:
        if backend is not None and backend != prefix:
            raise ValueError(f'Model name "{model_name}" has prefix "{prefix}" but backend="{backend}" was passed.')
        return prefix, rest

    return backend or 'vllm', model_name


class LLM:
    backend = None

    def __new__(cls, model_name, *args, backend=None, **kwargs):
        if cls is LLM:
            backend, _ = _split_backend(model_name, backend)
            if backend not in _BACKENDS:
                raise ValueError(f'Unknown LLM backend "{backend}". Choose one of {sorted(_BACKENDS)}.')
            cls = _BACKENDS[backend]

        return super().__new__(cls)

    def __init__(self, model_name, backend=None):
        _, name = _split_backend(model_name, backend or self.backend)

        self.api_model_name = name
        self.model_name = name if self.backend == 'vllm' else f'{self.backend}:{name}'
        self.sanitized_model_name = self.model_name.replace('/', '--').replace(':', '--')
        self.loaded = False
        self.model = None
        self.default_chat_kwargs = {}

    @classmethod
    def from_config(cls, config):
        config = dict(config)

        if 'model' not in config:
            raise ValueError(f'LLM config needs a "model" key, got keys {sorted(config)}.')
        model_name = config.pop('model')
        chat_kwargs = config.pop('chat_kwargs', None)
        api_key_env = config.pop('api_key_env', None)

        if api_key_env is not None:
            if api_key_env not in os.environ:
                raise KeyError(f'LLM config sets api_key_env="{api_key_env}", but that environment variable is not set.')
            config['api_key'] = os.environ[api_key_env]

        llm = cls(model_name, **config)
        if chat_kwargs:
            llm.set_default_chat_kwargs(chat_kwargs)

        return llm

    @classmethod
    def from_yaml(cls, path):
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, Mapping):
            raise ValueError(f'"{path}" must contain a YAML mapping with at least a "model" key.')

        return cls.from_config(config)

    def set_default_chat_kwargs(self, chat_kwargs):
        self.default_chat_kwargs = chat_kwargs

    def load_model(self, **kwargs):
        self.loaded = True

    def unload_model(self):
        self.loaded = False

    def __del__(self):
        try:
            self.unload_model()
        except Exception:
            pass

    def invoke(self, prompts, temperature=0.7, sampling_kwargs={}, chat_kwargs={}):
        if not prompts:
            return []

        single_prompt = not isinstance(prompts, list)
        if single_prompt:
            prompts = [prompts]

        if not self.loaded:
            self.load_model()

        texts = self._generate(prompts, temperature, sampling_kwargs, chat_kwargs or self.default_chat_kwargs)
        texts = [_strip_thinking(text) for text in texts]

        return texts[0] if single_prompt else texts

    def embed(self, prompts):
        raise NotImplementedError(f'The "{self.backend}" backend does not support embeddings.')

    def _generate(self, prompts, temperature, sampling_kwargs, chat_kwargs):
        raise NotImplementedError


class VLLM(LLM):
    backend = 'vllm'

    def __init__(self, model_name, load_model=False, backend=None, **vllm_args):
        super().__init__(model_name, backend)
        self.default_vllm_args = {'max_model_len': 4096, 'gpu_memory_utilization': 0.8} | vllm_args

        if load_model:
            self.load_model(**vllm_args)

    def unload_model(self):
        if not self.loaded:
            return

        try:
            del self.model
        except Exception as e:
            print(f"Failed to unload model: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            _require('ray').shutdown()

        self.model = None
        self.loaded = False

    def load_model(self, **vllm_args):
        if self.loaded:
            return

        vllm = _require('vllm')

        vllm_args = self.default_vllm_args | vllm_args
        self.model = vllm.LLM(
            model=self.api_model_name,
            trust_remote_code=True,
            language_model_only=True,
            **vllm_args
        )

        self.loaded = True

    def _generate(self, prompts, temperature, sampling_kwargs, chat_kwargs):
        vllm = _require('vllm')

        chats = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=2048, **sampling_kwargs)
        outputs = self.model.chat(chats, sampling_params, **chat_kwargs)

        return [output.outputs[0].text for output in outputs]

    def embed(self, prompts):
        if not self.loaded:
            self.load_model(runner='pooling')

        outputs = self.model.encode(prompts, pooling_task='embed')

        return torch.stack([o.outputs.data for o in outputs])


class _APILLM(LLM):
    def __init__(
            self,
            model_name,
            backend=None,
            api_key=None,
            base_url=None,
            max_tokens=16000,
            max_concurrency=8,
            max_retries=5,
            timeout=None,
            **request_kwargs
        ):
        super().__init__(model_name, backend)
        self.client_kwargs = {'api_key': api_key, 'base_url': base_url, 'max_retries': max_retries}
        if timeout is not None:
            self.client_kwargs['timeout'] = timeout
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.request_kwargs = request_kwargs

    def load_model(self, **kwargs):
        if not self.loaded:
            self.model = self._make_client()
            self.loaded = True

    def unload_model(self):
        if self.loaded and self.model is not None:
            self.model.close()
        self.model = None
        self.loaded = False

    def _generate(self, prompts, temperature, sampling_kwargs, chat_kwargs):
        kwargs = self.request_kwargs | chat_kwargs | sampling_kwargs

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(lambda prompt: self._complete(prompt, temperature, kwargs), prompts))

    def _make_client(self):
        raise NotImplementedError

    def _complete(self, prompt, temperature, kwargs):
        raise NotImplementedError


class OpenAILLM(_APILLM):
    backend = 'openai'

    def _make_client(self):
        return _require('openai').OpenAI(**self.client_kwargs)

    def _complete(self, prompt, temperature, kwargs):
        params = {'max_completion_tokens': self.max_tokens} | kwargs
        if temperature is not None:
            params['temperature'] = temperature

        response = self.model.chat.completions.create(
            model=self.api_model_name,
            messages=[{'role': 'user', 'content': prompt}],
            **params
        )

        return response.choices[0].message.content or ''

    def embed(self, prompts):
        if not self.loaded:
            self.load_model()

        response = self.model.embeddings.create(model=self.api_model_name, input=prompts)

        return torch.tensor([item.embedding for item in response.data])


class AnthropicLLM(_APILLM):
    backend = 'anthropic'

    def __init__(self, model_name, backend=None, **kwargs):
        super().__init__(model_name, backend, **kwargs)
        self._warned_temperature = False

    def _make_client(self):
        return _require('anthropic').Anthropic(**self.client_kwargs)

    def _complete(self, prompt, temperature, kwargs):
        if temperature is not None and not self._warned_temperature:
            warnings.warn('AnthropicLLM ignores temperature: current Claude models do not accept it.')
            self._warned_temperature = True

        response = self.model.messages.create(
            model=self.api_model_name,
            max_tokens=self.max_tokens,
            messages=[{'role': 'user', 'content': prompt}],
            **kwargs
        )

        if response.stop_reason == 'refusal':
            return ''

        return ''.join(block.text for block in response.content if block.type == 'text')


def load_llm(spec):
    if isinstance(spec, LLM):
        return spec

    if isinstance(spec, Mapping):
        return LLM.from_config(spec)

    if isinstance(spec, (str, os.PathLike)) and Path(spec).suffix in {'.yaml', '.yml'}:
        return LLM.from_yaml(spec)

    if isinstance(spec, str):
        return LLM(spec)

    raise TypeError(f'Cannot build an LLM from {type(spec).__name__}: pass a model name, a YAML path, a dict or an LLM.')


_BACKENDS = {
    'vllm': VLLM,
    'openai': OpenAILLM,
    'anthropic': AnthropicLLM,
}
