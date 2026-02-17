import vllm

import gc
import ray
import torch

class LLM():
    def __init__(self, model_path=None, load_model=False):
        self.model = None
        self.name = model_path
        self.model_path = ''

        if model_path is not None:
            self.model_path = self._name_mapping(model_path)
            if load_model:
                self.model = self.load_model()

    def _name_mapping(self, name):
        mapping = {
            'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
            'qwen2': 'Qwen/Qwen2.5-7B-Instruct',
            'gemma2': 'google/gemma-2-9b-it', 
            'phi3:14b': 'microsoft/Phi-3-medium-128k-instruct',
            'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
        }

        return mapping[name] if name in mapping else name
    
    def _delete_model(self):
        if self.model is None:
            return
        
        try:
            del self.model
        except Exception as e:
            print(f"Failed to unload model: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()

    def __del__(self):
        self._delete_model()

    def load_model(self, model_path=None):
        model_path = model_path or self.model_path
        if model_path is None:
            raise ValueError(f'Must pass a model_name or have self.model_name set.')

        self._delete_model()

        model_path = self._name_mapping(model_path)

        self.model_path = model_path

        if model_path == 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit':
            model = vllm.LLM(model=model_path, trust_remote_code=True,
                             max_model_len=2048,
                             gpu_memory_utilization=0.7,
                             cpu_offload_gb=45
                             )
        else:
            model = vllm.LLM(model=model_path, trust_remote_code=True,
                             max_model_len=2048,
                             gpu_memory_utilization=0.9,
                             )
    
        self.model = model
    
    def query(self, prompts, temperature):
        if self.model is None and self.model_path:
            self.load_model(self.model_path)

        chats = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=1024)

        outputs = self.model.chat(chats, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        return generated_texts