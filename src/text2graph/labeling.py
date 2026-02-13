from .component import Component
from .llm import LLM
from .sampling import RandomSampler, DegreeSampler

import regex as re
import torch
import torch.nn.functional as F

node_sampler_map = {
    'random_sampler': RandomSampler,
    'degree_sampler': DegreeSampler
}

class Labeler(Component):
    def __init__(self, name, node_sampler):
        super().__init__(name)

        self.node_sampler = node_sampler

    def label(self, inputs):
        pass

    def __call__(self):
        pass

class GroundTruthLabeler(Labeler):
    def __init__(self, node_sampler, label_attribute='true_label'):
        super().__init__('ground_truth_labeler', node_sampler)

        self.label_attribute = label_attribute

    def extract_labels(self, data, node_ids):
        ground_truths = data[self.label_attribute]

        labels = torch.full((data.num_nodes, ), -1)
        for node_id in node_ids:
            labels[node_id] = ground_truths[node_id]

        return labels
    
    def __call__(self, data):
        data = data.clone()

        node_ids = self.node_sampler.sample_nodes(data).tolist()
        labels = self.extract_labels(data, node_ids)
        
        if not data.get('label_info'):
            data.label_info = [{} for _ in range(data.num_nodes)]

        for node_id in node_ids:
            data.label_info[node_id] = {'source': 'ground_truth', 'prob': 1.0}
        
        data.y = labels

        return data

class LLMLabeler(Labeler):
    def __init__(self, prompt_template, node_sampler, input_builder=None, response_parser=None, model=None, model_path=None, parser_args={}, temperature=0.0):
        super().__init__(f'llm_labeler', node_sampler=node_sampler)

        assert model or model_path, 'Either pass a model or a model path'

        if model:
            self.model = model
        else:
            self.model = LLM(model_path)

        def default_input_builder(data, node_id, cap=1200):
            text = data.text[node_id]
            index = sum(len(token) for token in text.split()[:cap]) + cap
            text = text[:index]
            return {'text': text}

        def default_parser(response, options):
            m = re.search(r'([0-9]+)', response)
            if m: option = int(m[1])
            else: option = 0

            return options[option]

        self.prompt_template = prompt_template
        self.node_sampler = node_sampler
        self.input_builder = input_builder or default_input_builder
        self.response_parser = response_parser or default_parser
        self.parser_args = parser_args
        self.temperature = temperature

        if isinstance(node_sampler, str):
            self.node_sampler = node_sampler_map[node_sampler]
        else:
            self.node_sampler = node_sampler

        self.str_parameters = {
            'model': self.model.name,
            'sampler': self.node_sampler
        }

    def set_prompt_template(self, new_template):
        self.prompt_template = new_template

    def set_parser_args(self, new_args):
        self.parser_args = new_args

    def extract_labels(self, data, node_ids):
        inputs = [self.input_builder(data, node_id) for node_id in node_ids]
        prompts = [self.prompt_template.format(**input) for input in inputs]
        outputs = self.model.query(prompts, self.temperature)
        parsed = [self.response_parser(output, **self.parser_args) for output in outputs]

        labels = torch.full((data.num_nodes, ), -1)
        for i,node_id in enumerate(node_ids):
            labels[node_id] = data.label2id[parsed[i]]

        return labels
    
    def __call__(self, data):
        data = data.clone()

        node_ids = self.node_sampler.sample_nodes(data).tolist()
        labels = self.extract_labels(data, node_ids)

        if not data.get('label_info'):
            data.label_info = [{} for _ in range(data.num_nodes)]

        for node_id in node_ids:
            data.label_info[node_id] = {'source': self.model.name, 'prob': 1.0}
        
        data.y = labels

        return data
    
class LLMEnsembleLabeler(Labeler):
    def __init__(self, 
        prompt_template, 
        node_sampler, 
        model_paths, 
        resolver='majority_vote', 
        threshold=None,
        concatenate_decisions_to_x=None,
        decision_features_attribute=None, 
        input_builder=None, 
        response_parser=None, 
        parser_args={}, 
        temperature=0.0
    ):
        super().__init__(f'llm_ensemble_labeler', node_sampler=node_sampler)

        self.model_paths = model_paths

        def default_input_builder(data, node_id, cap=1200):
            text = data.text[node_id]
            index = sum(len(token) for token in text.split()[:cap]) + cap
            text = text[:index]
            return {'text': text}

        def default_parser(response, options):
            m = re.search(r'([0-9]+)', response)
            if m: option = int(m[1])
            else: option = 0

            return options[option]
        
        self.threshold = threshold
        def majority_vote_resolver(decisions, threshold=threshold):
            votes = torch.stack(list(decisions.values()))
            majority_votes,_ = torch.mode(votes, dim=0)
            probs = torch.sum(votes == majority_votes, dim=0)/votes.shape[0]

            if threshold:
                majority_votes[probs < threshold] = -1

            return majority_votes, probs

        if resolver == 'majority_vote':
            self.resolver = majority_vote_resolver

        self.decision_features_attribute = decision_features_attribute
        self.concatenate_decisions_to_x = concatenate_decisions_to_x
        self.prompt_template = prompt_template
        self.node_sampler = node_sampler
        self.input_builder = input_builder or default_input_builder
        self.response_parser = response_parser or default_parser
        self.parser_args = parser_args
        self.temperature = temperature

        if isinstance(node_sampler, str):
            self.node_sampler = node_sampler_map[node_sampler]
        else:
            self.node_sampler = node_sampler

        self.str_parameters = {
            'models': self.model_paths,
            'sampler': self.node_sampler,
            'resolver': self.resolver
        }

    def set_prompt_template(self, new_template):
        self.prompt_template = new_template

    def set_parser_args(self, new_args):
        self.parser_args = new_args

    def extract_labels(self, data, node_ids):
        inputs = [self.input_builder(data, node_id) for node_id in node_ids]
        prompts = [self.prompt_template.format(**input) for input in inputs]

        decisions = {}
        for model_path in self.model_paths:
            print(f'Voting with {model_path}...')

            model = LLM(model_path)
            outputs = model.query(prompts, self.temperature)
            del model

            parsed = [self.response_parser(output, **self.parser_args) for output in outputs]

            decisions[model_path] = torch.tensor([data.label2id[p] for p in parsed])

        preds,probs = self.resolver(decisions)

        preds_full = torch.full((data.num_nodes, ), -1)
        probs_full = torch.zeros((data.num_nodes, ))
        for i,node_id in enumerate(node_ids):
            preds_full[node_id] = preds[i]
            probs_full[node_id] = probs[i]

        if not any(preds_full != -1):
            raise ValueError(f'No nodes labeled with at least {self.threshold:.2f} agreement between ensemble models.')

        for model in self.model_paths:
            decisions_full = torch.full((data.num_nodes, ), -1)

            for i,node_id in enumerate(node_ids):
                decisions_full[node_id] = decisions[model][i]
            
            decisions[model] = decisions_full

        def one_hot_with_ignore(labels, num_classes, ignore_index):
            valid_mask = (labels != ignore_index).unsqueeze(-1)

            processed_labels = labels.clone()
            processed_labels[labels == ignore_index] = 0

            one_hot = F.one_hot(processed_labels, num_classes=num_classes).float()
            one_hot = one_hot * valid_mask.float()
            
            return one_hot

        decisions_stacked = torch.stack(list(decisions.values()))
        num_classes = len(data.id2label)
        decision_features = one_hot_with_ignore(decisions_stacked.t(), num_classes, -1)
        decision_features = decision_features.flatten(1)
            
        return preds_full, probs_full, decisions, decision_features
    
    def __call__(self, data):
        data = data.clone()

        node_ids = self.node_sampler.sample_nodes(data).tolist()
        preds,probs,decisions,decision_features = self.extract_labels(data, node_ids)

        if not data.get('label_info'):
            data.label_info = [{} for _ in range(data.num_nodes)]

        for node_id in node_ids:
            data.label_info[node_id] = {
                'source': ', '.join(self.model_paths) + f' @ {self.threshold:.2f}',
                'decisions': {model: decisions[model][node_id].item() for model in self.model_paths },
                'prob': probs[node_id].item()}
        
        data.y = preds
        
        if self.concatenate_decisions_to_x:
            if self.concatenate_decisions_to_x == 'prepend':
                data.x = torch.concat((decision_features, data.x), dim=1)
            else:
                data.x = torch.concat((data.x, decision_features), dim=1)

        elif self.decision_features_attribute:
            data[self.decision_features_attribute] = decision_features

        return data
