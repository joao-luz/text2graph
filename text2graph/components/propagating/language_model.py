from ..component import Component
from ..component_registry import register_component

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from torch_geometric.data import HeteroData


@register_component('lm_propagator')
class LMPropagator(Component):
    def __init__(
        self,
        model_path,
        epochs=15,
        lr=5e-5,
        patience=None,
        batch_size=16,
        max_length=128,
        node_type='documents',
        label_attribute='pseudo_y'
    ):
        self.model_path = model_path
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.max_length = max_length
        self.node_type = node_type
        self.label_attribute = label_attribute

        self.str_parameters = {
            'model_path': model_path,
            'epochs': epochs,
            'lr': lr,
            'patience': patience,
            'batch_size': batch_size,
            'max_length': max_length,
            'node_type': node_type,
            'label_attribute': label_attribute
        }

    def propagate(self, data, train_mask):
        unique_labels, new_labels = torch.unique(data[self.label_attribute][train_mask], sorted=True, return_inverse=True)

        num_classes = len(unique_labels)

        train_texts = [text for text,m in zip(data.text, train_mask) if m]
        train_labels = new_labels

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_classes
        )

        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        full_dataset = Dataset.from_dict({
            'text': data.text
        })

        def tokenize_fn(batch):
            return tokenizer(
                batch['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        full_dataset = full_dataset.map(tokenize_fn, batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        full_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        training_args = TrainingArguments(
            output_dir="./lm_propagator_results",
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            eval_strategy="no",
            save_strategy="no",
            logging_dir="./logs",
            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()

        out = trainer.predict(full_dataset)
        out_probs = torch.softmax(torch.tensor(out.predictions), 1)
        probs, preds = out_probs.max(axis=1)
        preds = unique_labels[preds]

        return preds, probs

    def run(self, context):
        data = context['graph']
        
        if self.node_type not in context['node_types']:
            raise ValueError(f'"{self.node_type}" not a valid type, only {context["node_types"]}')
        
        type_data = data[self.node_type] if isinstance(data, HeteroData) else data

        train_mask = type_data[self.label_attribute] != -1
        unlabeled_node_ids = torch.arange(type_data.num_nodes)[~train_mask]

        preds, probs = self.propagate(type_data, train_mask)

        if not context.get('label_info'):
            context['label_info'] = [{} for _ in range(context.num_nodes)]

        for node_id,pred,prob in zip(unlabeled_node_ids, preds[~train_mask], probs[~train_mask]):
            type_data['label_info'][node_id] = {'source': 'lm_propagator', 'prob': prob.item()}
            type_data[self.label_attribute][node_id] = pred

        return context