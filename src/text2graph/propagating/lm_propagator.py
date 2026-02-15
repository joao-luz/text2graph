from .propagating import LabelPropagator

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset


class LMPropagator(LabelPropagator):
    def __init__(
        self,
        model_path,
        epochs=15,
        lr=5e-5,
        patience=None,
        batch_size=16,
        max_length=128,
    ):
        super().__init__("lm_propagator")
        self.model_path = model_path
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.max_length = max_length

    def propagate(self, data, train_mask):
        unique_labels,new_labels = torch.unique(data.y[train_mask], sorted=True, return_inverse=True)

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

    def __call__(self, data):
        data = data.clone()

        train_mask = data.y != -1
        unlabeled_node_ids = torch.arange(data.num_nodes)[~train_mask]

        preds, probs = self.propagate(data, train_mask)

        if not data.get('label_info'):
            data.label_info = [{} for _ in range(data.num_nodes)]

        for node_id,pred,prob in zip(unlabeled_node_ids, preds[~train_mask], probs[~train_mask]):
            data.label_info[node_id] = {'source': 'lm_propagator', 'prob': prob.item()}
            data.y[node_id] = pred

        return data