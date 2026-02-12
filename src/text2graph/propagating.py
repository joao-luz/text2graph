import torch
import torch.nn.functional as F

from .component import Component
from .gcn import GCN
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

class LabelPropagator(Component):
    def __init__(self, name):
        super().__init__(name)

    def propagate(self, data):
        pass

class GCNPropagator(LabelPropagator):
    def __init__(self, hidden_channels=16, epochs=200, patience=None, lr=1e-3):
        super().__init__('gcn_propagator')

        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.patience = patience
        self.lr = lr

        self.str_parameters = {
            'epochs': self.epochs,
            'lr': self.lr,
            'patience': self.patience
        }

    def propagate(self, data, train_mask):
        unique_labels,new_labels = torch.unique(data.y[train_mask], sorted=True, return_inverse=True)

        num_classes = len(unique_labels)

        model = GCN(in_channels=data.x.shape[1], hidden_channels=self.hidden_channels, out_channels=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        min_loss = None
        curr_patience = 0
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.edge_index, data.edge_weight)
            loss = F.cross_entropy(out[train_mask], new_labels)

            if self.patience is not None and curr_patience == self.patience:
                print(f'Early stopping due to patience = {self.patience}')
                break
            
            patience_str = ''
            if min_loss is None or loss < min_loss:
                curr_patience = 0
                min_loss = loss
            else:
                curr_patience += 1
                patience_str = f', patience at {curr_patience}'

            print(f'Epoch {epoch}: Loss = {loss}{patience_str}')
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_weight)
            probs, preds = torch.softmax(out, 1).max(dim=1)
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