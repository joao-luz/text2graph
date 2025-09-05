import torch
import torch.nn.functional as F

from .component import Component
from .gcn import GCN
from torch_geometric.data import Data
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, DatasetDict

class LabelPropagator(Component):
    def __init__(self, name):
        super().__init__(name)

    def propagate(self, G):
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

    def propagate(self, G):
        # Find the existing classes
        classes = list(set([G.nodes[n].get('class')['label'] for n in G.nodes if G.nodes[n].get('class')]))
        label2id = {name: classes.index(name) for name in classes}
        num_classes = len(classes)

        # Build node feature matrix and label vector
        node_list = list(G.nodes())
        node_id_map = {n: i for i, n in enumerate(node_list)}

        # Node features
        features = []
        labels = []
        train_mask = []

        for n in node_list:
            emb = G.nodes[n]['embedding']
            features.append(emb)

            label = G.nodes[n].get('class')
            if label is not None:
                labels.append(label2id[label['label']])
                train_mask.append(True)
            else:
                labels.append(0)   # dummy
                train_mask.append(False)

        x = torch.stack(features)
        y = torch.tensor(labels, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)

        # Build edge_index and edge_weight
        edges = list(G.edges(data=True))
        edge_index = []
        edge_weight = []

        for u, v, data in edges:
            edge_index.append([node_id_map[u], node_id_map[v]])
            edge_weight.append(data.get('weight', 1.0))  # default weight 1

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

        # Build PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        data.train_mask = train_mask

        # Define and train the model
        model = GCN(in_channels=x.shape[1], hidden_channels=self.hidden_channels, out_channels=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        min_loss = None
        curr_patience = 0
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_weight)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

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

        # Predict labels for all nodes
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_weight)
            probs, pred = torch.softmax(out, 1).max(dim=1)

        # Return mapping from node id to predicted label
        node_predictions = { node: {
                                'label': classes[int(pred[node_id_map[node]].item())], 
                                'prob': probs[node_id_map[node]].item()
                            } for node in node_list }

        return node_predictions
    
    def __call__(self, G):
        predictions = self.propagate(G)
        for node,label_dict in predictions.items():
            label = label_dict['label']
            prob = label_dict['prob']
            if not G.nodes[node].get('class'):
                G.nodes[node]['class'] = {'label': label, 'source': 'lm_propagator', 'prob': prob}
        
        return G
    
class LMNodeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels[idx] is not None:  # labeled example
            item["labels"] = torch.tensor(self.labels[idx])
        return item

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

    def propagate(self, G):
        # Collect node texts and labels
        node_list = list(G.nodes())
        texts = []
        labels = []

        classes = list(set([G.nodes[n].get('class') for n in G.nodes if G.nodes[n].get('class')]))
        label2id = {name: i for i, name in enumerate(classes)}
        id2label = {i: name for name, i in label2id.items()}

        for n in node_list:
            texts.append(G.nodes[n].get("text", ""))  # assumes "text" field exists
            label = G.nodes[n].get('class')
            if label is not None:
                labels.append(label2id[label])
            else:
                labels.append(None)

        # Split into labeled and unlabeled
        train_texts = [t for t, l in zip(texts, labels) if l is not None]
        train_labels = [l for l in labels if l is not None]

        # Tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
        )

        # Build datasets
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        full_dataset = Dataset.from_dict({
            "text": texts,
            "label": [l if l is not None else -100 for l in labels]  # -100 ignored in loss
        })

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        full_dataset = full_dataset.map(tokenize_fn, batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        full_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Training arguments
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

        # Train
        trainer.train()

        # Predict
        preds = trainer.predict(full_dataset)
        probs, pred = preds.predictions.max(axis=1)

        node_predictions = {}
        for node, prob, pred_id, gold in zip(node_list, probs, pred, labels):
            if gold is None:
                node_predictions[node] = {'label': id2label[int(pred_id)], 'prob': prob}

        return node_predictions

    def __call__(self, G):
        predictions = self.propagate(G)
        for node, label_dict in predictions.items():
            label = label_dict['label']
            prob = label_dict['prob']
            if not G.nodes[node].get('class'):
                G.nodes[node]['class'] = {'label': label, 'source': 'lm_propagator', 'prob': prob}
        return G