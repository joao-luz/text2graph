# text2graph

A unified pipeline for text classification through graphs, for when there are no
labeled examples: an LLM labels a few documents, and graph models spread those
labels to the rest.

A `Pipeline` runs a sequence of components over a shared context dict. A typical
pipeline turns documents into nodes, embeds them, connects similar ones, picks a
few to label with one or several LLMs, and propagates those labels to the rest of
the graph with a GNN.

<!-- --8<-- [start:install] -->
## Installation

Straight from GitHub:

```bash
pip install "git+https://github.com/joao-luz/text2graph.git"
```

Or from a clone of the repository, in its root:

```bash
git clone https://github.com/joao-luz/text2graph.git
cd text2graph
pip install .
```

The LLM backends are optional extras:

```bash
pip install ".[vllm]"        # local models with vLLM (needs a GPU)
pip install ".[openai]"      # OpenAI or any OpenAI-compatible server
pip install ".[anthropic]"   # Anthropic
```

With an install from GitHub, add them like this:

```bash
pip install "text2graph[openai] @ git+https://github.com/joao-luz/text2graph.git"
```
<!-- --8<-- [end:install] -->

<!-- --8<-- [start:quickstart] -->
## Quickstart

Classify 2,000 AG News articles into 4 topics, with an LLM labeling only 50 of
them. This example uses the OpenAI API (`pip install ".[openai]"` and set
`OPENAI_API_KEY`).

```python
import logging

from datasets import load_dataset
from sklearn.metrics import classification_report

from text2graph import LLM, Pipeline
from text2graph.components import (
    DocumentsToNodes, SentenceEmbeddingExtractor, KNNEdgeCreator,
    KMeansSampler, LLMLabeler, GCNPropagator,
)

logging.basicConfig(level=logging.WARNING)
logging.getLogger('text2graph').setLevel(logging.INFO)  # show each step's progress

dataset = load_dataset('fancyzhx/ag_news', split='test[:2000]')
texts, labels = list(dataset['text']), list(dataset['label'])
classes = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}

prompt = (
    'Classify this news article into one of these topics:\n'
    '0) World\n1) Sports\n2) Business\n3) Sci/Tech\n\n'
    'Article: {text}\n\n'
    'Answer with the topic number only.'
)

llm = LLM('openai:gpt-5-mini')

pipeline = Pipeline([
    DocumentsToNodes(),                     # one node per document
    SentenceEmbeddingExtractor(model_path='sentence-transformers/all-MiniLM-L6-v2'),
    KNNEdgeCreator(k=10),                   # connect each document to its 10 most similar
    KMeansSampler(n=50),                    # pick 50 representative documents...
    LLMLabeler(model=llm, prompt_template=prompt, label_map=classes,
               temperature=None),           # ...and have the LLM label them
    GCNPropagator(epochs=200),              # train a GCN on those 50, label the rest
])

context = pipeline.run(documents=texts)

predictions = context['graph'].pseudo_y
print(classification_report(labels, predictions, target_names=list(classes.values())))
```

`pipeline.run` returns the context: the graph is in `context['graph']`, each
node's predicted label in its `pseudo_y` attribute, and how each node got its
label in `context['label_info']`. Embeddings and LLM responses are saved in
`./cache`; pass `load_from_cache=True` to `SentenceEmbeddingExtractor` and
`LLMLabeler` to reuse them in later runs instead of recomputing them.

`temperature=None` sends no temperature, which OpenAI's reasoning models need;
with other models, leave it out to use the labeler's default of 0.

The same pipeline can be described as a config dict, with `${...}` referring to
the keyword arguments given to `Pipeline`:

```python
config = {'steps': [
    {'name': 'documents_to_nodes'},
    {'name': 'sentence_embedding_extractor',
     'parameters': {'model_path': 'sentence-transformers/all-MiniLM-L6-v2'}},
    {'name': 'knn_edge_creator', 'parameters': {'k': 10}},
    {'name': 'kmeans_sampler', 'parameters': {'n': 50}},
    {'name': 'llm_labeler', 'parameters': {
        'model': '${llm}', 'prompt_template': '${prompt}',
        'label_map': '${classes}', 'temperature': None}},
    {'name': 'gcn_propagator', 'parameters': {'epochs': 200}},
]}

pipeline = Pipeline(config=config, llm=llm, prompt=prompt, classes=classes)
context = pipeline.run(documents=texts)
```

Or kept in a YAML file, `pipeline.yaml`:

```yaml
steps:
  - name: documents_to_nodes
  - name: sentence_embedding_extractor
    parameters:
      model_path: sentence-transformers/all-MiniLM-L6-v2
  - name: knn_edge_creator
    parameters:
      k: 10
  - name: kmeans_sampler
    parameters:
      n: 50
  - name: llm_labeler
    parameters:
      model: ${llm}
      prompt_template: ${prompt}
      label_map: ${classes}
      temperature: null
  - name: gcn_propagator
    parameters:
      epochs: 200
```

```python
pipeline = Pipeline.from_yaml('pipeline.yaml', llm=llm, prompt=prompt, classes=classes)
context = pipeline.run(documents=texts)
```
<!-- --8<-- [end:quickstart] -->

## Paper

This was, originally, a repository for the paper titled "Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios". If you wish to see the paper's repo, change into branch `paper`

Link to paper: https://ieeexplore.ieee.org/abstract/document/11264654

If you use this pipeline, cite our work as:

```bibtex
@INPROCEEDINGS{11264654,
  author={Sarcinelli, João Lucas Luz Lima and Marcacini, Ricardo Marcondes},
  booktitle={2025 IEEE/SBC 37th International Symposium on Computer Architecture and High Performance Computing Workshops (SBAC-PADW)}, 
  title={Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios}, 
  year={2025},
  volume={},
  number={},
  pages={124-130},
  keywords={Sentiment analysis;Energy consumption;Costs;Annotations;Large language models;High performance computing;Text categorization;Zero shot learning;Feature extraction;Graph neural networks;Large Language Models (LLMs);Graph Neural Networks (GNNs);Zero-Shot Learning;Text-to-Graph;Sustainable AI},
  doi={10.1109/SBAC-PADW69789.2025.00025}}
```
