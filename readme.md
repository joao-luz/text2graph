# Text2Graph Pipeline

This is a unified pipeline for solving text classification problems via graphs with no labeled instances and using an LLM for pseudo-labeling.

## Installation

To install the package, first create a virtual environment:

```
python3 -m venv .venv
```

Go into the root of the repository and run:

```
pip install . -f https://data.pyg.org/whl/torch-<torch_version>+<cuda_version>.html
```

Notice that this package makes use of [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and its additional libraries. Thus, to correctly install them, the extra wheel information is passed via the `-f` argument during `pip install`. To figure out the correct wheel link to use in your case, look into torch-geometric's [official instructions](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html).

## Running the example

To run the (currently single) example script, go into the `examples/` dir and run:

```
cd src
python3 run_pipeline.py
```

This example runs many pipelines and datasets defined by `.yaml` files in `examples/configs/`. Pipeline and dataset configurations may be set using `.yaml` files. For datasets, you must define a file with the following attributes:

```yaml
name:
  dataset_name
path:
  path/to/dataset_or_huggingface_path
label_feature:
  dataset_feature_for_labels
classes:
  0: supported
  1: classes
  2: in
  3: dataset

prompt_template: |
  Template for the LLM to process a text from the dataset. Should look something like this:

  You are a topic analysis tool for analysing news articles. Look at the following text:

  {text}

  What is the topic related to the text? Choose a topic from the following possible options:

  0) supported
  1) classes
  2) in
  3) dataset

  Respond ONLY with the topic number and no other information.

```
For the pipeline, you must define a name and the components. Each component goes under the `component:` item in a list form (see `examples/configs/pipelines/baseline.yaml` for an in depth example). The component is identified with a name and is given its parameters:

```yaml
dataset:
    name: name
    components:
        - name: first_component
          parameters:
            first_parameter: first_value
            second_parameter: second_value
            ...
```


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