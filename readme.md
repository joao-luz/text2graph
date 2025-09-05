# Text2Graph Pipeline

This is a unified pipeline for solving text classification problems via graphs with no labeled instances and using an LLM for pseudo-labeling.

To run, install dependencies from `requirements.txt` and execute

```bash
cd src
python3 run_pipeline.py --datasets first_dataset second_dataset --pipelines first_pipeline second_pipeline --runs 5
```

Pipeline and dataset configurations may be set using `.yaml` files. For datasets, you must define a file with the following attributes:

```yaml
dataset:
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
For the pipeline, you must define a name and the components. Each component goes under the `component:` item in a list form (see `configs/pipelines/baseline.yaml` for an in depth example). The component is identified with a name and is given its parameters:

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
