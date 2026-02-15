import pandas as pd

import copy

from codecarbon import EmissionsTracker
from pathlib import Path
from itertools import product
from datasets import load_dataset
from sklearn.metrics import classification_report
from argparse import ArgumentParser
import yaml

from text2graph.pipeline import Text2Graph

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--datasets', type=str, nargs='+', default=['agnews', 'ohsumed-single', 'r8', 'r52', 'imdb'])
    parser.add_argument('--pipelines', type=str, nargs='+', default=['baseline', 'ground_truth_labels', 'llm_only'])
    parser.add_argument('--config_dir', type=str, default='../configs')
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--skip_visualization', action='store_true', default=False)
    parser.add_argument('--rerun_existing', action='store_true', default=False)

    return parser.parse_args()

def load_from_yaml(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return config

def run_pipeline(dataset_config, pipeline_config, output_dir, skip_visualization):
    pipeline = Text2Graph(config=pipeline_config, skip_visualization=skip_visualization, output_dir=output_dir)
    dataset = load_dataset(dataset_config['dataset']['path'])['test']    

    label_feature = dataset_config['dataset']['label_feature']
    id2label = dataset_config['dataset']['classes']

    true_labels = dataset[label_feature]

    tracker = EmissionsTracker(output_file=f'{output_dir}/emissions.csv')
    tracker.start()
    data = pipeline(dataset['text'], true_labels=true_labels, id2label=id2label)
    tracker.stop()

    pred = data.y

    report = classification_report(true_labels, pred, output_dict=True, target_names=id2label.values())
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'{output_dir}/results.csv')

if __name__ == '__main__':
    args = parse_args()

    combinations = list(product(args.datasets, args.pipelines))
    for dataset_name, pipeline_name in combinations:
        dataset_config = load_from_yaml(f'{args.config_dir}/datasets/{dataset_name.split("/")[-1]}.yaml')
        pipeline_config = load_from_yaml(f'{args.config_dir}/pipelines/{pipeline_name}.yaml')

        # Replace default template with the dataset's template
        template = dataset_config['labeling_prompt_template']
        for component in pipeline_config['pipeline']['components']:
            if component['name'] in ['llm_labeler', 'llm_ensemble_labeler']:
                component['parameters']['prompt_template'] = template
                component['parameters']['parser_args'] = {'options': dataset_config['dataset']['classes']}

        for run in range(args.runs):
            current_config = copy.deepcopy(pipeline_config)
            output_dir = f'{args.output_dir}/{dataset_name.split("/")[-1]}/{pipeline_name}/{run}'

            if not args.rerun_existing and Path(f'{output_dir}/results.csv').is_file():
                continue

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            print(f'Running: pipeline={pipeline_name}, dataset={dataset_name}, run={run} ')
            run_pipeline(dataset_config, current_config, output_dir, args.skip_visualization)