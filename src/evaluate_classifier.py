#!/usr/bin/env python
import argparse
import json
import os
import sys
from pprint import pprint

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--epoch', type=int, help='Epoch to evaluate (latest epoch if none chosen)')
parser.add_argument('--save_latent_vectors', default=False, help='Save Z in .npy format for later visualization')
parser.add_argument('--comparison_dataset', type=str, help='Dataset for off-manifold comparison')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import evaluate_classifier, evaluate_openset, save_evaluation

options = load_options(options)
if not options.get('epoch'):
    options['epoch'] = get_current_epoch(options['result_dir'])
options['random_horizontal_flip'] = False

dataloader = CustomDataloader(last_batch=True, shuffle=False, **options)

networks = build_networks(dataloader.num_classes, **options)

comparison_dataloader = None
if options['comparison_dataset']:
    comparison_options = options.copy()
    comparison_options['dataset'] = options['comparison_dataset']
    comparison_dataloader = CustomDataloader(last_batch=True, shuffle=False, **comparison_options)
    comparison_name = options['comparison_dataset'].split('/')[-1].split('.')[0]
    labels_dir = os.path.join(options['result_dir'], 'labels')
    if os.path.exists(labels_dir):
        label_count = len(os.listdir(labels_dir))
    else:
        label_count = 0
    # Hack: ignore the label count
    """
    options['fold'] = 'openset_{}_{:04d}'.format(comparison_name, label_count)
    """
    options['fold'] = 'openset_{}'.format(comparison_name)

new_results = evaluate_classifier(networks, dataloader, comparison_dataloader, **options)
if options['comparison_dataset']:
    openset_results = evaluate_openset(networks, dataloader, comparison_dataloader, **options)
    pprint(openset_results)
    new_results[options['fold'] + '_openset'] = openset_results
    new_results[options['fold']]['active_learning_label_count'] = label_count


save_evaluation(new_results, options['result_dir'], options['epoch'])

