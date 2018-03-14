#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint

# Print --help message before importing the rest of the project
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--fold', default="test", help='Name of evaluation fold [default: test]')
parser.add_argument('--epoch', type=int, help='Epoch to evaluate (latest epoch if none chosen)')
parser.add_argument('--comparison_dataset', type=str, help='Dataset for off-manifold comparison')
parser.add_argument('--aux_dataset', type=str, help='aux_dataset used in train_classifier')
parser.add_argument('--mode', default='', help='One of: default, weibull, weibull-kplus1, baseline')
parser.add_argument('--roc_output', type=str, help='Optional filename for ROC data output')
options = vars(parser.parse_args())

# Import the rest of the project
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
from networks import build_networks
from options import load_options, get_current_epoch
from evaluation import save_evaluation
from comparison import evaluate_with_comparison

options = load_options(options)
if not options.get('epoch'):
    options['epoch'] = get_current_epoch(options['result_dir'])
# TODO: Globally disable dataset augmentation during evaluation
options['random_horizontal_flip'] = False

dataloader = CustomDataloader(last_batch=True, shuffle=False, **options)

# TODO: structure options in a way that doesn't require this sort of hack
train_dataloader_options = options.copy()
train_dataloader_options['fold'] = 'train'
dataloader_train = CustomDataloader(last_batch=True, shuffle=False, **train_dataloader_options)

networks = build_networks(dataloader.num_classes, **options)

new_results = evaluate_with_comparison(networks, dataloader, dataloader_train=dataloader_train, **options)

pprint(new_results)
save_evaluation(new_results, options['result_dir'], options['epoch'])
