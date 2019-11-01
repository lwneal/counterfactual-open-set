#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys

def is_true(x):
    return not not x and x.lower().startswith('t')

# Dataset (input) and result_dir (output) are always required
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')

# Other options can change with every run
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--fold', type=str, default='train', help='Fold [default: train]')
parser.add_argument('--start_epoch', type=int, help='Epoch to start from (defaults to most recent epoch)')
parser.add_argument('--count', type=int, default=1, help='Number of counterfactuals to generate')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader
import gen_openmax
from networks import build_networks
from options import load_options


# TODO: Right now, to edit cf_speed et al, you need to edit params.json

start_epoch = options['start_epoch']
options = load_options(options)
options['epoch'] = start_epoch

dataloader = CustomDataloader(**options)

# Batch size must be large enough to make a square grid visual
options['batch_size'] = dataloader.num_classes + 1

networks = build_networks(dataloader.num_classes, **options)

for i in range(options['count']):
    gen_openmax.generate(networks, dataloader, **options)
