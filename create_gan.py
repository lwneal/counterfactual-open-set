#!/usr/bin/env python
import argparse
import os
import sys

def is_true(x):
    return not not x and x not in ['false', 'False', '0']

# Dataset (input) and result_dir (output) must always be specified
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='Output directory for images and model checkpoints')
parser.add_argument('--dataset', help='Input filename (must be in .dataset format)')

# Other options have default values
parser.add_argument('--latent_size', type=int, default=100, help='Size of the latent z vector [default: 100]')
parser.add_argument('--image_size', type=int, default=32, help='Height / width of images [default: 32]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--random_horizontal_flip', type=is_true, default=False, help='Flip images during training. [default: False]')
parser.add_argument('--delete_background', type=is_true, default=False, help='Delete non-foreground pixels from images [default: False]')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import FlexibleCustomDataloader
from networks import build_networks, get_optimizers
from options import save_options

dataloader = FlexibleCustomDataloader(fold='train', **options)
networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)
save_options(options)
