#!/usr/bin/env python
import argparse
import os
import sys

def is_true(x):
    return not not x and x not in ['false', 'False', '0']

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints')
parser.add_argument('--dataset', required=True, help='Input filename (must be in .dataset format)')
parser.add_argument('--hypothesis', required=True, help='A helpful description so you can remember why you trained this network')
parser.add_argument('--latent_size', type=int, default=128, help='Size of the latent z vector [default: 128]')
parser.add_argument('--image_size', type=int, default=32, help='Height / width of images [default: 32]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--random_horizontal_flip', type=is_true, default=False, help='Flip images during training. [default: False]')
parser.add_argument('--delete_background', type=is_true, default=False, help='Delete non-foreground pixels from images [default: False]')

options = vars(parser.parse_args())

# TODO: more principled, with configuration
if options.get('result_dir') is None:
    dataset_name = options['dataset'].split('/')[-1].split('.')[0]
    import uuid
    random_hex = uuid.uuid4().hex[:8]
    options['result_dir'] = '/mnt/nfs/experiments/{}_{}'.format(
            dataset_name, random_hex)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import FlexibleCustomDataloader
from networks import build_networks, get_optimizers
from options import save_options

dataloader = FlexibleCustomDataloader(fold='train', **options)
networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, **options)
save_options(options)

# Use git ls-files to copy all files in the repository to the destination directory
from repo import copy_repo
copy_repo(options['result_dir'])
