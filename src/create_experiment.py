#!/usr/bin/env python
import argparse
import os
import sys

def is_true(x):
    return not not x and x not in ['false', 'False', '0']

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints')
parser.add_argument('--dataset', required=True, help='Input filename (must be in .dataset format)')
parser.add_argument('--hypothesis', required=True, help='A helpful description so that a month from now, you can remember why you ran this experiment.')

parser.add_argument('--latent_size', type=int, default=32, help='Size of the latent z vector [default: 32]')
parser.add_argument('--image_size', type=int, default=32, help='Height / width of images [default: 32]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size [default: 64]')
parser.add_argument('--random_horizontal_flip', type=is_true, default=False, help='Flip images during training. [default: False]')
parser.add_argument('--delete_background', type=is_true, default=False, help='Delete non-foreground pixels from images [default: False]')

parser.add_argument('--weight_decay', type=float, default=.0, help='Optimizer L2 weight decay [default: .0]')
parser.add_argument('--beta1', type=float, default=0.5, help='Optimizer beta1 for adam. [default: 0.5]')

parser.add_argument('--generator_weight', type=float, default=.001, help='Multiplier for generator adversarial loss [default: .001]')
parser.add_argument('--discriminator_weight', type=float, default=.001, help='Multiplier for discriminator adversarial loss [default: .001]')
parser.add_argument('--reconstruction_weight', type=float, default=1.0, help='Multiplier for mean-abs pixel error loss [default: 1.0]')

parser.add_argument('--cf_speed', type=float, default=.1, help='Learning rate for counterfactual descent [default: .1]')
parser.add_argument('--cf_max_iters', type=int, default=100, help='Maximum number of steps to take for CF trajectories [default: 100]')
parser.add_argument('--cf_distance_weight', type=float, default=1, help='Weight for latent distance loss [default: 1]')
parser.add_argument('--cf_gan_scale', type=int, default=4, help='Scale, for multiscale GAN [default: 4]')

parser.add_argument('--comparison_dataset', help='Optional comparison dataset for open set evaluation [default: None]')

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
