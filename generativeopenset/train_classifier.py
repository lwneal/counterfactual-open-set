#!/usr/bin/env python
import argparse
import os
import sys
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', help='Output directory for images and model checkpoints [default: .]', default='.')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for [default: 10]')
parser.add_argument('--aux_dataset', help='Path to aux_dataset file [default: None]')
parser.add_argument('--comparison_dataset', help='Optional comparison dataset for open set evaluation [default: None]')
parser.add_argument('--mode', default='', help='If set to "baseline" use the baseline classifier')

options = vars(parser.parse_args())

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataloader import CustomDataloader, FlexibleCustomDataloader
from training import train_classifier
from networks import build_networks, save_networks, get_optimizers
from options import load_options, get_current_epoch
from comparison import evaluate_with_comparison
from evaluation import save_evaluation

options = load_options(options)
dataloader = FlexibleCustomDataloader(fold='train', **options)
networks = build_networks(dataloader.num_classes, **options)
optimizers = get_optimizers(networks, finetune=True, **options)

eval_dataloader = CustomDataloader(last_batch=True, shuffle=False, fold='test', **options)

start_epoch = get_current_epoch(options['result_dir']) + 1
for epoch in range(start_epoch, start_epoch + options['epochs']):
    train_classifier(networks, optimizers, dataloader, epoch=epoch, **options)
    eval_results = evaluate_with_comparison(networks, eval_dataloader, **options)
    pprint(eval_results)
    save_evaluation(eval_results, options['result_dir'], epoch)
    save_networks(networks, epoch, options['result_dir'])
