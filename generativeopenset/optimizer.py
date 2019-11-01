#!/usr/bin/env python
"""
Usage:
    optimizer.py [<eval_type> <metric_name> <dataset>]
    
Runs and evaluates new models, iteratively updating parameters to optimize the given metric.

Arguments:
    eval_type: One of train, test, openset, etc.
    metric_name: One of accuracy, MSE, auc, etc.
    dataset: Eg. mnist, cifar10
"""
import os
import sys
import random
import json
import numpy as np
from docopt import docopt
from pprint import pprint


RESULTS_DIR = '/mnt/nfs/experiments'
DATA_DIR = '/mnt/data'
PATIENCE_SEC = 10

def run_new_experiment(params):
    print("TODO: run new experiment with params:")
    pprint(params)
    """
    result_dir = params['result_dir']
    os.mkdir(result_dir)
    with open(os.path.join(result_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, indent=2))
    cmd = "experiments/train_aac.py --result_dir {}".format(result_dir)
    """

def get_result_dirs():
    return [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]


def get_results(result_dir, epoch):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    if os.path.exists(filename):
        eval_folds = json.load(open(filename))
        return eval_folds
    return {}


def get_params(result_dir):
    filename = 'params.json'
    filename = os.path.join(RESULTS_DIR, result_dir, filename)
    return json.load(open(filename))


def get_editable_params(result_dir):
    # TODO: Allow optimization of discrete parameters
    editable_params = {}
    params = get_params(result_dir)
    for name in params:
        if type(params[name]) is float or name in ['latent_size', 'epochs']:
            editable_params[name] = params[name]
    return editable_params


def perturb_editable_params(editable_params):
    # Select one parameter at random and move it up or down
    name = random.choice(list(editable_params.keys()))
    multiplier = 1 + np.random.uniform(-.1, .1)
    param_type = type(editable_params[name])
    new_val = param_type(editable_params[name] * multiplier)
    print("Setting {} to {}".format(name, new_val))
    editable_params[name] = new_val
    return editable_params


def get_dataset_name(result_dir):
    params = get_params(result_dir)
    dataset = params['dataset']
    return dataset.split('/')[-1].replace('.dataset', '')


def epoch_from_filename(filename):
    numbers = filename.split('epoch_')[-1].rstrip('.pth').replace('.json', '')
    return int(numbers)


def is_valid_directory(result_dir):
    result_dir = os.path.join(RESULTS_DIR, result_dir)
    if not os.path.exists(result_dir) or not os.path.isdir(result_dir):
        return False
    if 'params.json' not in os.listdir(result_dir):
        return False
    dirs = os.listdir(result_dir)
    if 'robotno' in dirs or 'norobot' in dirs:
        print("Found robotno in {}, skipping".format(result_dir))
        return False
    return True


def get_epochs(result_dir):
    filenames = os.listdir(os.path.join(RESULTS_DIR, result_dir))
    pth_names = [f for f in filenames if f.startswith('eval') and f.endswith('.json')]
    return sorted(list(set(epoch_from_filename(f) for f in pth_names)))


def get_all_info(fold, metric, dataset):
    info = []
    for result_dir in get_result_dirs():
        #print('{}...'.format(result_dir), end='')
        if not is_valid_directory(result_dir):
            #print('invalid directory')
            continue
        if get_dataset_name(result_dir) != dataset:
            #print("bad dataset name {} does not match {}".format(get_dataset_name(result_dir), dataset))
            continue
        # Evaluate the most recent epoch
        epochs = get_epochs(result_dir)
        if not epochs:
            #print("no epochs")
            continue
        best_epoch, best_results = None, None
        for epoch in epochs:
            results = get_results(result_dir, epoch)
            if not results:
                continue
            if fold not in results:
                continue
            if best_results is None or results[fold][metric] > best_results[fold][metric]:
                best_epoch = epoch
                best_results = results
        if best_epoch is None:
            #print("no results")
            continue
        info.append((result_dir, best_results[fold][metric], best_epoch))
    info.sort(key=lambda x: x[1])
    return info


def start_new_job():
    dataset = 'svhn-04'
    fold = 'openset_svhn-59_openset'
    metric = 'auc_discriminator'

    if len(sys.argv) > 1:
        fold = sys.argv[1]
    if len(sys.argv) > 2:
        metric = sys.argv[2]
    if len(sys.argv) > 3:
        dataset = sys.argv[3]

    infos = get_all_info(fold=fold, metric=metric, dataset=dataset)
    if not infos:
        print("Error: No runs found for dataset {}".format(dataset))
        return

    print('{:<24} {:<8} {:>12} {:<64}'.format("Experiment", 'Epoch', metric, 'Hypothesis'))
    for (name, metric_val, epoch) in infos:
        params = json.load(open(os.path.join(RESULTS_DIR, name, 'params.json')))
        print('{:<24} {:<8} {:<12.4f} {:<64}'.format(name, epoch, metric_val, params['hypothesis']))

    best_result_dir, best_score, best_epochs = infos[-1]
    best_params = get_params(best_result_dir)

    print("The best {} run so far was {} with {} = {}".format(
        dataset, best_result_dir, metric, best_score))

    print("Parameters for best run:")
    pprint(best_params)

    """
    print("Parameters for next run:")
    new_params = best_params.copy()

    perturbed = perturb_editable_params(get_editable_params(best_result_dir))
    new_params.update(perturbed)

    random_hex = uuid.uuid4().hex[-8:]
    image_size = new_params['image_size']
    new_result_dir = '{0}_{1}x{1}_{2}'.format(dataset, image_size, random_hex)
    new_params['result_dir'] = os.path.join(RESULTS_DIR, new_result_dir)
    pprint(new_params)

    run_new_experiment(new_params)
    """
            

if __name__ == '__main__':
    args = docopt(__doc__)
    start_new_job()
