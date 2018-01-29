#!/usr/bin/env python
import numpy as np
import hashlib
import sys
import requests
import os
import json
from PIL import Image
from tqdm import tqdm

DATA_DIR = '/mnt/data'
DOWNLOAD_URL = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
LATEST_MD5 = ''


def save_set(fold, x, y, suffix='png'):
    examples = []
    fp = open('mnist_{}.dataset'.format(fold), 'w')
    print("Writing MNIST dataset {}".format(fold))
    for i in tqdm(range(len(x))):
        label = y[i]
        img_filename = 'mnist/{}/{:05d}_{:d}.{}'.format(fold, i, label, suffix)
        img_filename = os.path.join(DATA_DIR, img_filename)
        if not os.path.exists(img_filename):
            Image.fromarray(x[i]).save(os.path.expanduser(img_filename))
        entry = {
                'filename': img_filename,
                'label': str(label),
                'fold': fold,
        }
        examples.append(entry)
        fp.write(json.dumps(entry))
        fp.write('\n')
    fp.close()
    return examples


def download_mnist_data(path='mnist.npz'):
    if not os.path.exists(path):
        response = requests.get(DOWNLOAD_URL)
        open(path, 'wb').write(response.content)
    with np.load(path) as f:
        x_test, y_test = f['x_test'], f['y_test']
        VAL_SIZE = 5000
        # Use last 5k examples as a validation set
        x_train, y_train = f['x_train'][:-VAL_SIZE], f['y_train'][:-VAL_SIZE]
        x_val, y_val = f['x_train'][-VAL_SIZE:], f['y_train'][-VAL_SIZE:]
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def mkdir(dirname):
    import errno
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        if not os.path.isdir(dirname):
            raise


def is_latest_version(latest_md5):
    dataset_file = os.path.join(DATA_DIR, 'mnist.dataset')
    if not os.path.exists(dataset_file):
        return False
    data = open(dataset_file, 'rb').read()
    current_md5 = hashlib.md5(data).hexdigest()
    if current_md5 == latest_md5:
        print("Have latest version of MNIST: {}".format(current_md5))
        return True
    else:
        print("Have old version {} of MNIST, downloading version {}".format(current_md5, latest_md5))
        return False

def download_mnist(latest_md5):
    if is_latest_version(latest_md5):
        print("Already have the latest version of mnist.dataset, not downloading")
        return

    (train_x, train_y), (test_x, test_y), (val_x, val_y) = download_mnist_data()

    train = save_set('train', train_x, train_y)
    test = save_set('test', test_x, test_y)
    val = save_set('val', val_x, val_y)
    for example in train:
        example['fold'] = 'train'
    for example in test:
        example['fold'] = 'test'
    for example in val:
        example['fold'] = 'validation'
    with open('mnist.dataset', 'w') as fp:
        for example in train + test + val:
            fp.write(json.dumps(example, sort_keys=True) + '\n')

    # For open set classification experiments
    with open('mnist-05.dataset', 'w') as fp:
        for example in train + test + val:
            if int(example['label']) < 6:
                fp.write(json.dumps(example) + '\n')
    with open('mnist-69.dataset', 'w') as fp:
        for example in train + test + val:
            if int(example['label']) >= 6:
                fp.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    os.chdir(DATA_DIR)
    mkdir('mnist')
    mkdir('mnist/train')
    mkdir('mnist/test')
    mkdir('mnist/val')
    download_mnist(LATEST_MD5)
