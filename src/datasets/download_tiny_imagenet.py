#!/usr/bin/env python
import os
import numpy as np
import json
from subprocess import check_output


DATA_ROOT_DIR = '/mnt/data'
DATASET_DIR = os.path.join(DATA_ROOT_DIR, 'tiny_imagenet')
DATASET_NAME = 'tiny_imagenet'


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        os.system('wget -nc {}'.format(url))
        if url.endswith('.tgz') or url.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')


def get_width_height(filename):
    from PIL import Image
    img = Image.open(os.path.expanduser(filename))
    return (img.width, img.height)


def save_dataset(examples, output_filename):
    print("Writing {} items to {}".format(len(examples), output_filename))
    fp = open(output_filename, 'w')
    for example in examples:
        fp.write(json.dumps(example) + '\n')
    fp.close()


if __name__ == '__main__':
    print("Downloading dataset {}...".format(DATASET_NAME))
    mkdir(DATA_ROOT_DIR)
    mkdir(DATASET_DIR)
    os.chdir(DATASET_DIR)

    # Download and extract dataset
    print("Downloading dataset files...")
    download('tiny-imagenet-200', 'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    # Remove extra directory
    os.system('mv tiny-imagenet-200/* . && rmdir tiny-imagenet-200')

    wnids = open('wnids.txt').read().splitlines()

    wnid_names = {}
    for line in open('words.txt').readlines():
        wnid, name = line.strip().split('\t')
        wnid_names[wnid] = name

    test_filenames = os.listdir('test/images')

    examples = []

    # Collect training examples
    for wnid in os.listdir('train'):
        filenames = os.listdir(os.path.join('train', wnid, 'images'))
        for filename in filenames:
            file_path = os.path.join(DATASET_NAME, 'train', wnid, 'images', filename)
            examples.append({
                'filename': file_path,
                'label': wnid_names[wnid],
                'fold': 'train',
            })

    # Use validation set as a test set
    for line in open('val/val_annotations.txt').readlines():
        jpg_name, wnid, x0, y0, x1, y1 = line.split()
        examples.append({
            'filename': os.path.join(DATASET_NAME, 'val', 'images', jpg_name),
            'label': wnid_names[wnid],
            'fold': 'test',
        })
    save_dataset(examples, '{}/{}.dataset'.format(DATA_ROOT_DIR, DATASET_NAME))

    # Put the unlabeled test set in a separate dataset
    test_examples = []
    for jpg_name in os.listdir('test/images'):
        test_examples.append({
            'filename': os.path.join(DATASET_NAME, 'test', 'images', jpg_name),
            'fold': 'test',
        })
    save_dataset(test_examples, '{}/{}-unlabeled.dataset'.format(DATA_ROOT_DIR, DATASET_NAME))

    print("Finished building dataset {}".format(DATASET_NAME))
