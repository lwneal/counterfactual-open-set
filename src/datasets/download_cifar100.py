#!/usr/bin/env python
# Downloads the CIFAR-10 image dataset
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle

DATA_DIR = '/mnt/data'
DATASET_NAME = 'cifar100'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

IMAGES_LABELS_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    download('cifar-100-python.tar.gz', IMAGES_LABELS_URL)

    train_examples = get_examples('train')
    test_examples = get_examples('test')

    dataset_filename = os.path.join(DATA_DIR, 'cifar100.dataset')
    with open(dataset_filename, 'w') as fp:
        for e in train_examples + test_examples:
            fp.write(json.dumps(e) + '\n')
    print('Finished writing dataset {}'.format(dataset_filename))


def get_examples(fold):
    print('Converting CIFAR100 fold {}'.format(fold))
    vals = pickle.load(open('cifar-100-python/{}'.format(fold), 'rb'), encoding='bytes')

    fine_labels = vals[b'fine_labels']
    coarse_labels = vals[b'coarse_labels']
    filenames = vals[b'filenames']
    data = vals[b'data']

    assert len(fine_labels) == len(coarse_labels) == len(filenames)

    examples = []
    for i in tqdm(range(len(filenames))):
        pixels = data[i].reshape((3,32,32)).transpose((1,2,0))
        png_name = str(filenames[i], 'utf-8')
        filename = os.path.join(DATASET_PATH, png_name)
        Image.fromarray(pixels).save(filename)
        examples.append({
            'filename': filename.replace(DATA_DIR, ''),
            'label': fine_labels[i],
            'category': coarse_labels[i],
            'fold': fold
        })
    print('Wrote {} images for fold {}'.format(len(examples), fold))
    return examples


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def listdir(path):
    filenames = os.listdir(os.path.expanduser(path))
    filenames = sorted(filenames)
    return [os.path.join(path, fn) for fn in filenames]


def download(filename, url):
    if os.path.exists(filename):
        print("File {} already exists, skipping".format(filename))
    else:
        # TODO: security lol
        os.system('wget -nc {} -O {}'.format(url, filename))
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')
        elif filename.endswith('.zip'):
            os.system('unzip *.zip')


def train_test_split(filename):
    # Training examples end with 0, test with 1, validation with 2
    return [line.strip().endswith('0') for line in open(filename)]


def save_image_dataset(examples, output_filename):
    fp = open(output_filename, 'w')
    for line in examples:
        fp.write(json.dumps(line) + '\n')
    fp.close()
    print("Wrote {} items to {}".format(len(examples), output_filename))


if __name__ == '__main__':
    main()

