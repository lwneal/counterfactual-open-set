#!/usr/bin/env python
# Downloads the EMNIST letters dataset
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from scipy import io as sio

DATA_DIR = '/mnt/data'
DATASET_NAME = 'emnist'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

IMAGES_LABELS_URL = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
# Alternative format matching the original MNIST
#IMAGES_LABELS_URL = 'http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    download('matlab.zip', IMAGES_LABELS_URL)

    print("Converting EMNIST letters...")
    mkdir(os.path.join(DATASET_PATH, 'letters'))
    mat_filename = os.path.join(DATASET_PATH, 'matlab', 'emnist-letters.mat')

    mat = sio.loadmat(mat_filename)
    dataset = mat['dataset'][0][0]

    # TODO: Convert matlab-format data into numpy matrices
    train_digits = dataset[0][0][0][0]
    train_labels = dataset[0][0][0][1]

    test_digits = dataset[1][0][0][0]
    test_labels = dataset[1][0][0][1]

    print("Converting EMNIST letters training set...")
    train_examples = convert_emnist(train_digits, train_labels, fold='train')

    print("Converting EMNIST letters testing set...")
    test_examples = convert_emnist(test_digits, test_labels, fold='test')

    print("Saving .dataset file...")
    save_image_dataset(train_examples + test_examples)
    print("Dataset convertion finished")


def convert_emnist(digits, labels, fold):
    examples = []
    assert len(digits) == len(labels)
    for i in tqdm(range(len(digits))):
        label = chr(64 + labels[i])
        filename = 'letter_{:06d}.png'.format(i)
        filename = os.path.join(DATASET_PATH, 'letters', filename)
        pixels = mnist_to_np(digits[i])
        Image.fromarray(pixels).save(filename)
        examples.append({
            "filename": filename,
            "fold": fold,
            "label": label
        })
    return examples


def mnist_to_np(raw):
    return raw.reshape((28,28)).transpose()


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


def save_image_dataset(examples):
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    fp = open(output_filename, 'w')
    for line in examples:
        fp.write(json.dumps(line) + '\n')
    fp.close()
    print("Wrote {} items to {}".format(len(examples), output_filename))


if __name__ == '__main__':
    main()
