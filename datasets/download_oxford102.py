#!/usr/bin/env python
# Downloads the Oxford 102 flowers dataset
import os
import numpy as np
import json
from subprocess import check_output
from scipy import io as sio

DATA_DIR = '/mnt/data'
DATASET_NAME = 'oxford102'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

IMAGES_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
SEGMENTATION_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz'
LABELS_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
SPLITS_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
README_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt'


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files...".format(DATASET_NAME))

    download('102flowers.tgz', IMAGES_URL)
    download('102segmentations.tgz', SEGMENTATION_URL)
    download('imagelabels.mat', LABELS_URL)
    download('setid.mat', SPLITS_URL)
    download('README.txt', README_URL)

    image_dir = '/mnt/data/oxford102/jpg'
    image_filenames = listdir(image_dir)

    imagelabels_mat = sio.loadmat('imagelabels.mat')
    labels = imagelabels_mat['labels'][0]

    setid_mat = sio.loadmat('setid.mat')
    test_ids = set(setid_mat['trnid'][0])
    train_ids = set(setid_mat['tstid'][0])
    valid_ids = set(setid_mat['valid'][0])

    examples = []
    for filename in image_filenames:
        file_id = int(filename.partition('_')[2][:5])
        fold = 'valid' if file_id in valid_ids else 'train' if file_id in train_ids else 'test'
        seg_filename = filename.replace('jpg/image', 'segmim/segmim')
        examples.append({
            "filename": filename,
            "segmentation": seg_filename,
            "fold": fold,
            "label": int(labels[file_id - 1]),
        })
    save_image_dataset(examples)
    print("Successfully built dataset {}".format(DATASET_PATH))


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
