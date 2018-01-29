#!/usr/bin/env python
# Downloads the CelebA face dataset
import os
import numpy as np
import json
from subprocess import check_output


DOWNLOAD_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1'
PARTITION_URL =  'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxLE5t6HqyD8sQCmzWJRcHa/Eval/list_eval_partition.txt?dl=1'
ANNO_URL = 'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1'

DATA_DIR = '/mnt/data'
DATASET_NAME = 'celeba'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files...".format(DATASET_NAME))

    download('img_align_celeba.zip', DOWNLOAD_URL)
    download('list_eval_partition.txt', PARTITION_URL)
    download('list_attr_celeba.txt', ANNO_URL)

    images = listdir(os.path.join(DATASET_PATH, 'img_align_celeba'))

    # Read training/test split
    is_training = train_test_split('list_eval_partition.txt')
    assert len(is_training) == len(images)

    examples = []
    raw_attrs = load_attributes('list_attr_celeba.txt')
    for attr, is_training in zip(raw_attrs, is_training):
        example = {'is_' + k.lower(): v for (k, v) in attr.items()}
        example['fold'] = 'train' if is_training else 'test'
        examples.append(example)

    # Generate CSV file for the full dataset
    save_image_dataset(images, examples)
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


def load_attributes(filename):
    attrs = []
    lines = open(filename).readlines()
    columns = lines[1].split()
    for line in lines[2:]:
        attr = {}
        booleans = [word == '1' for word in line.split()[1:]]
        for attr_name, attr_val in zip(columns, booleans):
            attr[attr_name] = attr_val
        attrs.append(attr)
    return attrs


def save_image_dataset(images, attributes):
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    print("Writing {} items to {}".format(len(images), output_filename))

    fp = open(output_filename, 'w')
    for filename, attrs in zip(images, attributes):
        line = attrs
        line['filename'] = filename
        line['label'] = attrs['is_young']
        fp.write(json.dumps(line) + '\n')
    fp.close()


if __name__ == '__main__':
    main()
