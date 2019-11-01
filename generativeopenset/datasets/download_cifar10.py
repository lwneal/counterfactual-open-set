#!/usr/bin/env python
# Downloads the CIFAR-10 image dataset
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle

DATA_DIR = '/mnt/data'
DATASET_NAME = 'cifar10'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

IMAGES_LABELS_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    download('cifar-10-python.tar.gz', IMAGES_LABELS_URL)

    train_labels = []
    train_filenames = []
    train_data = []
    for i in range(1, 6):
        data_file = 'cifar-10-batches-py/data_batch_{}'.format(i)
        with open(data_file, 'rb') as fp:
            vals = pickle.load(fp, encoding='bytes')
            train_labels.extend(vals[b'labels'])
            train_filenames.extend(vals[b'filenames'])
            train_data.extend(vals[b'data'])

    test_labels = []
    test_filenames = []
    test_data = []
    with open('cifar-10-batches-py/test_batch', 'rb') as fp:
        vals = pickle.load(fp, encoding='bytes')
        test_labels.extend(vals[b'labels'])
        test_filenames.extend(vals[b'filenames'])
        test_data.extend(vals[b'data'])

    examples = []
    for lab, fn, dat in tqdm(zip(train_labels, train_filenames, train_data)):
        example = make_example(lab, fn, dat)
        example['fold'] = 'train'
        examples.append(example)

    for lab, fn, dat in tqdm(zip(test_labels, test_filenames, test_data)):
        example = make_example(lab, fn, dat)
        example['fold'] = 'test'
        examples.append(example)

    print("Saving .dataset files...")
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    save_image_dataset(examples, output_filename)

    animals = [e for e in examples if e['is_animal']]
    animal_filename = '{}/{}-animals.dataset'.format(DATA_DIR, DATASET_NAME)
    save_image_dataset(animals, animal_filename)

    machines = [e for e in examples if not e['is_animal']]
    machine_filename = '{}/{}-machines.dataset'.format(DATA_DIR, DATASET_NAME)
    save_image_dataset(machines, machine_filename)

    splits = [
        [3, 6, 7, 8],
        [1, 2, 4, 6],
        [2, 3, 4, 9],
        [0, 1, 2, 6],
        [4, 5, 6, 9],
    ]

    for idx, split in enumerate(splits):
        unknown_classes = [cifar_class(i) for i in split]
        known_examples = [e for e in examples if e['label'] not in unknown_classes]
        unknown_examples = [e for e in examples if e['label'] in unknown_classes]
        known_filename = '{}/{}-split{}a.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        unknown_filename = '{}/{}-split{}b.dataset'.format(DATA_DIR, DATASET_NAME, idx)
        save_image_dataset(known_examples, known_filename)
        save_image_dataset(unknown_examples, unknown_filename)
    print("Finished writing datasets")


def make_example(label, filename, data):
    pixels = data.reshape((3,32,32)).transpose((1,2,0))
    filename = str(filename, 'utf-8')
    Image.fromarray(pixels).save(filename)
    class_name = cifar_class(label)
    return {
            'filename': os.path.join(DATASET_PATH, filename),
            'label': class_name,
            'is_animal': is_animal(class_name),
            'is_flying': is_flying(class_name),
            'is_pet': is_pet(class_name),
    }


def cifar_class(label_idx):
    return CIFAR_CLASSES[label_idx]


def is_animal(label):
    return label in ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']


def is_flying(label):
    return label in ['bird', 'airplane']


def is_pet(label):
    return label in ['dog', 'cat']


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
