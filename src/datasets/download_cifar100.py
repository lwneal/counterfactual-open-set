#!/usr/bin/env python
# Downloads the CIFAR-10 image dataset
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import pickle

DATA_DIR = '/mnt/nfs/data'
DATASET_NAME = 'cifar100'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

IMAGES_LABELS_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

ANIMAL_CATEGORIES = [
  "aquatic_mammals",
  "fish",
  "insects",
  "large_carnivores",
  "large_omnivores_and_herbivores",
  "medium_mammals",
  "non-insect_invertebrates",
  "people",
  "reptiles",
  "small_mammals",
]


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files to {}...".format(DATASET_NAME, DATASET_PATH))
    download('cifar-100-python.tar.gz', IMAGES_LABELS_URL)

    examples = get_examples('train') + get_examples('test')

    dataset_filename = os.path.join(DATA_DIR, 'cifar100.dataset')
    with open(dataset_filename, 'w') as fp:
        for e in tqdm(examples):
            fp.write(json.dumps(e) + '\n')
    print('Finished writing dataset {}'.format(dataset_filename))

    animal_examples = [e for e in examples if e['category'] in ANIMAL_CATEGORIES]
    object_examples = [e for e in examples if e['category'] not in ANIMAL_CATEGORIES]

    print('Writing animals/objects splits...')
    with open(os.path.join(DATA_DIR, 'cifar100-animals.dataset'), 'w') as fp:
        for e in tqdm(animal_examples):
            fp.write(json.dumps(e) + '\n')

    with open(os.path.join(DATA_DIR, 'cifar100-not-animals.dataset'), 'w') as fp:
        for e in tqdm(object_examples):
            fp.write(json.dumps(e) + '\n')

    print('Writing label subset splits...')
    meta = pickle.load(open('cifar-100-python/meta', 'rb'))
    coarse_label_names = meta['coarse_label_names']
    fine_label_names = meta['fine_label_names']

    animal_labels = sorted(set(e['label'] for e in animal_examples))
    object_labels = sorted(set(e['label'] for e in object_examples))

    # From animals, select 10, 20, ... labels
    with open(os.path.join(DATA_DIR, 'cifar100-animals-10.dataset'), 'w') as fp:
        for e in tqdm(e for e in examples if e['label'] in animal_labels[:10]):
            fp.write(json.dumps(e) + '\n')
    with open(os.path.join(DATA_DIR, 'cifar100-animals-20.dataset'), 'w') as fp:
        for e in tqdm(e for e in examples if e['label'] in animal_labels[:20]):
            fp.write(json.dumps(e) + '\n')
    with open(os.path.join(DATA_DIR, 'cifar100-animals-50.dataset'), 'w') as fp:
        for e in tqdm(e for e in examples if e['label'] in animal_labels[:50]):
            fp.write(json.dumps(e) + '\n')

    # From non-animals, select 10, 20, ... labels
    with open(os.path.join(DATA_DIR, 'cifar100-not-animals-10.dataset'), 'w') as fp:
        for e in tqdm(e for e in examples if e['label'] in object_labels[:10]):
            fp.write(json.dumps(e) + '\n')
    with open(os.path.join(DATA_DIR, 'cifar100-not-animals-20.dataset'), 'w') as fp:
        for e in tqdm(e for e in examples if e['label'] in object_labels[:20]):
            fp.write(json.dumps(e) + '\n')
    with open(os.path.join(DATA_DIR, 'cifar100-not-animals-50.dataset'), 'w') as fp:
        for e in tqdm(e for e in examples if e['label'] in object_labels[:50]):
            fp.write(json.dumps(e) + '\n')

    print('Finished writing all datasets')




def get_examples(fold):
    print('Converting CIFAR100 fold {}'.format(fold))
    vals = pickle.load(open('cifar-100-python/{}'.format(fold), 'rb'), encoding='bytes')

    meta = pickle.load(open('cifar-100-python/meta', 'rb'))
    coarse_label_names = meta['coarse_label_names']
    fine_label_names = meta['fine_label_names']

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

        fine_label = fine_label_names[fine_labels[i]]
        category = coarse_label_names[coarse_labels[i]]
        examples.append({
            'filename': filename.replace(DATA_DIR + '/', ''),
            'label': fine_label,
            'category': category,
            'fold': fold,
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

