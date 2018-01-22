#!/usr/bin/env python
# Downloads the CUB200_2011 dataset
# (Roughly double the number of images per class of the original CUB200)
import os
import numpy as np
import json
from subprocess import check_output


DATA_DIR = '/mnt/data'
CUB_DIR = os.path.join(DATA_DIR, 'cub200_2011')
DATASET_NAME = 'cub200_2011'


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


def save_dataset(examples):
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    print("Writing {} items to {}".format(len(examples), output_filename))
    fp = open(output_filename, 'w')
    for example in examples:
        fp.write(json.dumps(example) + '\n')
    fp.close()


def train_test_split(filename='train_test_split.txt'):
    # Training examples end with 1, test with 0
    return [line.endswith('1\n') for line in open(filename)]


def get_attribute_names(filename='attributes.txt'):
    lines = open(filename).readlines()
    idx_to_name = {}
    for line in lines:
        idx, name = line.split()
        idx_to_name[int(idx)] = name
    return idx_to_name


def parse_attributes(filename):
    names = get_attribute_names()
    lines = open(filename).readlines()
    examples = {}
    for line in lines:
        tokens = line.split()
        # Note that the array starts at 1
        example_idx = int(tokens[0]) - 1
        if example_idx not in examples:
            examples[example_idx] = {}
        # Index into attribute names table
        attr_idx = int(tokens[1])
        # Value: 0 or 1
        attr_value = int(tokens[2])
        # Certainty Values
        # 1 not visible
        # 2 guessing
        # 3 probably
        # 4 definitely
        attr_certainty = int(tokens[3])
        # How many seconds the turker took
        attr_time = float(tokens[4])
        attr_name = names[attr_idx]
        if attr_name in examples[example_idx]:
            print("Warning: Double-entry for example {} attribute {}".format(
                example_idx, attr_name))
        examples[example_idx][attr_name] = attr_value
    # Format into a list with one entry per example
    return [examples[i] for i in range(len(examples))]
        


if __name__ == '__main__':
    print("CUB200_2011 dataset download script initializing...")
    mkdir(DATA_DIR)
    mkdir(CUB_DIR)
    os.chdir(CUB_DIR)

    # Download and extract dataset
    print("Downloading CUB200_2011 dataset files...")
    download('images', 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz')
    download('segmentations', 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz')
    download('annotations', 'https://lwneal.com/cub200_2011_txt_annotations.tar.gz')

    if os.path.exists('CUB_200_2011'):
        os.system('mv CUB_200_2011/* . && rmdir CUB_200_2011')

    # Generate CSV file for the full dataset
    lines = open('images.txt').readlines()
    ids = [int(line.split()[0]) for line in lines]
    image_filenames = [line.split()[1] for line in lines]

    print("Loading CUB200 bounding boxes...")
    boxes = open('bounding_boxes.txt').readlines()
    boxes = [[float(w) for w in line.split()[1:]] for line in boxes]

    print("Parsing CUB200 attributes...")
    attributes = parse_attributes('attributes/image_attribute_labels.txt')

    print("Parsing train/test split...")
    is_training = train_test_split()

    examples = []
    for i in range(len(image_filenames)):
        example = attributes[i].copy()

        example['filename'] = '~/data/cub200_2011/images/{}'.format(image_filenames[i])
        example['segmentation'] = '~/data/cub200_2011/segmentations/{}'.format(image_filenames[i]).replace('jpg','png')

        width, height = get_width_height(example['filename'])
        left, top, box_width, box_height = boxes[i]
        x0 = left / width
        x1 = (left + box_width) / width
        y0 = top / height
        y1 = (top + box_height) / height
        example['box'] = (x0, x1, y0, y1)

        example['label'] = image_filenames[i].lstrip('.0123456789').split('/')[0]

        example['fold'] = 'train' if is_training[i] else 'test'

        examples.append(example)

    save_dataset(examples)

    print("Finished building CUB200_2011 dataset")
