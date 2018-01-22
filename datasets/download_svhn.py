#!/usr/bin/env python
# Downloads the CelebA face dataset
import os
import numpy as np
import json
from subprocess import check_output
from scipy import io as sio
from tqdm import tqdm

from PIL import Image
import matplotlib
matplotlib.use('Agg')


import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from imutil import show

DOWNLOAD_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'
DOWNLOAD_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'

DOWNLOAD_URL_TRAIN_MAT = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DOWNLOAD_URL_TEST_MAT = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'


DATA_DIR = '/mnt/data'
DATASET_NAME = 'svhn'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

def from_mat(mat, img_dir, fold):
    data  = mat['X']
    labels = mat['y']
    examples = []
    for i in tqdm(range(len(labels))):
        img = data[:,:,:,i]
        label = labels[i][0]
        if label == 10:
            label = 0
        filename = os.path.join(img_dir, "{}_{:06d}.jpg".format(fold, i))
        Image.fromarray(img).save(filename)
        examples.append({
            'fold': fold,
            'label': str(label),
            'filename': filename,
        })
    return examples


def main():
    print("{} dataset download script initializing...".format(DATASET_NAME))
    mkdir(DATA_DIR)
    mkdir(DATASET_PATH)
    os.chdir(DATASET_PATH)

    print("Downloading {} dataset files...".format(DATASET_NAME))

    #download('train.tar.gz', DOWNLOAD_URL_TRAIN)
    #download('test.tar.gz', DOWNLOAD_URL_TEST)
    download('train_32x32.mat', DOWNLOAD_URL_TRAIN_MAT)
    download('test_32x32.mat', DOWNLOAD_URL_TEST_MAT)

    
    #os.chdir(DATASET_PATH + '/train')
    train_mat = sio.loadmat('train_32x32.mat')
    test_mat = sio.loadmat('test_32x32.mat')

    test_data = test_mat['X']
    test_labels = test_mat['y']

    IMG_DIR = os.path.join(DATASET_PATH, 'images')
    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
    examples = from_mat(train_mat, IMG_DIR, 'train') + from_mat(test_mat, IMG_DIR, 'test')

    # Generate CSV file for the full dataset
    save_examples(examples)
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



def save_examples(examples):
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    print("Writing {} items to {}".format(len(examples), output_filename))

    fp = open(output_filename, 'w')
    for ex in examples:
        fp.write(json.dumps(ex) + '\n')
    fp.close()


if __name__ == '__main__':
    main()
