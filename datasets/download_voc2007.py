#!/usr/bin/env python
import os
import numpy as np
import json
from subprocess import check_output, Popen
from scipy import io as sio

DATA_DIR = '/mnt/data'
DATASET_NAME = 'voc2007'
DATASET_PATH = os.path.join(DATA_DIR, DATASET_NAME)

TRAIN_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"


class DatasetDownloader():
    def __init__(self, name):
        self.name = name
        print("{} dataset download script initializing...".format(name))
        mkdir(DATA_DIR)
        mkdir(self.name)
        os.chdir(os.path.join(DATA_DIR, self.name))

    def download(self, filename, url):
        if os.path.exists(filename):
            print("File {} already exists, skipping".format(filename))
            return
        cmd = ['wget', '-nc', url, '-O', filename]
        Popen(cmd).wait()
        if filename.endswith('.tgz') or filename.endswith('.tar.gz'):
            os.system('ls *gz | xargs -n 1 tar xzvf')
        elif filename.endswith('.tar'):
            os.system('ls *.tar | xargs -n 1 tar xvf')
        elif filename.endswith('.zip'):
            os.system('unzip *.zip')


def mkdir(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print('Creating directory {}'.format(path))
        os.mkdir(path)


def listdir(path):
    filenames = os.listdir(os.path.expanduser(path))
    filenames = sorted(filenames)
    return [os.path.join(path, fn) for fn in filenames]


def main():
    print("Downloading {} dataset files...".format(DATASET_NAME))
    dl = DatasetDownloader('voc2007')
    dl.download('VOCtrainval_06-Nov-2007.tar', TRAIN_URL)
    print("TODO...")
    raise NotImplementedError


def save_image_dataset(examples):
    output_filename = '{}/{}.dataset'.format(DATA_DIR, DATASET_NAME)
    fp = open(output_filename, 'w')
    for line in examples:
        fp.write(json.dumps(line) + '\n')
    fp.close()
    print("Wrote {} items to {}".format(len(examples), output_filename))


if __name__ == '__main__':
    main()
