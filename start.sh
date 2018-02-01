#!/bin/bash

pip install -r requirements.txt

if [ ! -f /mnt/data/svhn.dataset ]; then
    python src/datasets/download_svhn.py
fi
#python src/datasets/download_mnist.py
#python src/datasets/download_celeba.py
#python src/datasets/download_oxford102.py

python src/train_gan.py 2>&1 | tee stdout.txt
