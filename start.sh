#!/bin/bash
set -e

pip install -r requirements.txt

# TODO: do this in python, based on --dataset
if [ ! -f /mnt/data/svhn-04.dataset ]; then
    python src/datasets/download_svhn.py
fi
if [ ! -f /mnt/data/cifar10.dataset ]; then
    python src/datasets/download_cifar10.py
fi
if [ ! -f /mnt/data/mnist.dataset ]; then
    python src/datasets/download_mnist.py
fi

#python src/datasets/download_svhn.py
#python src/datasets/download_mnist.py
#python src/datasets/download_celeba.py
#python src/datasets/download_oxford102.py

for i in `seq 100`; do
    python src/train_gan.py --epochs 1 2>&1 | tee -a stdout.txt
    python src/evaluate_classifier.py --result_dir . --comparison_dataset /mnt/data/svhn-59.dataset 2>&1 | tee -a stdout.txt
done
