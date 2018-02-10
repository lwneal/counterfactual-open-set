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
if [ ! -f /mnt/data/oxford102.dataset ]; then
    python src/datasets/download_oxford102.py
fi
if [ ! -f /mnt/data/celeba.dataset ]; then
    python src/datasets/download_celeba.py
fi

GAN_EPOCHS=30
CLASSIFIER_EPOCHS=30
CF_COUNT=200

export IMUTIL_SHOW=""

# Train the intial generative model (E+G+D+C)
python src/train_gan.py --epochs $GAN_EPOCHS 2>&1 | tee -a stdout.txt

# Generate a number of counterfactuals, in K+2 by K+2 square grids
python src/generate_counterfactual.py --result_dir . --count $CF_COUNT 2>&1 | tee -a stdout.txt


# Automatically label the rightmost column in each grid (ignore the others)
# TODO: Something more elegant than this?
python src/auto_label.py --output_filename generated_images.dataset 2>&1 | tee -a stdout.txt


# Train a new classifier, now using the aux_dataset containing the counterfactuals
python src/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images.dataset --comparison_dataset /mnt/data/svhn-59.dataset 2>&1 | tee -a stdout.txt


# Evaluate it one more time just for good measure
python src/evaluate_classifier.py --result_dir . --comparison_dataset /mnt/data/svhn-59.dataset 2>&1 | tee -a stdout.txt

