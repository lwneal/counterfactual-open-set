#!/bin/bash
# Stop script if any command returns an error code
set -e

# Output stdout with no buffering, so it can be piped
export PYTHONUNBUFFERED=1

# Do not show images in-terminal, even if imgcat is installed
export IMUTIL_SHOW=""

pip install -r requirements.txt

# TODO: do this in python, based on --dataset
if [ ! -f /mnt/data/svhn-04.dataset ]; then
    python src/datasets/download_svhn.py
fi
if [ ! -f /mnt/data/cifar10-animals.dataset ]; then
    python src/datasets/download_cifar10.py
fi
if [ ! -f /mnt/data/mnist-not5.dataset ]; then
    python src/datasets/download_mnist.py
fi
if [ ! -f /mnt/data/oxford102.dataset ]; then
    python src/datasets/download_oxford102.py
fi
if [ ! -f /mnt/data/celeba.dataset ]; then
    python src/datasets/download_celeba.py
fi

GAN_EPOCHS=100
CLASSIFIER_EPOCHS=10
CF_COUNT=10


# Train the intial generative model (E+G+D+C_k)
python src/train_gan.py --epochs $GAN_EPOCHS

# Baseline: Evaluate the regular classifier (C_k+1)
python src/evaluate_classifier.py --result_dir . --mode baseline

for i in `seq 10`; do
    # Generate a number of counterfactuals, in K+2 by K+2 square grids
    python src/generate_counterfactual.py --result_dir . --count $CF_COUNT

    # Automatically label the rightmost column in each grid (ignore the others)
    python src/auto_label.py --output_filename generated_images.dataset

    # Train a new classifier, now using the aux_dataset containing the counterfactuals
    python src/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images.dataset

    # Evaluate it one more time just for good measure
    python src/evaluate_classifier.py --result_dir . --comparison_dataset /mnt/data/svhn-59.dataset --aux_dataset generated_images.dataset
done
