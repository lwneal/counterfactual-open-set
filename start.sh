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

GAN_EPOCHS=1
CLASSIFIER_EPOCHS=10

# Train the intial generative model (E+G+D+C)
python src/train_gan.py --epochs $GAN_EPOCHS 2>&1 | tee -a stdout.txt

# Generate a number of counterfactuals
python src/generate_counterfactual.py --result_dir .

# TODO: Automatically label those counterfactuals
# for now just copy this pre-generated set of 950 cf's
cp /mnt/nfs/experiments/svhn-04_aux_example.dataset .

# Then train a new classifier, using the aux_dataset
python src/train_classifier.py --epochs $CLASSIFIER_EPOCHS

# Then evaluate the classifier
python src/evaluate_classifier.py --result_dir . --comparison_dataset /mnt/data/svhn-59.dataset
