#!/bin/bash
# Stop script if any command returns an error code
set -e

echo "Running job on `hostname` ip `hostname -I`"

# Output stdout with no buffering, so it can be piped
export PYTHONUNBUFFERED=1

# Do not show images in-terminal, even if imgcat is installed
export IMUTIL_SHOW=""

# Install requirements in order, including dependencies
while read p; do
    pip install $p
done < requirements.txt

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

GAN_EPOCHS=30
CLASSIFIER_EPOCHS=3
CF_COUNT=30
GENERATOR_MODE=counterfactual
#GENERATOR_MODE=ge_et_al


# Train the intial generative model (E+G+D+C_k)
python src/train_gan.py --epochs $GAN_EPOCHS

# Baseline: Evaluate the regular classifier (C_k+1)
python src/evaluate_classifier.py --result_dir . --mode baseline
python src/evaluate_classifier.py --result_dir . --mode weibull

# For 100, 200, ... generated examples:
for i in `seq 10`; do
    # Generate a number of counterfactuals, in K+2 by K+2 square grids
    python src/generate_${GENERATOR_MODE}.py --result_dir . --count $CF_COUNT

    # Automatically label the rightmost column in each grid (ignore the others)
    python src/auto_label_${GENERATOR_MODE}.py --output_filename generated_images_${GENERATOR_MODE}.dataset

    # Train a new classifier, now using the aux_dataset containing the counterfactuals
    python src/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images_${GENERATOR_MODE}.dataset

    # Evaluate it one more time just for good measure
    python src/evaluate_classifier.py --result_dir . --aux_dataset generated_images_${GENERATOR_MODE}.dataset
done
python src/evaluate_classifier.py --result_dir . --mode weibull-kplusone
