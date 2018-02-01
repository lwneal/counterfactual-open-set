#!/bin/bash

pip install -r requirements.txt

# TODO: Download just the correct dataset for this experiment
python src/datasets/download_svhn.py
python src/datasets/download_mnist.py

python src/train_gan.py --result_dir . --epochs 5 2>&1 | tee stdout.txt
