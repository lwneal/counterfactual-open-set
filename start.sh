#!/bin/bash

pip install -r requirements.txt

if [ ! -f /mnt/data/svhn-04.dataset ]; then
    python src/datasets/download_svhn.py
fi

#python src/datasets/download_svhn.py
#python src/datasets/download_mnist.py
#python src/datasets/download_celeba.py
#python src/datasets/download_oxford102.py

python src/train_gan.py --epochs 30 2>&1 | tee stdout.txt
for i in `seq 30`; do
    python src/evaluate_classifier.py --epoch $i --result_dir . --comparison_dataset /mnt/data/svhn-59.dataset 2>&1 | tee stdout.txt
done
