import numpy as np

import plotting  # init matplotlib
import matplotlib.pyplot as plt
from imutil import show

curves = [
    ('Softmax Thresholding', 'roc_cifar_baseline.npy'),
    ('OpenMax', 'roc_cifar_openmax.npy'),
    ('G-OpenMax', 'roc_cifar_gopenmax.npy'),
    ('Ours',    'roc_cifar_ours.npy'),
]

for name, filename in curves:
    x, y = np.load(filename)
    plt.plot(x, y)
plt.legend(list(zip(*curves))[0])

show(plt.gca())
