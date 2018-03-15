import sys
import numpy as np

import plotting  # init matplotlib
import matplotlib.pyplot as plt
from imutil import show

output_filename = sys.argv[1]

dataset = 'cifar'
#dataset = 'svhn'

curves = [
    ('--', 'Softmax Thresholding', 'roc_{}_baseline.npy'.format(dataset)),
    ('-.', 'OpenMax', 'roc_{}_openmax.npy'.format(dataset)),
    (':', 'G-OpenMax', 'roc_{}_gopenmax.npy'.format(dataset)),
    ('-', 'Ours',    'roc_{}_ours.npy'.format(dataset)),
]
names = list(zip(*curves))[1]

for dots, name, filename in curves:
    x, y = np.load(filename)
    plt.plot(x, y, linestyle=dots)
plt.title('Open Set Detection: CIFAR', fontsize=18)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)

# Zoomed-in version
#plt.xscale('log')
#plt.xlim(.01)
#plt.yscale('log')

plt.legend(names)
show(plt.gca())
plt.savefig(output_filename)
