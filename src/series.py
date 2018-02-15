#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import time
import whattimeisit
from tqdm import tqdm

def resample(x, desired_length):
    width = len(x) // desired_length
    padding = (0, width - x.size % width)
    padded = np.pad(x, padding, mode='constant', constant_values=np.NaN)
    return np.nanmean(padded.reshape(-1, width), axis=1)


def sparkline(data, length=16):
    BARS = u'▁▂▃▅▆▇'
    if len(data) > length:
        truncate = (len(data) // length) * length
        samples = resample(data[:truncate], length)
    else:
        step = len(data) / length
        samples = [data[int(step * i / length)] for i in range(length)]
    incr = min(samples)
    width = (max(samples) - min(samples)) / (len(BARS) - 1)
    bins = [i*width+incr for i in range(len(BARS))]
    indexes = [i for n in samples
                       for i, thres in enumerate(bins)
                                  if thres <= n < thres+width]
    return ''.join(BARS[i] for i in indexes)


class TimeSeries:
    def __str__(self):
        return self.format_all()

    def __init__(self, title=None, epoch_length=None):
        self.series = {}
        self.predictions = {}
        self.start_time = time.time()
        self.last_printed_at = time.time()
        self.title = title
        self.epoch_length = epoch_length

    def collect(self, name, value):
        if not self.series:
            self.start_time = time.time()
        if name not in self.series:
            self.series[name] = []
        self.series[name].append(convert_to_scalar(value))

    def collect_prediction(self, name, logits, ground_truth):
        if name not in self.predictions:
            self.predictions[name] = {'correct': 0, 'total': 0}
        _, pred_idx = logits.max(1)
        _, label_idx = ground_truth.max(1)
        correct = convert_to_scalar(sum(pred_idx == label_idx))
        self.predictions[name]['correct'] += correct
        self.predictions[name]['total'] += len(ground_truth)

    def print_every(self, n_sec=4):
        if time.time() - self.last_printed_at > n_sec:
            print(self.format_all())
            self.last_printed_at = time.time()

    def format_all(self):
        lines = ['']
        if self.title:
            lines.append(self.title)
        duration = time.time() - self.start_time
        lines.append("Collected {:.3f} sec ending {}".format(
            duration, whattimeisit()))
        maxlen = max(len(c) for c in self.series.values())
        if self.epoch_length:
            lines.append(tqdm.format_meter(maxlen, self.epoch_length, duration, ncols=80))
        else:
            lines.append("Collected {:8d} points ({:.2f}/sec)".format(maxlen, maxlen / duration))
        lines.append("{:>32}{:>14}{:>16}".format('Name', 'Avg.', 'Last 10'))
        for name in sorted(self.series):
            values = np.array(self.series[name])
            name = shorten(name)
            lines.append("{:>32}:      {:8.4f}      {:8.4f} {}".format(
                name, values.mean(), values[-10:].mean(), sparkline(values)))
        if self.predictions:
            lines.append('Predictions:')
        for name, pred in self.predictions.items():
            acc = 100 * pred['correct'] / pred['total']
            name = shorten(name)
            lines.append('{:>32}:\t{:.2f}% ({}/{})'.format(
                name, acc, pred['correct'], pred['total']))
        lines.append('\n')
        text = '\n'.join(lines)
        # Cache the most recent printed text to a file
        open('.last_summary.log', 'w').write(text)
        return text

    def write_to_file(self):
        filename = 'timeseries.{}.npy'.format(int(time.time()))
        ts = np.array(self.series[0])
        ts.save(filename)
        sys.stderr.write('Wrote array shape {} to file {}\n'.format(ts.shape, filename))


# We assume x is a scalar.
# If x is not a scalar, that is a problem that we will fix right now.
def convert_to_scalar(x):
    if type(x).__name__ == 'FloatTensor':
        x = x.cpu()[0]
    elif type(x).__name__ == 'Variable':
        x = x.data.cpu()[0]
    try:
        return float(x)
    except:
        pass
    return 0

def shorten(words, maxlen=30):
    if len(words) > 27:
        words = words[:20] + '...' + words[-9:]
    return words
