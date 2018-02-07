import numpy as np
import time
import whattimeisit


class TimeSeries:
    def __str__(self):
        return self.format_all()

    def __init__(self):
        self.series = {}
        self.predictions = {}
        self.start_time = time.time()
        self.last_printed_at = time.time()

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
        lines = []
        duration = time.time() - self.start_time
        lines.append("Statistics for {:.3f} sec ending {}".format(
            duration, whattimeisit()))
        for name, values in self.series.items():
            values = np.array(values)
            name = shorten(name)
            lines.append("{:>32}:\t{:.4f} {:>8d} points, {:.2f} samples/sec".format(
                name, values.mean(), len(values), len(values) / duration))
        lines.append('Predictions:')
        for name, pred in self.predictions.items():
            acc = 100 * pred['correct'] / pred['total']
            name = shorten(name)
            lines.append('{:>32}:\t{:.2f}% ({}/{})'.format(
                name, acc, pred['correct'], pred['total']))
        lines.append('\n')
        return '\n'.join(lines)


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
