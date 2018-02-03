import numpy as np
import time


class TimeSeries:
    def __str__(self):
        lines = []
        duration = time.time() - self.start_time
        lines.append("Logged for {:.3f} sec".format(duration))
        for name in sorted(self.series):
            values = np.array(self.series[name])
            lines.append("{:<24}:\t{:>8d} points, avg value {:.4f}".format(
                name, len(values), values.mean()))
        return '\n'.join(lines)

    def __init__(self):
        self.series = {}
        self.start_time = time.time()

    def collect(self, name, value):
        if not self.series:
            self.start_time = time.time()
        if name not in self.series:
            self.series[name] = []
        self.series[name].append(convert_to_scalar(value))


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
