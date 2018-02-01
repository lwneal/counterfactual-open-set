import json
import numpy as np
import pandas as pd
from imutil import show

# Hack to keep matplotlib from crashing when run without X
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Apply sane defaults to matplotlib
import seaborn as sns
sns.set_style('darkgrid')


class NoDataAvailable(Exception):
    pass


def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    df = pd.DataFrame({'x': x, 'y': y})
    plot = df.plot(x='x', y='y')

    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')
    
    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot


def parse_active_learning_series(eval_filename, prefix='active_trajectories', statistic='accuracy'):
    try:
        evaluations = json.load(open(eval_filename))
    except:
        print("Error: could not load JSON from file {}".format(eval_filename))
        raise NoDataAvailable
    keys = sorted([k for k in evaluations if k.startswith(prefix)])
    if len(keys) == 0:
        raise NoDataAvailable
    x = [int(k.split('_')[-1]) for k in keys]
    y = [evaluations[k][statistic] for k in keys]
    return x, y


def plot_active_learning(eval_filename="results_epoch_0025.json"):
    try:
        x, y = parse_active_learning_series(eval_filename)
    except NoDataAvailable:
        return None
    plot = plot_xy(x, y, x_axis="Number of Examples", y_axis="Accuracy", title=eval_filename)
    plot_filename = "{}.jpg".format(eval_filename.replace('.json', ''))
    show(plot, filename=plot_filename)
    return plot


def compare_active_learning(eval_filename, baseline_eval_filename, title=None, this_name='This Method', baseline_name='Baseline', prefix='grid', statistic='accuracy'):
    try:
        x, y = parse_active_learning_series(eval_filename, prefix=prefix, statistic=statistic)
        x2, y2 = parse_active_learning_series(baseline_eval_filename, prefix=prefix, statistic=statistic)
    except NoDataAvailable:
        return None
    print("Comparing {} with baseline {}".format(eval_filename, baseline_eval_filename))

    plt.plot(x, y, "g") # this method
    plt.plot(x2, y2, "b") # baseline
    this_approach_name = eval_filename.split('/')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Queries')
    plt.legend([this_name, baseline_name])
    if title:
        plt.suptitle(title)

    fig_filename = eval_filename.replace('.json', '-vs-baseline.png')
    plt.savefig(fig_filename)
    show(fig_filename)
    return fig_filename


def compare_multiple(list_of_filenames, list_of_names, output_filename, title=None):
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    assert len(list_of_filenames) <= len(styles)
    plt.figure(figsize=(9.5,6))
    for filename, style in zip(list_of_filenames, styles):
        x, y = parse_active_learning_series(filename)
        plt.plot(x, y, style)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Queries')
    plt.legend(list_of_names)
    if title:
        plt.suptitle(title)

    plt.savefig(output_filename)
    show(output_filename)
    return output_filename

