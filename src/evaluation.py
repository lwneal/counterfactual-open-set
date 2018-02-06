import json
import sys
import os
import numpy as np
import time
import torch
from pprint import pprint
from torch.autograd import Variable
from torch.nn.functional import nll_loss
from torch.nn.functional import softmax, sigmoid
import imutil
from sklearn.metrics import roc_curve, auc


# Returns 1 for items that are known, 0 for unknown
def predict_openset(networks, images, threshold=0.):
    netC = networks['classifier']
    preds = netC(images)
    maxval, _ = preds.max(dim=1) 
    return maxval > threshold


def evaluate_classifier(networks, dataloader, open_set_dataloader=None, **options):
    for net in networks.values():
        net.eval()
    netG = networks['generator']
    netC = networks['classifier']
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']

    classification_closed_correct = 0
    classification_correct = 0
    classification_total = 0
    for images, labels in dataloader:
        images = Variable(images, volatile=True)
        # Predict a classification among known classes
        net_y = netC(images)
        class_predictions = softmax(net_y, dim=1)
        
        # Also predict whether each example belongs to any class at all
        is_known = predict_openset(networks, images)

        _, predicted = class_predictions.max(1)
        classification_closed_correct += sum(predicted.data == labels)
        classification_correct += sum((predicted.data == labels) * is_known.data)
        classification_total += len(labels)

    openset_correct = 0
    openset_total = 1
    if open_set_dataloader is not None:
        for images, labels in open_set_dataloader:
            images = Variable(images, volatile=True)
            # Predict whether each example is known/unknown
            is_known = predict_openset(networks, images)
            openset_correct += sum(is_known.data == 0)
            openset_total += len(labels)

    stats = {
        options['fold']: {
            'accuracy': float(classification_correct + openset_correct) / (classification_total + openset_total),
            'classification_accuracy': float(classification_correct) / (classification_total),
            'closed_set_accuracy': float(classification_closed_correct) / (classification_total),
            'openset_recall': float(openset_correct) / (openset_total),
        }
    }
    return stats


def pca(vectors):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vectors)
    return pca.transform(vectors)


# Plots a list of 2d points
def plot(dots, output_filename, title=None, labels=None):
    # Incantation to enable headless mode
    import matplotlib
    matplotlib.use('Agg')
    # Apply sane Seaborn defaults to Matplotlib
    import seaborn as sns
    sns.set_style('darkgrid')
    import matplotlib.pyplot as plt

    import pandas as pd
    df = pd.DataFrame(dots)
    df.columns = ['Z_1', 'Z_2']
    df['label'] = labels
    plot = sns.pairplot(df, size=4, hue='label', plot_kws={'s':12})
    if title:
        plot.fig.suptitle(title)
    plot.savefig(output_filename)


# Open Set Classification
# Given two datasets, one on-manifold and another off-manifold, predict
# whether each item is on-manifold or off-manifold using the discriminator
# or the autoencoder loss.
# Plot an ROC curve for each and report AUC
# dataloader_on: Test set of the same items the network was trained on
# dataloader_off: Separate dataset from a different distribution
def evaluate_openset(networks, dataloader_on, dataloader_off, **options):
    for net in networks.values():
        net.eval()
    result_dir = options['result_dir']
    image_size = options['image_size']
    latent_size = options['latent_size']

    d_scores_on = get_openset_scores(dataloader_on, networks)
    d_scores_off = get_openset_scores(dataloader_off, networks)

    y_true = np.array([0] * len(d_scores_on) + [1] * len(d_scores_off))
    y_discriminator = np.concatenate([d_scores_on, d_scores_off])

    auc_d, plot_d = plot_roc(y_true, y_discriminator, 'Discriminator ROC vs {}'.format(dataloader_off.dsf.name))

    save_plot(plot_d, 'roc_discriminator', **options)

    return {
        'auc_discriminator': auc_d,
    }


def combine_scores(score_list):
    example_count = len(score_list[0])
    assert all(len(x) == example_count for x in score_list)

    normalized_scores = np.ones(example_count)
    for score in score_list:
        score -= score.min()
        score /= score.max()
        normalized_scores *= score
        normalized_scores /= normalized_scores.max()
    return normalized_scores


def save_plot(plot, title, **options):
    comparison_name = options['comparison_dataset'].split('/')[-1].replace('.dataset', '')
    filename = 'plot_{}_vs_{}_epoch_{:04d}.png'.format(title, comparison_name, options['epoch'])
    filename = os.path.join(options['result_dir'], filename)
    plot.figure.savefig(filename)
    

def get_openset_scores(dataloader, networks):
    netC = networks['classifier']

    # The implicit K+1th class (the open set class) is computed
    #  by assuming an extra linear output with constant value 0
    discriminator_scores = []
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        preds = netC(images)
        z = torch.exp(preds).sum(dim=1)
        prob_known = z / (z + 1)
        prob_unknown = 1 - prob_known
        discriminator_scores.extend(prob_unknown.data.cpu().numpy())

    discriminator_scores = np.array(discriminator_scores)
    return discriminator_scores


def plot_roc(y_true, y_score, title="Receiver Operating Characteristic"):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    from plotting import plot_xy
    plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    return auc_score, plot


def save_evaluation(new_results, result_dir, epoch):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(result_dir, filename)
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        old_results = json.load(open(filename))
    else:
        old_results = {}
    old_results.update(new_results)
    with open(filename, 'w') as fp:
        json.dump(old_results, fp, indent=2, sort_keys=True)
