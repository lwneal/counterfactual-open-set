import json
import os

import libmr
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

from plotting import plot_xy

WEIBULL_TAIL_SIZE = 20


def evaluate_classifier(networks, dataloader, open_set_dataloader=None, **options):
    for net in networks.values():
        net.eval()
    if options.get('mode') == 'baseline':
        print("Using the K-class classifier")
        netC = networks['classifier_k']
    elif options.get('mode') == 'weibull':
        print("Weibull mode: Using the K-class classifier")
        netC = networks['classifier_k']
    else:
        print("Using the K+1 open set classifier")
        netC = networks['classifier_kplusone']
    fold = options.get('fold', 'evaluation')

    classification_closed_correct = 0
    classification_total = 0
    for images, labels in dataloader:
        images = Variable(images, volatile=True)
        # Predict a classification among known classes
        net_y = netC(images)
        class_predictions = F.softmax(net_y, dim=1)

        _, predicted = class_predictions.max(1)
        classification_closed_correct += sum(predicted.data == labels)
        classification_total += len(labels)

    stats = {
        fold: {
            'closed_set_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }
    return stats


def pca(vectors):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vectors)
    return pca.transform(vectors)


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

    d_scores_on = get_openset_scores(dataloader_on, networks, **options)
    d_scores_off = get_openset_scores(dataloader_off, networks, **options)

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
    current_epoch = options.get('epoch', 0)
    comparison_name = options['comparison_dataset'].split('/')[-1].replace('.dataset', '')
    filename = 'plot_{}_vs_{}_epoch_{:04d}.png'.format(title, comparison_name, current_epoch)
    filename = os.path.join(options['result_dir'], filename)
    plot.figure.savefig(filename)
    

def get_openset_scores(dataloader, networks, dataloader_train=None, **options):
    if options.get('mode') == 'weibull':
        openset_scores = openset_weibull(dataloader, dataloader_train, networks['classifier_k'])
    elif options.get('mode') == 'weibull-kplusone':
        openset_scores = openset_weibull(dataloader, dataloader_train, networks['classifier_kplusone'])
    elif options.get('mode') == 'baseline':
        openset_scores = openset_kplusone(dataloader, networks['classifier_k'])
    elif options.get('mode') == 'autoencoder':
        openset_scores = openset_autoencoder(dataloader, networks)
    else:
        openset_scores = openset_kplusone(dataloader, networks['classifier_kplusone'])
    return openset_scores


def openset_autoencoder(dataloader, networks, scale=4):
    netE = networks['encoder']
    netG = networks['generator']
    netE.train()
    netG.train()

    openset_scores = []
    for images, labels in dataloader:
        images = Variable(images)
        reconstructions = netG(netE(images, 4), 4)
        mse = ((reconstructions - images) ** 2).sum(dim=-1).sum(dim=-1).sum(dim=-1)
        openset_scores.extend([v for v in mse.data.cpu().numpy()])
    return openset_scores


def openset_weibull(dataloader_test, dataloader_train, netC):
    # First generate pre-softmax 'activation vectors' for all training examples
    print("Weibull: computing features for all correctly-classified training data")
    activation_vectors = {}
    for images, labels in dataloader_train:
        logits = netC(images)
        correctly_labeled = (logits.data.max(1)[1] == labels)
        labels_np = labels.cpu().numpy()
        logits_np = logits.data.cpu().numpy()
        for i, label in enumerate(labels_np):
            if not correctly_labeled[i]:
                continue
            # If correctly labeled, add this to the list of activation_vectors for this class
            if label not in activation_vectors:
                activation_vectors[label] = []
            activation_vectors[label].append(logits_np[i])
    print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))
    for class_idx in activation_vectors:
        print("Class {}: {} images".format(class_idx, len(activation_vectors[class_idx])))

    # Compute a mean activation vector for each class
    print("Weibull computing mean activation vectors...")
    mean_activation_vectors = {}
    for class_idx in activation_vectors:
        mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)

    # Initialize one libMR Wiebull object for each class
    print("Fitting Weibull to distance distribution of each class")
    weibulls = {}
    for class_idx in activation_vectors:
        distances = []
        mav = mean_activation_vectors[class_idx]
        for v in activation_vectors[class_idx]:
            distances.append(np.linalg.norm(v - mav))
        mr = libmr.MR()
        tail_size = min(len(distances), WEIBULL_TAIL_SIZE)
        mr.fit_high(distances, tail_size)
        weibulls[class_idx] = mr
        print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))

    # Apply Weibull score to every logit
    weibull_scores = []
    logits = []
    classes = activation_vectors.keys()
    for images, labels in dataloader_test:
        batch_logits = netC(images).data.cpu().numpy()
        batch_weibull = np.zeros(shape=batch_logits.shape)
        for activation_vector in batch_logits:
            weibull_row = np.ones(len(classes))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores.append(weibull_row)
            logits.append(activation_vector)
    weibull_scores = np.array(weibull_scores)
    logits = np.array(logits)

    # TODO: Try using the alpha array hack from https://arxiv.org/pdf/1511.06233.pdf

    modified_logits = (logits - logits.min()) * weibull_scores
    softmax_scores = -np.log(np.sum(np.exp(logits), axis=1))
    openmax_scores = -np.log(np.sum(np.exp(modified_logits), axis=1))
    return np.array(openmax_scores)


def openset_kplusone(dataloader, netC):
    openset_scores = []
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        preds = netC(images)
        # The implicit K+1th class (the open set class) is computed
        #  by assuming an extra linear output with constant value 0
        z = torch.exp(preds).sum(dim=1)
        prob_known = z / (z + 1)
        prob_unknown = 1 - prob_known
        openset_scores.extend(prob_unknown.data.cpu().numpy())
    return np.array(openset_scores)


def plot_roc(y_true, y_score, title="Receiver Operating Characteristic"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    return auc_score, plot


def save_evaluation(new_results, result_dir, epoch):
    if not os.path.exists('evaluations'):
        os.mkdir('evaluations')
    filename = 'evaluations/eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(result_dir, filename)
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        old_results = json.load(open(filename))
    else:
        old_results = {}
    old_results.update(new_results)
    with open(filename, 'w') as fp:
        json.dump(old_results, fp, indent=2, sort_keys=True)
