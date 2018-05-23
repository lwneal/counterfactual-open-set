import os
import time
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
from vector import clamp_to_unit_sphere
import imutil


def to_torch(z, requires_grad=False):
    return Variable(torch.FloatTensor(z), requires_grad=requires_grad).cuda()


def to_np(z):
    return z.data.cpu().numpy()


# Generates synthetic open set images using the algorithm in:
# https://arxiv.org/pdf/1707.07418.pdf
def generate(networks, dataloader, **options):
    """
    # TODO: Fix Dropout/BatchNormalization outside of training
    for net in networks:
        networks[net].eval()
    """
    result_dir = options['result_dir']
    gan_scale = options['cf_gan_scale']
    netG = networks['generator']
    netC = networks['classifier_k']
    netE = networks['encoder']

    # Ge et al: Selection of Generated Samples
    # A conditional GAN is used to generate images, and
    # incorrectly predicted samples are selected as candidates
    # Here, we don't use a conditional GAN, so instead of
    # incorrect prediction we take the predicted samples
    # with classification confidence less than a threshold
    classification_threshold = .5
    openset_images = []
    while len(openset_images) < 64:
        # Start with randomly-selected images from the dataloader
        start_images, _ = dataloader.get_batch()
        end_images, _ = dataloader.get_batch()

        # Interpolate between them in latent space
        z_0 = netE(start_images, gan_scale)
        z_1 = netE(end_images, gan_scale)
        theta = np.random.uniform(size=len(z_0))
        theta = Variable(torch.FloatTensor(theta)).cuda()
        theta = theta.unsqueeze(-1)
        z_interp = theta * z_0 + (1 - theta) * z_1

        images = netG(z_interp, gan_scale)
        preds = F.softmax(netC(images))
        confidence = preds.max(dim=1)[0]
        images = images.data.cpu().numpy()
        for idx, conf in enumerate(confidence.data.cpu().numpy()):
            if conf < classification_threshold:
                openset_images.append(images[idx])
    openset_images = openset_images[:64]

    images = np.array(openset_images).transpose((0,2,3,1))
    dummy_class = 0
    video_filename = make_video_filename(result_dir, dataloader, dummy_class, dummy_class, label_type='grid')

    # Save the images in npy/jpg format as input for the labeling system
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)
    imutil.show(images, display=False, filename=video_filename.replace('.mjpeg', '.jpg'))

    # Save the images in jpg format to display to the user
    name = 'counterfactual_{}.jpg'.format(int(time.time()))
    jpg_filename = os.path.join(result_dir, 'images', name)
    imutil.show(images, filename=jpg_filename)
    return images


# Trajectories are written to result_dir/trajectories/
def make_video_filename(result_dir, dataloader, start_class, target_class, label_type='active'):
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    start_class_name = dataloader.lab_conv.labels[start_class]
    target_class_name = dataloader.lab_conv.labels[target_class]
    video_filename = '{}-{}-{}-{}.mjpeg'.format(label_type, trajectory_id, start_class_name, target_class_name)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(result_dir, video_filename)
    path = os.path.join(result_dir, 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)
    return video_filename
