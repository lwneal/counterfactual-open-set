import time
import os
import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import models
from torch.autograd import Variable
from torch.nn.functional import nll_loss, binary_cross_entropy
import torch.nn.functional as F
from torch.nn.functional import softmax, log_softmax, relu
import imutil
from vector import gen_noise, clamp_to_unit_sphere
from dataloader import FlexibleCustomDataloader
from series import TimeSeries


def seed(val=42):
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)


def log_sum_exp(inputs, dim=None, keepdim=False):
    return inputs - log_softmax(inputs, dim=1)


def train_gan(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
    netE = networks['encoder']
    netD = networks['discriminator']
    netG = networks['generator']
    netC = networks['classifier']
    optimizerE = optimizers['encoder']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerC = optimizers['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    def make_noise(scale):
        noise_t = torch.FloatTensor(batch_size, latent_size * scale * scale)
        noise_t.normal_(0, 1)
        noise = Variable(noise_t).cuda()
        return clamp_to_unit_sphere(noise, scale**2)

    aux_dataloader = None
    dataset_filename = options.get('aux_dataset')
    if dataset_filename and os.path.exists(dataset_filename):
        print("Using aux_dataset {}".format(dataset_filename))
        aux_dataloader = FlexibleCustomDataloader(dataset_filename, batch_size=batch_size, image_size=image_size)

    log = TimeSeries()

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        ac_scale = random.choice([1, 2, 4, 8])
        sample_scale = 1
        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify sampled images as fake
        noise = make_noise(sample_scale)
        fake_images = netG(noise, sample_scale)
        logits = netD(fake_images)[:,0]
        loss_fake_sampled = F.relu(logits).mean()
        log.collect('Discriminator Sampled', loss_fake_sampled)
        loss_fake_sampled.backward()

        # Classify autoencoded images as fake
        more_images, more_labels = dataloader.get_batch()
        more_images = Variable(more_images)
        fake_images = netG(netE(more_images, ac_scale), ac_scale)
        logits_fake = netD(fake_images)[:,0]
        loss_fake = F.relu(logits_fake)
        log.collect('Discriminator Autoencoded', loss_fake)

        # Classify real examples as real
        logits = netD(images)[:,0]
        loss_real = F.relu(-logits).mean()
        log.collect('Discriminator Real', loss_real)

        errD = (loss_real.mean() + loss_fake.mean()) * options['discriminator_weight']
        errD.backward()
        log.collect('Discriminator Total', errD)

        optimizerD.step()
        ############################

        ############################
        # Generator Update
        ###########################
        netG.zero_grad()
        netE.zero_grad()

        # Minimize fakeness of sampled images
        noise = make_noise(sample_scale)
        fake_images_sampled = netG(noise, sample_scale)
        logits = netD(fake_images_sampled)[:,0]
        errSampled = F.softplus(-logits).mean() * options['generator_weight']
        errSampled.backward()
        log.collect('Generator Sampled', errSampled)

        # Minimize fakeness of autoencoded images
        fake_images = netG(netE(images, sample_scale), sample_scale)
        logits = netD(fake_images)[:,0]
        errG = F.softplus(-logits).mean() * options['generator_weight']
        errG.backward()
        log.collect('Generator Autoencoded', errG)

        ############################
        # Encoder Update
        ###########################
        # Minimize reconstruction loss
        reconstructed = netG(netE(images, ac_scale), ac_scale)
        err_reconstruction = torch.mean(torch.abs(images - reconstructed)) * options['reconstruction_weight']
        err_reconstruction.backward()
        log.collect('Pixel Reconstruction Loss', err_reconstruction)

        optimizerG.step()
        optimizerE.step()
        ###########################

        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes with hinge loss
        classifier_logits = netC(images) 
        errC = F.softplus(classifier_logits * -labels).mean()
        errC.backward()
        log.collect('Classifier Loss', errC)

        optimizerC.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        log.collect_prediction('Classifier Accuracy', netC(images), labels)
        #log.collect_prediction('Discriminator Accuracy, Real Data', netD(images), labels)

        log.print_every()

        if i % 100 == 0:
            def image_filename(*args):
                image_path = os.path.join(result_dir, 'images')
                name = '_'.join(str(s) for s in args)
                name += '_{}'.format(int(time.time() * 1000))
                return os.path.join(image_path, name) + '.jpg'

            seed()
            fixed_noise = make_noise(sample_scale)
            seed(int(time.time()))

            demo_fakes = netG(fixed_noise, sample_scale)
            img = demo_fakes.data[:16]

            filename = image_filename('samples', 'scale', sample_scale)
            caption = "S scale={} epoch={} iter={}".format(sample_scale, epoch, i)
            imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)


            aac_before = images[:8]
            aac_after = netG(netE(aac_before, ac_scale), ac_scale)
            img = torch.cat((aac_before, aac_after))

            filename = image_filename('reconstruction', 'scale', ac_scale)
            caption = "R scale={} epoch={} iter={}".format(ac_scale, epoch, i)
            imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)
    return True


def train_classifier(networks, optimizers, dataloader, epoch=None, **options):
    for net in networks.values():
        net.train()
    netC = networks['classifier']
    optimizerC = optimizers['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    aux_dataloader = None
    dataset_filename = options.get('aux_dataset')
    if dataset_filename and os.path.exists(dataset_filename):
        print("Using aux_dataset {}".format(dataset_filename))
        aux_dataloader = FlexibleCustomDataloader(dataset_filename, batch_size=batch_size, image_size=image_size)

    log = TimeSeries()

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes
        classifier_logits = netC(images)
        augmented_logits = F.pad(classifier_logits, (0,1))
        _, labels_idx = labels.max(dim=1)
        errC = nll_loss(F.log_softmax(augmented_logits, dim=1), labels_idx)
        errC.backward()
        log.collect('Classifier Loss', errC)

        # Classify RED labeled aux_dataset as open set
        aux_images, aux_labels = aux_dataloader.get_batch()
        classifier_logits = netC(Variable(aux_images))
        augmented_logits = F.pad(classifier_logits, (0,1))
        is_positive = Variable(aux_labels.max(dim=1)[0])
        is_openset = 1 - is_positive
        log_soft_open = F.log_softmax(augmented_logits)[:, -1] * is_openset
        errOpenSet = -log_soft_open.sum() / is_openset.sum()
        errOpenSet.backward()
        log.collect('Open Set Loss', errOpenSet)

        optimizerC.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        log.collect_prediction('Classifier Accuracy', netC(images), labels)

        log.print_every()

    return True
