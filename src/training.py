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

    start_time = time.time()
    correct = 0
    total = 0

    log = TimeSeries()

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        gan_scale = 4
        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Hack: Just use a single class (back to dcgan)
        # Classify sampled images as fake
        noise = make_noise(gan_scale)
        fake_images = netG(noise, gan_scale)
        logits = netD(fake_images)[:,0]
        loss_fake = F.relu(logits).mean()
        log.collect('Discriminator Loss on Generated Examples', loss_fake)

        # Classify real examples as real
        logits = netD(images)[:,0]
        loss_real = F.relu(-logits).mean()
        log.collect('Discriminator Loss on Real Examples', loss_real)

        errD = (loss_real.mean() + loss_fake.mean()) * options['discriminator_weight']
        errD.backward()
        log.collect('Discriminator Loss Total', errD)

        optimizerD.step()
        ############################

        ############################
        # Generator Update
        ###########################
        netG.zero_grad()
        netE.zero_grad()

        # Minimize fakeness of sampled images
        # noise = make_noise(gan_scale)
        # fake_images = netG(noise, gan_scale)

        # Minimize fakeness of autoencoded images
        fake_images = netG(netE(images, gan_scale), gan_scale)

        # Hack: Just use a single class (back to dcgan)
        logits = netD(fake_images)[:,0]
        errG = F.relu(-logits).mean() * options['generator_weight']
        errG.backward()
        log.collect('Generator Loss', errG)

        ############################
        # Encoder Update
        ###########################
        # Minimize reconstruction loss (of samples)
        reconstructed = netG(netE(images, gan_scale), gan_scale)
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
        errC = F.relu(classifier_logits * -labels).mean()
        errC.backward()
        log.collect('Classifier Loss', errC)

        optimizerC.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        log.collect_prediction('Classifier Accuracy', netC(images), labels)
        log.collect_prediction('Discriminator Accuracy, Real Data', netD(images), labels)

        if i % 10 == 0:
            if i % 100 == 0:
                def image_filename(*args):
                    image_path = os.path.join(result_dir, 'images')
                    name = '_'.join(str(s) for s in args)
                    name += '_{}'.format(int(time.time() * 1000))
                    return os.path.join(image_path, name) + '.jpg'

                seed()
                fixed_noise = make_noise(gan_scale)
                seed(int(time.time()))

                demo_fakes = netG(fixed_noise, gan_scale)
                img = demo_fakes.data[:16]

                filename = image_filename('samples', 'scale', gan_scale)
                caption = "S scale={} epoch={} iter={}".format(gan_scale, epoch, i)
                imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)


                aac_before = images[:8]
                aac_after = netG(netE(aac_before, gan_scale), gan_scale)
                img = torch.cat((aac_before, aac_after))

                filename = image_filename('reconstruction', 'scale', gan_scale)
                caption = "R scale={} epoch={} iter={}".format(gan_scale, epoch, i)
                imutil.show(img, filename=filename, resize_to=(256,256), caption=caption)

            print(log)
    return True
