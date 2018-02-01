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
        return clamp_to_unit_sphere(noise, scale)


    start_time = time.time()
    correct = 0
    total = 0

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        gan_scale = random.choice([1, 2, 4, 8])
        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify sampled images as fake
        noise = make_noise(gan_scale)
        fake_images = netG(noise, gan_scale)

        fake_logits = netD(fake_images)
        augmented_logits = F.pad(fake_logits, pad=(0,1))
        log_prob_fake = F.log_softmax(augmented_logits, dim=1)[:, -1]
        errD = -log_prob_fake.mean()
        errD.backward()

        # Classify real examples into the correct K classes
        real_logits = netD(images)
        positive_labels = (labels == 1).type(torch.cuda.FloatTensor)
        augmented_logits = F.pad(real_logits, pad=(0,1))
        augmented_labels = F.pad(positive_labels, pad=(0,1))
        log_prob_real = F.log_softmax(augmented_logits, dim=1) * augmented_labels
        errC = -log_prob_real.mean()
        errC.backward()

        optimizerD.step()
        ############################

        ############################
        # Autoencoder Update
        ###########################
        netE.zero_grad()
        netG.zero_grad()

        # Minimize reconstruction loss
        reconstructed = netG(netE(images, gan_scale), gan_scale)
        errE = torch.mean(torch.abs(images - reconstructed))
        errE.backward()

        # Minimize fakeness of sampled images
        noise = make_noise(gan_scale)
        fake_images = netG(noise, gan_scale)

        fake_logits = netD(fake_images)
        augmented_logits = F.pad(fake_logits, pad=(0,1))
        log_prob_not_fake = F.log_softmax(-augmented_logits, dim=1)[:, -1]
        errG = -log_prob_not_fake.mean() * options['generator_weight']
        errG.backward()

        optimizerE.step()
        optimizerG.step()
        ###########################

        ############################
        # Classifier Update
        ############################
        netC.zero_grad()

        # Classify real examples into the correct K classes
        real_logits = netC(images)
        positive_labels = (labels == 1).type(torch.cuda.FloatTensor)
        augmented_logits = F.pad(real_logits, pad=(0,1))
        augmented_labels = F.pad(positive_labels, pad=(0,1))
        log_prob_real = F.log_softmax(augmented_logits, dim=1) * augmented_labels
        errC_real = -log_prob_real.mean()
        errC_real.backward()

        # Fake examples as open set
        noise = make_noise(gan_scale)
        fake_images = netG(noise, gan_scale)
        fake_logits = netC(fake_images)
        augmented_logits = F.pad(fake_logits, pad=(0,1))
        log_prob_fake = F.log_softmax(augmented_logits, dim=1)[:, -1]
        errC_fake = -log_prob_fake.mean()
        errC_fake.backward()

        optimizerC.step()
        ############################

        # Keep track of accuracy on positive-labeled examples for monitoring
        logits = netC(images)
        _, pred_idx = logits.max(1)
        _, label_idx = labels.max(1)
        correct += sum(pred_idx == label_idx).data.cpu().numpy()[0]
        total += len(labels)

        if i % 100 == 0:
            for gan_scale in (8, 4, 2, 1):
                seed()
                fixed_noise = make_noise(gan_scale)
                seed(int(time.time()))
                print("Generator Samples scale {}:".format(gan_scale))
                demo_fakes = netG(fixed_noise, gan_scale)
                img = demo_fakes.data[:16]
                filename = "{}/images/samples_{}_{}.jpg".format(result_dir, gan_scale, int(time.time()))
                imutil.show(img, filename=filename, resize_to=(256,256), caption="Samples scale {}".format(gan_scale))

                print("Autoencoder Reconstructions scale {}:".format(gan_scale))
                aac_before = images[:8]
                aac_after = netG(netE(aac_before, gan_scale), gan_scale)
                filename = "{}/images/reconstruction_{}_{}.jpg".format(result_dir, gan_scale, int(time.time()))
                img = torch.cat((aac_before, aac_after))
                imutil.show(img, filename=filename, resize_to=(256,256), caption="Reconstruction scale {}".format(gan_scale))

            bps = (i+1) / (time.time() - start_time)
            ed = errD.data[0]
            eg = errG.data[0]
            ec = errC.data[0]
            acc = correct / max(total, 1)
            msg = '[{}][{}/{}] D:{:.3f} G:{:.3f} C:{:.3f} Acc. {:.3f} {:.3f} batch/sec'
            msg = msg.format(
                  epoch, i+1, len(dataloader),
                  ed, eg, ec, acc, bps)
            print(msg)
    return True
