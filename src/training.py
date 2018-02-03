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
    netE1 = networks['encoder']
    netE2 = networks['encoder2']

    netD = networks['discriminator']

    netG = networks['generator']

    netC1 = networks['classifier']
    netC2 = networks['classifier2']

    optimizerE1 = optimizers['encoder']
    optimizerE2 = optimizers['encoder2']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    optimizerC1 = optimizers['classifier']
    optimizerC2 = optimizers['classifier2']

    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    def make_noise(scale, latent_size):
        noise_t = torch.FloatTensor(batch_size, latent_size * scale * scale)
        noise_t.normal_(0, 1)
        noise = Variable(noise_t).cuda()
        return clamp_to_unit_sphere(noise, scale)

    aux_dataloader = None
    dataset_filename = options.get('aux_dataset')
    if dataset_filename and os.path.exists(dataset_filename):
        print("Using aux_dataset {}".format(dataset_filename))
        aux_dataloader = FlexibleCustomDataloader(dataset_filename, batch_size=batch_size, image_size=image_size)

    start_time = time.time()
    good_correct = 0
    bad_correct = 0
    total = 0

    log = TimeSeries()

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        #gan_scale = random.choice([1, 2, 4, 8])
        gan_scale = 1
        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify sampled images as fake
        noise = make_noise(gan_scale, latent_size*2)
        fake_images = netG(noise, gan_scale)
        fake_logits = netD(fake_images)
        augmented_logits = F.pad(fake_logits, pad=(0, 1))
        log_prob_fake = -(F.log_softmax(augmented_logits, dim=1)[:, -1]).mean()
        log.collect('errD_fake', log_prob_fake)

        # Classify real examples into the correct K classes
        real_logits = netD(images)
        augmented_logits = F.pad(real_logits, pad=(0, 1))
        positive_labels = (labels == 1).type(torch.cuda.FloatTensor)
        augmented_labels = F.pad(positive_labels, pad=(0, 1))
        log_prob_real = -(F.log_softmax(augmented_logits, dim=1) * augmented_labels).mean()
        log.collect('errD_real', log_prob_real)

        errD = (log_prob_fake.mean() + log_prob_real.mean()) * options['discriminator_weight']
        errD.backward()
        log.collect('errD', errD)

        optimizerD.step()
        ############################

        ############################
        # Generator Update
        ###########################
        netG.zero_grad()
        netE1.zero_grad()
        netE2.zero_grad()

        # Minimize fakeness of sampled images
        noise = make_noise(gan_scale, latent_size*2)
        fake_images = netG(noise)
        fake_logits = netD(fake_images)
        augmented_logits = F.pad(fake_logits, pad=(0, 1))
        log_prob_not_fake = F.log_softmax(-augmented_logits, dim=1)[:, -1]
        errG = -log_prob_not_fake.mean() * options['generator_weight']
        errG.backward()
        log.collect('errG', errG)

        ############################
        # Encoder Update
        ###########################
        # Minimize reconstruction loss
        z1 = netE1(images)
        z2 = netE2(images)
        reconstructed = netG(torch.cat([z1, z2], dim=1))
        err_reconstruction = torch.mean(torch.abs(images - reconstructed)) * options['reconstruction_weight']
        err_reconstruction.backward()
        log.collect('err_reconstruction', err_reconstruction)

        optimizerG.step()
        optimizerE1.step()
        optimizerE2.step()
        ###########################

        ############################
        # Cooperative classifier and encoder
        ############################
        netC1.zero_grad()
        netE1.zero_grad()

        # Classify real examples into the correct K classes with hinge loss
        classifier_logits = netC1(netE1(images))
        #errC1 = F.relu(classifier_logits * labels).mean()
        _, label_idx = labels.max(dim=1)
        errC1 = nll_loss(log_softmax(classifier_logits, dim=1), label_idx)
        errC1.backward()
        log.collect('errC1', errC1)

        optimizerC1.step()
        optimizerE1.step()
        ############################

        ############################
        # Adversarial classifier and encoder
        ############################
        netC2.zero_grad()
        # Classify real examples into the correct K classes with hinge loss
        classifier_logits = netC2(netE2(images))
        _, label_idx = labels.max(dim=1)
        errC2 = nll_loss(log_softmax(classifier_logits, dim=1), label_idx)
        errC2.backward()
        log.collect('errC2', errC2)
        optimizerC2.step()


        # Force netE2 to be blind to class
        netE2.zero_grad()
        classifier_logits = netC2(netE2(images))
        _, label_idx = labels.max(dim=1)
        errC2Adv = (softmax(classifier_logits, dim=1) * log_softmax(classifier_logits, dim=1)).mean()
        errC2Adv.backward()
        log.collect('errC2Adv', errC2Adv)
        optimizerE2.step()
        ###########################



        # Keep track of accuracy on positive-labeled examples for monitoring
        good_logits = netC1(netE1(images))
        bad_logits = netC2(netE2(images))
        _, good_pred_idx = good_logits.max(1)
        _, bad_pred_idx = bad_logits.max(1)
        _, label_idx = labels.max(1)

        good_correct += sum(good_pred_idx == label_idx).data.cpu().numpy()[0]
        bad_correct += sum(bad_pred_idx == label_idx).data.cpu().numpy()[0]
        total += len(labels)

        if i % 10 == 0:
            if i % 100 == 0:
                #for gan_scale in (8, 4, 2, 1):
                for gan_scale in [1]:
                    seed()
                    fixed_noise = make_noise(gan_scale, latent_size*2)
                    seed(int(time.time()))
                    print("Generator Samples scale {}:".format(gan_scale))
                    demo_fakes = netG(fixed_noise, gan_scale)
                    img = demo_fakes.data[:16]
                    filename = "{}/images/samples_{}_{}.jpg".format(result_dir, gan_scale, int(time.time()))
                    imutil.show(img, filename=filename, resize_to=(256,256), caption="Samples scale {}".format(gan_scale))

                    print("Autoencoder Reconstructions scale {}:".format(gan_scale))
                    aac_before = images[:8]
                    z1 = netE1(images[:8])
                    z2 = netE2(images[:8])
                    aac_after = netG(torch.cat([z1, z2], dim=1))
                    filename = "{}/images/reconstruction_{}_{}.jpg".format(result_dir, gan_scale, int(time.time()))
                    img = torch.cat((aac_before, aac_after))
                    imutil.show(img, filename=filename, resize_to=(256,256), caption="Reconstruction scale {}".format(gan_scale))

                    print("Dual Latent Space Crossover {}:".format(gan_scale))
                    img = netG(torch.cat([netE1(images[:16]), make_noise(gan_scale, latent_size)[:16]], dim=1))
                    filename = "{}/images/change_content_{}_{}.jpg".format(result_dir, gan_scale, int(time.time()))
                    imutil.show(img, filename=filename, resize_to=(256,256), caption="Noise for z2 {}".format(gan_scale))

                    print("Dual Latent Space Crossover {}:".format(gan_scale))
                    img = netG(torch.cat([make_noise(gan_scale, latent_size)[:16], netE2(images[:16])], dim=1))
                    filename = "{}/images/change_style_{}_{}.jpg".format(result_dir, gan_scale, int(time.time()))
                    imutil.show(img, filename=filename, resize_to=(256,256), caption="Noise for z1 {}".format(gan_scale))

            log.collect("GoodAcc", good_correct / total)
            log.collect("BadAcc", bad_correct / total)
            print(log)

            """
            bps = (i+1) / (time.time() - start_time)
            ed = errD.data[0]
            eg = errG.data[0]
            ec = errC.data[0]
            er = err_reconstruction.data[0]
            good_acc = correct / max(total, 1)
            bad_acc = correct / max(total, 1)
            msg = '[{}][{}/{}] AAC: {:.3f} D:{:.3f} G:{:.3f} C:{:.3f} GoodAcc. {:.3f} BadAcc. {:.3f} {:.3f} batch/sec'
            msg = msg.format(
                  epoch, i+1, len(dataloader),
                  er, ed, eg, ec, acc, bps)
            print(msg)
            """
    return True
