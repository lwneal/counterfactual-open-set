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
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    seed()
    fixed_noise = Variable(torch.FloatTensor(batch_size, latent_size).normal_(0, 1)).cuda()
    fixed_noise = clamp_to_unit_sphere(fixed_noise)

    start_time = time.time()
    correct = 0
    total = 0

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images)
        labels = Variable(class_labels)

        ############################
        # Generator Updates
        ############################
        netG.zero_grad()
        z = gen_noise(batch_size, latent_size)
        z = Variable(z).cuda()
        gen_images = netG(z)
        
        # Feature Matching: Average of one batch of real vs. generated
        features_real = netD(images, return_features=True)
        features_gen = netD(gen_images, return_features=True)
        fm_loss = torch.mean((features_real.mean(0) - features_gen.mean(0)) ** 2)

        # Pull-away term from https://github.com/kimiyoung/ssl_bad_gan
        nsample = features_gen.size(0)
        denom = features_gen.norm(dim=0).expand_as(features_gen)
        gen_feat_norm = features_gen / denom
        cosine = torch.mm(features_gen, features_gen.t())
        mask = Variable((torch.ones(cosine.size()) - torch.diag(torch.ones(nsample))).cuda())
        pt_loss = torch.sum((cosine * mask) ** 2) / (nsample * (nsample + 1))
        pt_loss /= (128 * 128)

        errG = fm_loss
        if options['pt_loss'] != 0:
            errG += options['pt_loss'] * pt_loss

        # Classify generated examples as "not fake"
        gen_logits = netD(gen_images)
        augmented_logits = F.pad(-gen_logits, pad=(0,1))
        log_prob_gen = F.log_softmax(augmented_logits, dim=1)[:, -1]
        errG += -log_prob_gen.mean()

        errG.backward()
        optimizerG.step()
        ###########################

        ############################
        # Discriminator Updates
        ###########################
        netD.zero_grad()

        # Classify generated examples as "fake" (ie the K+1th "open" class)
        z = gen_noise(batch_size, latent_size)
        z = Variable(z).cuda()
        fake_images = netG(z).detach()
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
        #log_prob_real = F.log_softmax(augmented_logits, dim=1)[:, 0]
        errC = -log_prob_real.mean()
        errC.backward()

        optimizerD.step()
        ############################

        # TODO: Encoder updates

        # Keep track of accuracy on positive-labeled examples for monitoring
        _, pred_idx = real_logits.max(1)
        _, label_idx = labels.max(1)
        correct += sum(pred_idx == label_idx).data.cpu().numpy()[0]
        total += len(labels)

        if i % 100 == 0:
            demo_fakes = netG(fixed_noise)
            img = torch.cat([demo_fakes.data[:36]])
            filename = "{}/demo_{}.jpg".format(result_dir, int(time.time()))
            imutil.show(img, filename=filename, resize_to=(512,512))

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
            print("log_prob_real {:.3f}".format(log_prob_real.mean().data[0]))
            print("log_prob_fake {:.3f}".format(log_prob_fake.mean().data[0]))
            print("log_prob_gen {:.3f}".format(log_prob_gen.mean().data[0]))
            print("pt_loss {:.3f}".format(pt_loss.data[0]))
            print("fm_loss {:.3f}".format(fm_loss.data[0]))
            print("Accuracy {}/{}".format(correct, total))
    return True
