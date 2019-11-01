# Utilities for noise generation, clamping etc
import torch
import time
import numpy as np
from torch.autograd import Variable


def make_noise(batch_size, latent_size, scale, fixed_seed=None):
    noise_t = torch.FloatTensor(batch_size, latent_size * scale * scale)
    if fixed_seed is not None:
        seed(fixed_seed)
    noise_t.normal_(0, 1)
    noise = Variable(noise_t).cuda()
    result = clamp_to_unit_sphere(noise, scale**2)
    if fixed_seed is not None:
        seed(int(time.time()))
    return result


def seed(val=42):
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)


# TODO: Merge this with the more fully-featured make_noise()
def gen_noise(K, latent_size):
    noise = torch.zeros((K, latent_size))
    noise.normal_(0, 1)
    noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace = subspace / norm.expand(1, -1).t()  # + epsilon
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)
