import torch
from torch.autograd import Variable


def gen_noise(K, latent_size):
    noise = torch.zeros((K, latent_size))
    noise.normal_(0, 1)
    noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x):
    #norm = torch.norm(x, p=2, dim=1)
    #norm = norm.expand(1, x.size()[0])
    #return torch.mul(x, 1/norm.t())

    # Split the latent space into pieces, normalize each one
    GRID_SIZE = 4
    batch_size, latent_size = x.shape
    latent_subspaces = []
    for i in range(GRID_SIZE):
        step = latent_size // GRID_SIZE
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone()
        norm = torch.norm(subspace, p=2, dim=1)
        subspace /= norm.expand(1, -1).t()  # + epsilon
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together
    return torch.cat(latent_subspaces, dim=1)

