import torch


def gen_noise(K, latent_size):
    noise = torch.zeros((K, latent_size))
    noise.normal_(0, 1)
    noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x):
    norm = torch.norm(x, p=2, dim=1)
    norm = norm.expand(1, x.size()[0])
    return torch.mul(x, 1/norm.t())


