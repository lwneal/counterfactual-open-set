import os
import network_definitions
import torch
from torch import optim
from torch import nn


def build_networks(num_classes, epoch=None, latent_size=10, batch_size=64, **options):
    networks = {}

    EncoderClass = network_definitions.encoder32
    networks['encoder'] = EncoderClass(latent_size=latent_size)

    GeneratorClass = network_definitions.generator32
    networks['generator'] = GeneratorClass(latent_size=latent_size)

    DiscrimClass = network_definitions.multiclassDiscriminator32
    networks['discriminator'] = DiscrimClass(num_classes=num_classes, latent_size=latent_size)

    for net_name in networks:
        pth = get_pth_by_epoch(options['result_dir'], net_name, epoch)
        if pth:
            print("Loading {} from checkpoint {}".format(net_name, pth))
            networks[net_name].load_state_dict(torch.load(pth))
        else:
            print("Using randomly-initialized weights for {}".format(net_name))
    return networks


def get_network_class(name):
    if type(name) is not str or not hasattr(network_definitions, name):
        print("Error: could not construct network '{}'".format(name))
        print("Available networks are:")
        for net_name in dir(network_definitions):
            classobj = getattr(network_definitions, net_name)
            if type(classobj) is type and issubclass(classobj, nn.Module):
                print('\t' + net_name)
        exit()
    return getattr(network_definitions, name)


def save_networks(networks, epoch, result_dir):
    for name in networks:
        weights = networks[name].state_dict()
        filename = '{}/{}_epoch_{:04d}.pth'.format(result_dir, name, epoch)
        torch.save(weights, filename)


def get_optimizers(networks, lr=.0001, beta1=.5, beta2=.999, weight_decay=.0, **options):
    optimizers = {}
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizers


def get_latest_pth(result_dir, name):
    if not os.path.isdir(result_dir):
        return None
    files = os.listdir(result_dir)
    files = [f for f in files if f.startswith(name) and f.endswith('.pth')]
    if not files:
        return None
    files = [os.path.join(result_dir, f) for f in files]
    ordered_by_mtime = sorted(files, key=lambda x: os.stat(x).st_mtime)
    return ordered_by_mtime[-1]
    

def get_pth_by_epoch(result_dir, name, epoch):
    if epoch == None:
        return get_latest_pth(result_dir, name)
    files = os.listdir(result_dir)
    suffix = 'epoch_{:04d}.pth'.format(epoch)
    files = [f for f in files if f.startswith(name) and f.endswith(suffix)]
    if not files:
        print("WARNING: No file available for network {} epoch {}".format(name, epoch))
        return None
    return os.path.join(result_dir, files[0])
