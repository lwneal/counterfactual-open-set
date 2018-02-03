import os
import network_definitions
import torch
from torch import optim
from torch import nn
from imutil import ensure_directory_exists


def build_networks(num_classes, epoch=None, latent_size=10, batch_size=64, **options):
    networks = {}

    # One encoder learns to encode the CLASS of the input example
    EncoderClass = network_definitions.encoder32
    networks['encoder'] = EncoderClass(latent_size=latent_size)

    # Another encoder learns to encode all information EXCEPT the class
    EncoderClass = network_definitions.encoder32
    networks['encoder2'] = EncoderClass(latent_size=latent_size)

    # Generator now takes as input TWO latent codes, one from each encoder
    GeneratorClass = network_definitions.generator32
    networks['generator'] = GeneratorClass(latent_size=latent_size * 2)

    DiscrimClass = network_definitions.multiclassDiscriminator32
    networks['discriminator'] = DiscrimClass(num_classes=num_classes, latent_size=latent_size)

    # One classifier cooperates with the first encoder to properly encode class
    ClassifierClass = network_definitions.classifier32
    networks['classifier'] = ClassifierClass(num_classes=num_classes, latent_size=latent_size)

    # The other classifier adversarially prevents the second encoder from learning class
    ClassifierClass = network_definitions.classifier32
    networks['classifier2'] = ClassifierClass(num_classes=num_classes, latent_size=latent_size)

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
        filename = '{}/checkpoints/{}_epoch_{:04d}.pth'.format(result_dir, name, epoch)
        ensure_directory_exists(filename)
        torch.save(weights, filename)


def get_optimizers(networks, lr=.0001, beta1=.5, beta2=.999, weight_decay=.0, **options):
    optimizers = {}
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizers


def get_pth_by_epoch(result_dir, name, epoch=None):
    checkpoint_path = os.path.join(result_dir, 'checkpoints/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    suffix = '.pth'
    if epoch is not None:
        suffix = 'epoch_{:04d}.pth'.format(epoch)
    files = [f for f in files if f.startswith(name) and f.endswith(suffix)]
    if not files:
        return None
    files = [os.path.join(checkpoint_path, fn) for fn in files]
    files.sort(key=lambda x: os.stat(x).st_mtime)
    return files[-1]
