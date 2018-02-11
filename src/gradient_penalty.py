import torch
from torch import autograd

def calc_gradient_penalty(netD, real_data, fake_data, penalty_lambda=10.0):
    alpha = torch.rand(real_data.size()[0], 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()
    
    # Traditional WGAN-GP
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    # Possibly more reasonable
    #interpolates = torch.cat([real_data, fake_data])
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size()).cuda()
    gradients = autograd.grad(
            outputs=disc_interpolates, 
            inputs=interpolates, 
            grad_outputs=ones, 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda
    return penalty
