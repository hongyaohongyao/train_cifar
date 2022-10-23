import torch
import numpy as np
from torch.autograd import Variable


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_train(net,
                inputs,
                targets,
                criterion,
                args,
                use_cuda=True,
                **kwargs):
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                   args.mixup_alpha, use_cuda)
    inputs, targets_a, targets_b = map(Variable,
                                       (inputs, targets_a, targets_b))
    outputs = net(inputs)
    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    _, predicted = torch.max(outputs.data, 1)
    correct = (lam * predicted.eq(targets_a.data).cpu().sum().float() +
               (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
               ) / targets.size(0)
    return loss, correct
