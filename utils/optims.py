import math
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.adam import Adam
from torch.optim.sgd import SGD


def split_params(model: nn.Module):
    param_other, param_weight_decay, param_bias = list(), list(), list()  # optimizer parameter groups
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                param_bias.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                param_weight_decay.append(v)  # apply weight decay
            else:
                param_other.append(v)  # all else
    return param_weight_decay, param_bias, param_other


def split_optimizer(model: nn.Module, cfg: dict):
    param_weight_decay, param_bias, param_other = split_params(model)
    if cfg['optimizer'] == 'Adam':
        optimizer = Adam(param_other, lr=cfg['lr'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = SGD(param_other, lr=cfg['lr'], momentum=cfg['momentum'])
    else:
        raise NotImplementedError("optimizer {:s} is not support!".format(cfg['optimizer']))
    optimizer.add_param_group(
        {'params': param_weight_decay, 'weight_decay': cfg['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': param_bias})
    return optimizer


def cosine_lr_scheduler(optimizer, epochs):
    l_f = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1
    return LambdaLR(optimizer, lr_lambda=l_f)


class WarmUpCosineDecayLRAdjust(object):
    def __init__(self, init_lr, epochs, warm_up_epoch=1, iter_per_epoch=1000, final_ratio=0.01, decay_rate=1.0):
        self.init_lr = init_lr
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.warm_up_iters = warm_up_epoch * iter_per_epoch
        self.epochs = epochs
        self.final_ratio = final_ratio
        self.decay_rate = decay_rate

    def cosine_lr(self, epoch):
        lr_weighs = (((1 + math.cos(epoch * math.pi / (self.epochs - self.warm_up_epoch - 1))) / 2) ** self.decay_rate) \
                    * (1 - self.final_ratio) + self.final_ratio
        return lr_weighs

    def linear_lr(self, iter):
        return 1. / self.warm_up_iters + iter / self.warm_up_iters

    def get_lr(self, iter, epoch):
        if epoch < self.warm_up_epoch:
            lr_weights = self.linear_lr(self.iter_per_epoch * epoch + iter)
        else:
            lr_weights = self.cosine_lr(epoch - self.warm_up_epoch)
        return lr_weights

    def __call__(self, optimizer, iter, epoch):
        lr_weights = self.get_lr(iter, epoch)
        lr = lr_weights * self.init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class WarmUpCosineDecayMultiStepLRAdjust(object):
    def __init__(self, init_lr, epochs, milestones, warm_up_epoch=1, cosine_weights=1.0, iter_per_epoch=1000):
        self.init_lr = init_lr
        self.epochs = epochs
        self.milestones = milestones
        self.cosine_weights = cosine_weights
        self.warm_up_epoch = warm_up_epoch
        self.iter_per_epoch = iter_per_epoch
        self.warm_up_iter = warm_up_epoch * iter_per_epoch

    def cosine_lr(self, top_iter, sub_iter):
        return ((1 + math.cos(top_iter * math.pi / sub_iter)) / 2) ** self.cosine_weights * 0.9 + 0.1

    def linear_lr(self, iter):
        return 1. / self.warm_up_iter + iter / self.warm_up_iter

    def get_lr(self, iter, epoch):
        if epoch < self.warm_up_epoch:
            lr_weights = self.linear_lr(self.iter_per_epoch * epoch + iter)
        else:
            pow_num = (np.array(self.milestones) <= epoch).sum().astype(np.int)
            if pow_num == 0:
                current_iter = (epoch - self.warm_up_epoch) * self.iter_per_epoch + iter
                sub_iter = (self.milestones[0] - self.warm_up_epoch) * self.iter_per_epoch - 1
                lr_weights = self.cosine_lr(current_iter, sub_iter)
            elif pow_num == len(self.milestones):
                lr_weights = 0.1 ** pow_num
            else:
                current_iter = (epoch - self.milestones[pow_num - 1]) * self.iter_per_epoch + iter
                sub_iter = (self.milestones[pow_num] - self.milestones[pow_num - 1]) * self.iter_per_epoch - 1
                lr_weights = 0.1 ** pow_num * self.cosine_lr(current_iter, sub_iter)
        return lr_weights

    def __call__(self, optimizer, iter, epoch):
        lr_weights = self.get_lr(iter, epoch)
        lr = lr_weights * self.init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
