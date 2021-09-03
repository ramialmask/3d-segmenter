import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from loss.dice_loss import dice_loss
from loss.weighted_binary_cross_entropy_loss import wbce

class MixedDiceLoss(_Loss):
    def __init__(self, split=0.5, weight=None, size_average=None, reduce=None, reduction='sum'):
        super(MixedDiceLoss, self).__init__(weight)
        self.split = split
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return self.split * dice_loss(input, target, reduction=self.reduction) + \
                (1- self.split) * wbce(input, target, weights=self.weight, reduction=self.reduction)
