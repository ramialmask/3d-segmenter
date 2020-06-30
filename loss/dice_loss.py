import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def dice_loss(input_tensor, target_tensor, reduction='sum'):
    input_tensor = torch.sigmoid(input_tensor)
    smooth = 1
    intersection = torch.sum((input_tensor * target_tensor))
    coeff = (2. * intersection + smooth) / (torch.sum(target_tensor) + torch.sum(input_tensor) + smooth)
    diceloss = 1. - coeff
    assert(not torch.isnan(diceloss))
    assert(not torch.isinf(diceloss))


    return diceloss

def dice_loss_2(input_tensor, target_tensor, reduction='sum'):
    props = torch.sigmoid(input_tensor)
    num = props * target_tensor
    
    num = torch.sum(num, dim=4)
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)

    den1 = props*props
    den1 = torch.sum(den1, dim=4)
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)

    den2 = props*props
    den2 = torch.sum(den2, dim=4)
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)

    dice = 2*(num/(den1 + den2))
    dice_eso = dice[:,1:]
    dice_total = -1*torch.sum(dice_eso)/dice_eso.size(0)
    print(f"{dice_eso.shape} | {dice_total.shape}\n")
    return dice_total


class DiceLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='sum'):
        super(DiceLoss, self).__init__(weight)

    def forward(self, input, target):
        return dice_loss(input, target, reduction=self.reduction)

