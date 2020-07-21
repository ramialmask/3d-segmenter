import sys
import torch
from torch.nn.modules.loss import _WeightedLoss
import numpy as np

def inv_class_frequency(input_tensor):
    un = uniques(input_tensor)
    if len(un[0]) > 1:
        weights = [float(un[1][0]) / np.prod(input_tensor.size(), dtype=np.float),
            float(un[1][1]) / np.prod(input_tensor.size(), dtype=np.float)]
        return weights
    else:
        return [0,0]

def calc_class_frequency(i_t):
    input_tensor = i_t.clone().detach().cpu()
    n_total = input_tensor.view(-1).size()[0]
    n_FG    = input_tensor.sum().numpy()
    n_BG    = (1-input_tensor).sum().numpy()
    assert(n_total == n_FG + n_BG)  # will fail for non-binary input volumes
    eps = 0.00001
    weights = [n_total/(n_FG + eps), n_total/(n_BG + eps)]

    return weights

def uniques(volume):
    """Extending torchs uniques method to perform like numpy.uniques(count=True)
    Args:
        volume (torch.tensor)   :   Input volume
    Returns:
        list (torch.tensor)     :   Unique values in the input volume
        list (int)              :   Count of unique values
    """
    volume_uniques = volume.unique(sorted=True)
    vl = [(volume == i).sum() for i in volume_uniques]
    volume_uniques_count = torch.stack(vl)
    return volume_uniques.type(volume.type()), volume_uniques_count.type(volume.type())

def bce(input_tensor, target_tensor, weights=None, class_frequency=False, reduction='mean'):
    # Calculate Class Frequency
    if class_frequency:
        weights = calc_class_frequency(target_tensor)

    # If weights are given or class frequency is activated calculate with weights
    # Add 0.00001 to take into account that a normed matrix will contain 0 and 1
    loss_add = 0.00001
    if weights is not None:
        loss = (target_tensor * torch.log(input_tensor + loss_add)) * weights[0] + \
            ((1 - target_tensor) * torch.log(1 - input_tensor + loss_add)) * weights[1]
    else:
        loss = target_tensor * torch.log(input_tensor + loss_add) \
                + (1 - target_tensor) * torch.log(1 - input_tensor + loss_add)

    loss_r = getattr(torch, reduction)(loss) * -1
    if torch.isnan(loss_r):
        print(f"Target {torch.max(target_tensor)}")
        print(f"Input {torch.max(input_tensor)}")
        print(f"Loss {loss_r}")
    assert(not torch.isnan(loss_r))
    assert(not torch.isinf(loss_r))
    return loss_r

class WeightedBinaryCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='sum', class_frequency=False):
        super(WeightedBinaryCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.class_frequency = class_frequency

    def forward(self, input, target):
        return bce(input, target, weights=self.weight, class_frequency=self.class_frequency, reduction=self.reduction)
