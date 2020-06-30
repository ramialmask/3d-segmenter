import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Define Network
class Deep_Vessel_Net_FC(nn.Module):
    def __init__(self, print_=False):
        super(Deep_Vessel_Net_FC, self).__init__()
        self.num_channels = 1
        # ONLY FOR TESTING PURPOSES
        self.print=print_

        # 1. 3x3x3-Conv, 2 -> 5
        self.conv1 = nn.Conv3d(self.num_channels, 5, kernel_size=(3,3,3)) #nn.Conv3d
        # 2. 5x5x5-Conv, 5 -> 10
        self.conv2 = nn.Conv3d(5, 10, kernel_size=(5,5,5))
        # 3. 5x5x5-Conv, 10-> 20
        self.conv3 = nn.Conv3d(10, 20, kernel_size=(5,5,5))
        # 4. 3x3x3-Conv, 20-> 50
        self.conv4 = nn.Conv3d(20, 50, kernel_size=(3,3,3))
        # 5. FC
        self.conv5 = nn.Conv3d(50,1, kernel_size=(1,1,1))#, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        t_ = torch.load(path)
        print("T_ {}".format(t_))
        self.load_state_dict(t_)
        print("Loaded model from {0}".format(str(path)))

