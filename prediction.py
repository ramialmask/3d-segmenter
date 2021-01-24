import os
import shutil
import torch

import numpy as np
import pandas as pd

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from models.unet_3d_oliver import Unet3D
from models.deep_vessel_3d import Deep_Vessel_Net_FC
from statistics import calc_statistics, calc_metrices_stats
from loss.dice_loss import DiceLoss
from loss.cl_dice_loss import CenterlineDiceLoss, MixedDiceLoss, WBCECenterlineLoss
from loss.weighted_binary_cross_entropy_loss import WeightedBinaryCrossEntropyLoss
from loaders import *
from util import *
from blobanalysis import get_patch_overlap
from dataset.training_dataset import TrainingDataset
from classify_patches import classify_patch

def _net():
    net = Unet3D()
    return net

def predict(settings):
    print("Reading parameters")
    input_list = os.listdir(settings["prediction"]["input_path"])
    model_name = settings["prediction"]["model_name"]
    model_path = settings["prediction"]["model_path"]
    output_path= settings["prediction"]["output_path"]
    model_type = settings["network"] #TODO
    
    print("Loading model")
    net = _net()
    net.load_model(model_path + model_name)
    
    print("Get dataloader")
    dataloader, dataset = get_loader(settings, input_list, True)

    net.cuda()
    net.eval()

    print("Start prediction preparations")
    d_len = len(dataloader)

    item_dict = -1
    if not dataset.original_information()[2] == -1:
        # In order to save patches, the subvolumes need to be saved in 
        # a dict grouped by their name
        item_dict = {}
    else:
        reconstructed_patches = []
    
    print("Predicting...")
    for item in dataloader:
        volume       = item["volume"]
        segmentation = item["segmentation"]
        volume       = volume.cuda()
        segmentation = segmentation.cuda()

        logits       = net(volume)

        res          = logits.detach().cpu().numpy()
        res[res > 0.5]  = 1.0 
        res[res <= 0.5] = 0.0

        item_name = item["name"][0]
        if not item_dict == -1:
            if item_name not in item_dict:
                item_dict[item_name] = []
            item_dict[item_name].append(res.astype(np.float32))
        else:
            res = res.squeeze().squeeze()
            reconstructed_patches.append([item_name, res.astype(dataset.original_information()[1])])

        res             = np.ravel(res)
        target          = np.ravel(segmentation.cpu().numpy())

    print("Reconstruct patches")
    if not item_dict == -1:
        reconstructed_patches = reconstruct_patches(item_dict, dataset)
    
    print("Writing patches")
    for item_name, reconstructed_prediction in reconstructed_patches:
        item_save_path = f"{output_path}/{item_name}"
        write_nifti(item_save_path, reconstructed_prediction)
        


# Initialize cuda
torch.cuda.init()
torch.cuda.set_device(0)

settings = read_meta_dict("./","predict")

predict(settings)

