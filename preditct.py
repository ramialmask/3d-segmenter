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

import datetime
# TODO
# import settings
# get dataloader batch size 1
# look how its done in testing
# do it


def _net():
    # Make it 2channel
    net = Unet3D()
    return net

def predict(net, dataloader, dataset):
    """Predict a given dataset
    """
    net.eval()
    d_len = len(dataloader)
    

    item_dict = -1
    if not dataset.original_information()[2] == -1:
        # In order to save patches, the subvolumes need to be saved in 
        # a dict grouped by their name
        item_dict = {}
    else:
        reconstructed_patches = []


    for i, item in enumerate(dataloader):
        print(f"Predicting {i}/{d_len}", end="\r", flush=True)
        volume       = item["volume"]

        volume       = volume.cuda()

        logits       = net(volume)

        res             = logits.detach().cpu().numpy()
        res[res > 0.5] = 1.0 
        res[res <= 0.5] = 0.0

        item_name = item["name"][0]
        if not item_dict == -1:
            if item_name not in item_dict:
                item_dict[item_name] = []
            item_dict[item_name].append(res.astype(np.float32))
        else:
            res = res.squeeze().squeeze()
            reconstructed_patches.append([item_name, res.astype(dataset.original_information()[1])])

    if not item_dict == -1:
        reconstructed_patches = reconstruct_patches(item_dict, dataset)
    return reconstructed_patches

# Initialize cuda
torch.cuda.init()
torch.cuda.set_device(0)

# Read the meta dictionary
settings = read_meta_dict("./","predict")

model_name          = settings["prediction"]["input_model_name"]
model_data_path     = settings["prediction"]["input_model_path"] + model_name
output_patch_path   = settings["prediction"]["output_patch_path"] + f"{model_name.replace('.dat','')} | {datetime.datetime.now()}"

os.mkdir(output_patch_path)

print("Loading Net..")
net = _net()
net.load_model(model_data_path)
net = net.cuda()
print("Loaded Net.")


predict_list                       = os.listdir(settings["prediction"]["input_patch_path"])

for i in range(20):
    print(f"Loading data {i}/20...")
    sub_list = predict_list[i*int(len(predict_list)/20):(i+1)*int(len(predict_list)/20)]
    predict_loader, predict_dataset       = get_loader(settings, sub_list, mode=2)
    print("Loaded data.")

    print("Predicting data...")
    reconstructed_patches       = predict(net, predict_loader, predict_dataset)
    print("Predicted data.")

    for item_name, reconstructed_prediction in reconstructed_patches:
        item_save_path = output_patch_path + item_name
        print(f"Writing {item_save_path}", end="\r", flush=True)
        write_nifti(item_save_path, reconstructed_prediction)
