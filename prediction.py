import torch

import numpy as np

from models.unet_2d import Unet2D
from loaders import read_meta_dict, get_prediction_loader
from util import *


def _net(settings):
    net_class       = settings["network"]
    model_path      = settings["paths"]["input_model_path"] + settings["paths"]["input_model"]
    net             = globals()[net_class]()
    net.load_model(model_path)
    net             = net.cuda()
    return net

def predict(net, dataloader, dataset, batchsize):
    net.eval()
    item_dict = {}
    d_len = len(dataloader)
    c = 0
    print(f"Predicting {d_len} items...")

    for item in dataloader:
        c += 1
        print(f"Predicting item {c} of {d_len}...",end="\r",flush=True)
        volume  = item["volume"]
        volume  = volume.cuda()
        logits  = net(volume)
        res     = logits.detach().cpu().numpy()

        res[res > 0.5]  = 1.0
        res[res <=0.5]  = 0.0
        
        for batch in range(batchsize):
            item_name   = item["name"][batch]
            item_z      = item_name.split("$")[0]
            item_image  = item_name.split("$")[1]

            if item_image in item_dict.keys():
                item_dict[item_image].append((item_z, res[batch].squeeze()))
            else:
                item_dict[item_image] = [(item_z, res[batch].squeeze())]

    reconstructed_patches = reconstruct_patches_2d(item_dict, dataset)
    print("\nPrediction finished.")
    return reconstructed_patches

def prediction(settings):
    net = _net(settings)
    dataloader, dataset = get_prediction_loader(settings)
    batchsize  = int(settings["dataloader"]["batch_size"])
    reconstructed_patches = predict(net, dataloader, dataset, batchsize)

    prediction_path = settings["paths"]["output_prediction_path"] + settings["paths"]["output_folder_prefix"]
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    print(f"Writing {len(reconstructed_patches)} items...")
    _dict_to_patches(reconstructed_patches, prediction_path)
    print("\nDone.")

def _dict_to_patches(patch_list, prediction_path):
    #TODO empty dict
    #TODO if name w/o _ exists put all n_ there else make new entry
    #TODO use util thing
    big_dict = {}
    for item_name, patch in patch_list:
        item_index = item_name.split("_")[1]
        item_sub_index = item_name.split("_")[2].replace(".nii.gz","")
        if not item_index in big_dict.keys():
            big_dict[item_index] = {}
        big_dict[item_index][item_sub_index] = patch

    for item_name in big_dict.keys():
        p_l = []
        for i in range(8): # TODO
            p_l.append(big_dict[item_name][str(i)])
        patch = stitch_volume(p_l, (200,200,200)) # TODO
        item_save_path  = f"{prediction_path}/patchvolume_{item_name}.nii.gz"
        print(f"Writing {item_save_path}", end="\r",flush=True)
        write_nifti(item_save_path, patch)
        
# Initialize cuda
torch.cuda.init()
torch.cuda.set_device(0)

# Read the meta dictionary
settings = read_meta_dict("./","predict")

# Run the program
prediction(settings)
