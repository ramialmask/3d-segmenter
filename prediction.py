import torch

import numpy as np

from models.unet_2d import Unet2D
from loaders import read_meta_dict, get_prediction_loader
from util import *
from loaders import get_loader


def _net(settings):
    net_class       = settings["network"]
    model_path      = settings["paths"]["input_model_path"] + settings["paths"]["input_model"]
    print(f"Loading model {model_path}")
    net             = globals()[net_class]()
    net.load_model(model_path)
    net             = net.cuda()
    return net


def prediction(settings):
    net = _net(settings)
    input_list = os.listdir(settings["paths"]["input_raw_path"])
    dataloader, dataset = get_loader(settings,input_list,False,True)
    batchsize  = int(settings["dataloader"]["batch_size"])

    prediction_path = settings["paths"]["output_prediction_path"] + settings["paths"]["output_folder_prefix"]


    item_dict = {}
    reconstructed_patches = []

    orig_shape, orig_type, mb_size = dataset.original_information()

    len_loader = len(dataloader)
    for item_index, item in enumerate(dataloader):
        item_name = item["name"][0]
        print(f"Predicting {item_name} {item_index}/{len_loader}", end="\r", flush=True)
        volume       = item["volume"]
        volume       = volume.cuda()

        logits       = net(volume)

        res             = logits.detach().cpu().numpy()
        res[res > 0.5] = 1.0 
        res[res <= 0.5] = 0.0


        item_z = int(item_name.split("$")[0])
        item_image = item_name.split("$")[1]
        

        if item_image in item_dict.keys():
            item_dict[item_image].append((item_z, res.squeeze().squeeze()))
        else:
            item_dict[item_image] = [(item_z, res.squeeze().squeeze())]
        


    if orig_shape[0] > mb_size:
        intermediate_patches    = reconstruct_patches_2d(item_dict, dataset)
        reconstructed_patches   = dict_to_patches(intermediate_patches, orig_shape)
    else:
        reconstructed_patches   = reconstruct_patches_2d(item_dict, dataset)
        
    for item_name, reconstructed_prediction in reconstructed_patches:
        item_save_path      = f"{prediction_path}/{item_name}"

        print(f"Writing {item_save_path}")
        write_nifti(item_save_path, reconstructed_prediction)            

print(f"\n\nPytorch version {torch.__version__}\n\n")

# Initialize cuda
torch.cuda.init()
torch.cuda.set_device(0)

# Read the meta dictionary
settings = read_meta_dict("./","predict")

# Run the program
prediction(settings)
