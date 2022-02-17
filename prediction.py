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
    from monai.networks.nets import unet
    net = unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(4,8,16),
            strides=(2,2,2),
            num_res_units=4
    )
    t_ = torch.load(model_path)
    net.load_state_dict(t_)
    net             = net.cuda()
    return net


def prediction(settings, index_start=-1, index_end=-1):
    net = _net(settings)
    if index_start == -1:
        input_list = os.listdir(settings["paths"]["input_raw_path"])
    else:
        input_list = os.listdir(settings["paths"]["input_raw_path"])[index_start:index_end]
    dataloader, dataset = get_loader(settings,input_list,False,True)
    batchsize  = int(settings["dataloader"]["batch_size"])

    prediction_path = settings["paths"]["output_prediction_path"] + settings["paths"]["output_folder_prefix"]
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)


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


        if not item_dict == -1:
            if item_name not in item_dict:
                item_dict[item_name] = []
            item_dict[item_name].append(res.astype(np.float32))
        else:
            res = res.squeeze().squeeze()
            reconstructed_patches.append([item_name, res.astype(dataset.original_information()[1])])
        


    if not item_dict == -1:
        reconstructed_patches = reconstruct_patches(item_dict, dataset)
        
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
len_p = len(os.listdir(settings["paths"]["input_raw_path"]))
step_s = 200
if len_p <= step_s:
    prediction(settings, 0, len_p)
else:
    for step in range(0, len_p, step_s):
        print(f"Predicting {step} - {step + step_s}/{len_p}")
        if len_p - step < step_s:
            prediction(settings, step, len_p-1)
        else:
            prediction(settings, step, step+step_s)
