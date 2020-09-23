import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
"""
- Load all datasets into one huge list
- iterate over this list as method of getitem
Pro:
    faster access to data (already in ram)
    no awkwardly large getitem method
    no multilist (dict of list of lists) as item but a simple dict
Cons:
    needs more ram 
    longer setup time (init should load stuff into RAM)
"""

def cut_volume(item, name):
    shape = item.shape
    new_shape = (100, 100, 100)
    width = int(shape[0] / new_shape[0])
    counter = 0
    result_list = []
    for x in range(0, width):
        for y in range(0, width):
            for z in range(0, width):
                sub_item = item[x*new_shape[0]:x*new_shape[0]+new_shape[0],y*new_shape[0]:y*new_shape[0]+new_shape[0], z*new_shape[0]:z*new_shape[0] +new_shape[0]]
                out_name = name.replace(".nii.gz",f"_{counter}.nii.gz")
                result_list.append((out_name, sub_item))
                counter+=1                  
    return result_list

def prepare_lists(settings, item, image, image_list, name_list, transform, norm):
    
    image = image.astype(np.int64)

    if transform:
        image       = transform(image)

    if norm:
        image       = norm(image)

    for z in range(image.shape[-1]):
        # Pad
        padding_value = int(settings["preprocessing"]["padding"])
        image_z = image[:,:,z]
        image_z = np.pad(image_z, padding_value, "reflect")
    

        # Torchify
        image_z = torch.tensor(image_z).float()
        image_z = image_z.unsqueeze(0)

        image_list.append(image_z)
        name_list.append(f"{z}${item}")


class PredictionDataset2D(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]

        # create list
        image_list = []
        name_list = []

        split_len = len(split)

        # Load data
        for item_no, item in enumerate(split):
            print(f"Loading item {item_no} of {split_len}", end="\r", flush=True)
            item_nii_path   = os.path.join(nii_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)

            # TODO parameterize
            if image.shape[0] > 100:
                cut_images = cut_volume(image, item)
                for cut_name, cut_image in cut_images:
                    self.original_shape, self.original_type = image.shape, image.dtype
                    prepare_lists(settings, cut_name, cut_image, image_list, name_list, transform, norm)
            else:
                self.original_shape, self.original_type = image.shape, image.dtype
                prepare_lists(settings, item, image, image_list, name_list, transform, norm)

            

        self.item_list = [image_list, name_list]

    def original_information(self):
        return self.original_shape, self.original_type

    def __len__(self):
        return len(self.item_list[0])

    def __getitem__(self, idx):
        return {"volume"        :self.item_list[0][idx],\
                "name"          :self.item_list[1][idx]}
