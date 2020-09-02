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

class TrainingDataset2DWeighted(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        gt_path = settings["paths"]["input_gt_path"]
        weight_path = settings["paths"]["input_weigth_path"]

        # create list
        image_list = []
        gt_list = []
        w_list = []
        name_list = []

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)
            item_gt_path    = os.path.join(gt_path, item)
            item_w_path     = os.path.join(weight_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1)
            image_w     = np.swapaxes(nib.load(item_w_path).dataobj, 0, 1)
            
            self.original_shape, self.original_type = image.shape, image.dtype
            
            image = image.astype(np.int64)
            image_gt = image_gt.astype(np.int64)
            image_w = image_w.astype(np.int64)

            if transform:
                image       = transform(image)
                image_gt    = transform(image_gt)
                image_w     = transform(image_w)

            if norm:
                image       = norm(image)

            for z in range(image.shape[-1]):
                # Pad
                padding_value = int(self.settings["preprocessing"]["padding"])
                image_gt_z = image_gt[:,:,z]
                image_w_z = image_w[:,:,z]
                image_z = image[:,:,z]
                image_z = np.pad(image_z, padding_value, "reflect")
            

                # Torchify
                image_z = torch.tensor(image_z).float()
                image_gt_z = torch.tensor(image_gt_z).float()
                image_w_z = torch.tensor(image_w_z).float()
                image_z = image_z.unsqueeze(0)
                image_gt_z = image_gt_z.unsqueeze(0)
                image_w_z = image_w_z.unsqueeze(0)

                image_list.append(image_z)
                gt_list.append(image_gt_z)
                w_list.append(image_w_z)
                name_list.append(f"{z}${item}")

        self.item_list = [image_list, gt_list, w_list, name_list]

    def original_information(self):
        return self.original_shape, self.original_type

    def __len__(self):
        return len(self.item_list[0])

    def __getitem__(self, idx):
        return {"volume"        :self.item_list[0][idx],\
                "segmentation"  :self.item_list[1][idx],\
                "weights"       :self.item_list[2][idx],\
                "name"          :self.item_list[3][idx]}
