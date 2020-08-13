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

class TrainingDataset2D(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        gt_path = settings["paths"]["input_gt_path"]

        # create list
        image_list = []
        gt_list = []

        # Load data
        for item in split:

            item_nii_path   = os.path.join(nii_path, item)
            item_gt_path    = os.path.join(gt_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1).astype(np.int64)

            image = image.astype(np.int64)

            if transform:
                image       = transform(image)
                image_gt    = transform(image_gt)

            if norm:
                image       = norm(image)

            for z in range(image.shape[-1]):
                # Pad
                padding_value = int(self.settings["preprocessing"]["padding"])
                image_z = image[:,:,z]
                image_z = np.pad(image_z, padding_value, "reflect")
                image_list.append(image_z)
                gt_list.append(image_gt[:,:,z])
            name_list.append(item)

        self.item_list = [image_list, gt_list, name_list]

    def original_information(self):
        return self.original_shape, self.original_type, self.vdivs

    def __len__(self):
        return len(self.item_list[0])

    def __getitem__(self, idx):
        return {"volume":self.item_list[0][idx], "segmentation":self.item_list[1][idx], "name":self.item_list[2][idx]}