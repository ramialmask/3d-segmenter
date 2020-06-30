import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
from utilities.util import find_divs, get_patch_data3d
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
class TrainingDataset(Dataset):
    #TODO Needs transformation, rotation, splits?
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        gt_path = settings["paths"]["input_gt_path"]

        # create list
        nii_list = []
        gt_list = []

        mb_size = int(self.settings["dataloader"]["block_size"])
        

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)
            item_gt_path    = os.path.join(gt_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1).astype(np.int64)
            image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1).astype(np.int64)

            if transform:
                image       = transform(image)
                image_gt    = transform(image_gt)

            if norm:
                image       = norm(image)
                # image_gt    = norm(image_gt)
            
            # Torchify
            image = torch.tensor(image).float()
            image_gt = torch.tensor(image_gt).float()

            if image.shape[0] > mb_size:
                vdivs = find_divs(self.settings, image)
                offset_value = int(self.settings["preprocessing"]["padding"])
                offset_volume = (offset_value, offset_value, offset_value)
                offset_segmentation = (0,0,0)

                image_list = [x for x in get_patch_data3d(image, divs=vdivs, offset=offset_volume).unsqueeze(1)]
                image_gt_list  = [x for x in get_patch_data3d(image_gt, divs=vdivs,offset=offset_segmentation).unsqueeze(1)]

                nii_list = nii_list + image_list
                gt_list = gt_list + image_gt_list
            else:
                nii_list.append(image.unsqueeze(1))
                gt_list.append(image_gt.unsqueeze(1))


        self.item_list = [nii_list, gt_list]

    def __len__(self):
        return len(self.item_list[0])

    def __getitem__(self, idx):
        return {"volume":self.item_list[0][idx], "segmentation":self.item_list[1][idx]}



