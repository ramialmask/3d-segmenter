import os
import cv2

import torch
import numpy as np
import nibabel as nib
import random

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

def difference_of_gaussians(img, kernel_size):
    img_1 = cv2.GaussianBlur(img, (1,1), 0)
    img_2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img_1 - img_2

class TrainingDataset2D(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        gt_path = settings["paths"]["input_gt_path"]

        # create list
        image_list = []
        gt_list = []
        name_list = []

        self.transform_p = transform

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)
            item_gt_path    = os.path.join(gt_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1)

            image_gt[image_gt > 0] = 1
            
            self.original_shape, self.original_type = image.shape, image.dtype
            
            image = image.astype(np.int64)
            image_gt = image_gt.astype(np.int64)

            if norm:
                image       = norm(image)

            for z in range(image.shape[-1]):
                # Pad
                padding_value = int(self.settings["preprocessing"]["padding"])
                padding_gt_value = int(self.settings["preprocessing"]["gt_padding"])

                image_gt_z = image_gt[:,:,z]
                if padding_gt_value > 0:
                    image_gt_z = np.pad(image_gt_z, padding_gt_value, "reflect")
                image_z = image[:,:,z]
                image_z = np.pad(image_z, padding_value, "reflect")
            

                # Torchify
                image_z = torch.tensor(image_z).float()
                image_gt_z = torch.tensor(image_gt_z).float()
                image_z = image_z.unsqueeze(0)
                image_gt_z = image_gt_z.unsqueeze(0)

                image_list.append(image_z)
                gt_list.append(image_gt_z)
                name_list.append(f"{z}${item}")

        self.item_list = [image_list, gt_list, name_list]

    def original_information(self):
        return self.original_shape, self.original_type

    def __len__(self):
        return len(self.item_list[0])

    def transform(self, volume, segmentation):
        # Volume and segmentation
        # Random horizontal flip
        if random.random() > 0.5:
            volume          = TF.functional.hflip(volume)
            segmentation    = TF.functional.hflip(segmentation)
        # Random vertical flip
        if random.random() > 0.5:
            volume          = TF.functional.vflip(volume)
            segmentation    = TF.functional.vflip(segmentation)

        # Only volume
        # Random gaussian blur
        # if random.random() > 0.9:
        #     volume = self.gaussian(volume)

        return volume, segmentation


    def __getitem__(self, idx):
        volume = self.item_list[0][idx]
        segmentation = self.item_list[1][idx]
        name = self.item_list[2][idx]

        if self.transform_p:
            volume, segmentation = self.transform(volume, segmentation)

        return {"volume":volume, "segmentation":segmentation, "name":name}
