import os
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

#TODO Parametrize
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

#TODO 2CHannel / GT inclusion
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

#TODO New Trainingdataset which uses no hard coded bounds, only dict

class TrainingDataset2D(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Flag is set true if we use background channel, too
        bg_channel = int(settings["dataloader"]["num_channels"]) == 2

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        gt_path = settings["paths"]["input_gt_path"]
        if bg_channel:
            bg_path = settings["paths"]["input_bg_path"]

        # create list
        image_list = []
        gt_list = []
        name_list = []
        if bg_channel:
            bg_list = []

        self.transform_p = transform

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)
            item_gt_path    = os.path.join(gt_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1)

            if bg_channel:
                item_bg_path    = os.path.join(bg_path, item)
                image_bg        = np.swapaxes(nib.load(item_bg_path).dataobj, 0, 1)

            image_gt[image_gt > 0] = 1
            
            self.original_shape, self.original_type = image.shape, image.dtype
            
            image = image.astype(np.int64)
            image_gt = image_gt.astype(np.int64)
            if bg_channel:
                image_bg = image_bg.astype(np.int64)

            if norm:
                image       = norm(image)
                if bg_channel:
                    image_bg = norm(image_bg)

            for z in range(image.shape[-1]):
                # Pad
                padding_value = int(self.settings["preprocessing"]["padding"])
                padding_gt_value = int(self.settings["preprocessing"]["gt_padding"])

                image_gt_z = image_gt[:,:,z]
                if padding_gt_value > 0:
                    image_gt_z = np.pad(image_gt_z, padding_gt_value, "reflect")
                image_z = image[:,:,z]
                image_z = np.pad(image_z, padding_value, "reflect")
            
                image_z     = np.expand_dims(image_z, 0)
                image_gt_z  = np.expand_dims(image_gt_z, 0)

                if bg_channel:
                    image_z_bg = image_bg[:,:,z]
                    image_z_bg = np.pad(image_z_bg, padding_value, "reflect")
                    image_z_bg = np.expand_dims(image_z_bg, 0)
                    image_z = np.concatenate((image_z, image_z_bg), 0)

                # Torchify
                image_z = torch.tensor(image_z).float()
                image_gt_z = torch.tensor(image_gt_z).float()

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

