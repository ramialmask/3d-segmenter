import os
import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset

# Load large chunk of data
# Make it tensor
# If bg: concatenate
# return _everything_ in __getitem__

class LargePredictionDataset(Dataset):
    def __init__(self, settings, transform=None, norm=None):
        self.settings = settings
        bg_channel = int(settings["dataloader"]["num_channels"]) == 2
        
        nii_path = settings["paths"]["input_raw_path"]
        self.item_list = []

        if bg_channel:
            bg_path = settings["paths"]["input_bg_path"]

        for item in os.listdir(nii_path):
            item_path = os.path.join(nii_path, item)
            print(f"Reading {item_path}")
            image = nib.load(item_path)
            image = np.array(image.get_fdata())
            image = image.squeeze()
            image = np.swapaxes(image, 0, 1)
            print(image.shape)
            if norm:
                image = norm(image)


            image = np.expand_dims(image, 0)
            if bg_channel:
                item_bg_path = os.path.join(bg_path, item)
                image_bg = nib.load(item_bg_path)
                image_bg = np.array(image_bg.get_fdata())
                image_bg = image_bg.squeeze()
                image_bg = np.swapaxes(image_bg, 0, 1)
                if norm:
                    image_bg = norm(image_bg)
                image_bg = np.expand_dims(image_bg, 0)
                image = np.concatenate((image, image_bg), 0)
            image = torch.tensor(image).float()
            self.item_list.append({"volume":image, "name":item})
            

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return self.item_list[idx]
