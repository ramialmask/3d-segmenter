import os

import torch
import numpy as np
import nibabel as nib
import random

import torchvision.transforms as TF
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

def find_divs(settings, volume):
    """Find divs for @get_volume_from_patches3d according to the blocksize
    """
    shape = volume.shape
    mb_size = int(settings["dataloader"]["block_size"])
    return tuple(s / m for s,m in zip(shape,(mb_size, mb_size, mb_size)))

def get_patch_data3d(volume3d, divs=(3,3,6), offset=(6,6,6), seg=False):
    """Generate minibatches, by Giles Tetteh
    Args:
        - volume3d (np.array)       :   The volume to cut
        - divs (tuple, optional)    :   Amount to divide each side
        - offset (tuple, optional)  :   Offset for each div
    """
    if "torch" in str(type(volume3d)):
        volume3d = volume3d.numpy()
    patches = []
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    #print("V3dshape {}".format(volume3d.shape))
    patch_mean = np.mean(volume3d)
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                patch = np.ones(patch_shape, dtype=volume3d.dtype) * patch_mean
                x_s = x - offset[0] if x - offset[0] >= 0 else 0
                x_e = x + widths[0] + offset[0] if x + \
                        widths[0] + offset[0] <= shape[0] else shape[0]
                y_s = y - offset[1] if y - offset[1] >= 0 else 0
                y_e = y + widths[1] + offset[1] if y + \
                        widths[1] + offset[1] <= shape[1] else shape[1]
                z_s = z - offset[2] if z - offset[2] >= 0 else 0
                z_e = z + widths[2] + offset[2] if z + \
                        widths[2] + offset[2] <= shape[2] else shape[2]

                vp = volume3d[x_s:x_e,y_s:y_e,z_s:z_e]
                px_s = offset[0] - (x - x_s)
                px_e = px_s + (x_e - x_s)
                py_s = offset[1] - (y - y_s)
                py_e = py_s + (y_e - y_s)
                pz_s = offset[2] - (z - z_s)
                pz_e = pz_s + (z_e - z_s)
                patch[px_s:px_e, py_s:py_e, pz_s:pz_e] = vp
                patches.append(patch)

    return torch.tensor(np.array(patches, dtype = volume3d.dtype))

class TrainingDataset(Dataset):
    #TODO Background!
    def __init__(self, settings, split, transform=None, norm=None, train=False):
        self.settings = settings

        self.train = train

        # Flag is set true if we use background channel, too
        bg_channel = int(settings["dataloader"]["num_channels"]) == 2
        
        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        if train:
            gt_path = settings["paths"]["input_gt_path"]
        if bg_channel:
            bg_path = settings["paths"]["input_bg_path"]

        # create list
        nii_list = []
        if train:
            gt_list = []
        if bg_channel:
            bg_list = []

        mb_size = int(self.settings["dataloader"]["block_size"])
        self.vdivs = -1
        name_list = []
        

        self.transform_p = transform
        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)
            if train:
                item_gt_path    = os.path.join(gt_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            image       = np.squeeze(image)
            if train:
                image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1).astype(np.int64)
                image_gt[image_gt > 0] = 1
                image_gt    = np.squeeze(image_gt)

            if bg_channel:
                item_bg_path    = os.path.join(bg_path, item)
                image_bg        = np.swapaxes(nib.load(item_bg_path).dataobj, 0, 1)

            self.original_shape = image.shape
            self.original_type = image.dtype
            image = image.astype(np.int64)
            if bg_channel:
                image_bg = image_bg.astype(np.int64)

            if norm:
                image       = norm(image)
                # image_gt    = norm(image_gt)
                if bg_channel:
                    image_bg = norm(image_bg)
            

            if image.shape[0] > mb_size:
                # Torchify
                #TODO torchify downstairs
                # image = torch.tensor(image).float()
                if train:
                    image_gt = torch.tensor(image_gt).float()

                vdivs = find_divs(self.settings, image)
                self.vdivs = vdivs
                offset_value = int(self.settings["preprocessing"]["padding"])
                offset_volume = (offset_value, offset_value, offset_value)
                offset_seg_value = 0
                if int(self.settings["preprocessing"]["gt_padding"]) > 0:
                    offset_seg_value = int(self.settings["preprocessing"]["gt_padding"])
                offset_segmentation = (offset_seg_value,offset_seg_value,offset_seg_value)

                image_list = [np.expand_dims(x, 0) for x in get_patch_data3d(image, divs=vdivs, offset=offset_volume)]
                if train:
                    image_gt_list  = [x for x in get_patch_data3d(image_gt, divs=vdivs,offset=offset_segmentation).unsqueeze(1)]

                if bg_channel:
                    image_bg_list = [np.expand_dims(x, 0) for x in get_patch_data3d(image_bg, divs=vdivs, offset=offset_volume)]
                    image_list = [torch.tensor(np.concatenate(a, 0)).float() for a in list(zip(image_list, image_bg_list))]
                else:
                    image_list = [torch.tensor(img).float() for img in image_list]


                nii_list = nii_list + image_list
                if train:
                    gt_list = gt_list + image_gt_list
                name_list = name_list + [item for i in range(len(image_list))]
            else:
                # Pad
                padding_value = int(self.settings["preprocessing"]["padding"])
                if train:
                    gt_padding_value = int(self.settings["preprocessing"]["gt_padding"])
                    if gt_padding_value > 0:
                        image_gt = np.pad(image_gt, gt_padding_value, "reflect")

                image = np.pad(image, padding_value, "reflect")
                image = np.expand_dims(image, 0)
                if bg_channel:
                    image_bg = np.pad(image_bg, padding_value, "reflect")
                    image_bg = np.expand_dims(image_bg, 0)
                    image = np.concatenate((image, image_bg), 0)


                # Torchify
                image = torch.tensor(image).float()
                nii_list.append(image)
                if train:
                    image_gt = torch.tensor(image_gt).float()
                    gt_list.append(image_gt.unsqueeze(0))
                name_list.append(item)

        if train:
            self.item_list = [nii_list, gt_list, name_list]
        else:
            self.item_list = [nii_list, name_list]

    def original_information(self):
        return self.original_shape, self.original_type, self.vdivs

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

        if self.train:
            name = self.item_list[2][idx]
            segmentation = self.item_list[1][idx]

            if self.transform_p:
                volume, segmentation = self.transform(volume, segmentation)

            # from matplotlib import pyplot as plt
            # plt.figure()
            # fig, axes = plt.subplots(ncols = 2)
            # vol          = volume.numpy()
            # seg    = segmentation.numpy()
            # axes[0].imshow(np.max(vol[0,:,:,:], axis=2))
            # axes[1].imshow(np.max(seg[0,:,:,:], axis=2))
            # plt.show()
            # exit()
            return {"volume":volume, "segmentation":segmentation, "name":name}
        else:
            name = self.item_list[1][idx]
            return {"volume":volume, "name":name}




