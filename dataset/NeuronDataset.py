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
class NeuronDataset(Dataset):
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


