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

# Cuts a volume into smaller cubes
def cut_volume(item, name, new_shape):
    """
    Cuts a volume into smaller cubes
    Args:
        - item (np.array)   : image array
        - name (string)     : name of the volume
        - new_shape(tuple)  : new shape in the form of (int, int, int)
    """
    shape = item.shape
    width = int(shape[0] / new_shape)
    counter = 0
    result_list = []
    for x in range(0, width):
        for y in range(0, width):
            for z in range(0, width):
                sub_item = item[x*new_shape:x*new_shape+new_shape,y*new_shape:y*new_shape+new_shape, z*new_shape:z*new_shape +new_shape]
                out_name = name.replace(".nii.gz",f"_{counter}.nii.gz")
                result_list.append((out_name, sub_item))
                counter+=1                  
    return result_list

def prepare_lists(settings, item, image, image_list, name_list, image_gt=None, gt_list=None, image_bg=None):
    """
    Cut down an image into single z-layers, padd them and arrange them into a list 
    which can be reconstructed by the corresponding name list
    Args:
        - settings (dict)   : settings dictionary
        - item (string)     : name of the volume
        - image (np.array)  : image array
        - image_list (list) : list containing all already prepared images or empty list
        - name_list (list)  : list of the corresponding names
        - norm (function)   : normalization function
        - image_gt(np.array): Ground Truth image array
        - gt_list (list)    : list containing all already prepared gt images or empty list
        - image_bg(np.array): backtround image array
    """

    train = not(image_gt is None)
    background = not(image_bg is None)

    # Make sure that there is never a gt image without a coresponding list and vice versa
    assert(((image_gt is None) and (gt_list is None)) or (not(image_gt is None) and not(gt_list is None)))

    

    for z in range(image.shape[-1]):
        # Pad
        padding_value = int(settings["preprocessing"]["padding"])
        image_z = image[:,:,z]
        image_z = np.pad(image_z, padding_value, "reflect")

        if train:
            padding_gt_value = int(settings["preprocessing"]["gt_padding"])
            image_gt_z = image_gt[:,:,z]
            image_gt_z = np.pad(image_gt_z, padding_gt_value, "reflect")

        if background:
            image_bg_z = image_bg[:,:,z]
            image_bg_z = np.pad(image_bg_z, padding_value, "reflect")

        # Torchify
        image_z = np.expand_dims(image_z,0)

        if train:
            image_gt_z  = np.expand_dims(image_gt_z,0)
            image_gt_z = image_gt_z.astype(np.int32)
            image_gt_z  = torch.tensor(image_gt_z).float()
            gt_list.append(image_gt_z)

        if background:
            image_bg_z  = np.expand_dims(image_bg_z,0)
            image_z     = np.concatenate((image_z, image_bg_z), 0)

        image_z = torch.tensor(image_z).float()

        #XXX
        # if image_z.shape[-1] > 104 or image_bg_z.shape[-1] > 104 or image_gt_z.shape[-1] > 100:
        #     print(f"{item} {image_z.shape} {image_bg.shape} {image_gt_z.shape}")

        image_list.append(image_z)
        name_list.append(f"{z}${item}")

class Dataset2DSlow(Dataset):
    def __init__(self, settings, splot, transform=False, norm=None):
        pass
    #TODO Ganze Datenladenlogik in getget_item ianstelle des Konstruktors
    #TODO Buffer für Listen

class Dataset2D(Dataset):
    def __init__(self, settings, split, train=True, transform=False, norm=None):
        self.settings   = settings
        self.train      = train
        self.transform_p= transform # transform parameter 

        assert((train and transform) or (train and not transform) or (not train and not transform))

        # Flag is set true if we use background channel, too
        self.bg_channel = int(settings["dataloader"]["num_channels"]) == 2

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]
        if self.train:
            gt_path = settings["paths"]["input_gt_path"]
        if self.bg_channel:
            bg_path = settings["paths"]["input_bg_path"]

        # create list
        image_list = []
        name_list = []
        if self.train:
            gt_list = []
        if self.bg_channel:
            bg_list = []

        # Data size which can be fed to the network
        self.mb_size = int(settings["dataloader"]["block_size"])

        # Load data
        for item_no, item in enumerate(split):

            # Load nifti files
            item_nii_path   = os.path.join(nii_path, item)
            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1)
            
            if self.train:
                item_gt_path    = os.path.join(gt_path, item)
                image_gt    = np.swapaxes(nib.load(item_gt_path).dataobj, 0, 1)
                # Instance labels to binary labels
                image_gt[image_gt > 0] = 1


            if self.bg_channel:
                item_bg_path    = os.path.join(bg_path, item)
                image_bg        = np.swapaxes(nib.load(item_bg_path).dataobj, 0, 1)

            
            # save for reconstruction
            self.original_shape, self.original_type = image.shape, image.dtype
            
            if norm:
                image       = norm(image)
                if self.bg_channel:
                    image_bg = norm(image_bg)
            
            # Check if our input image is too big
            if image.shape[0] > self.mb_size:
                # Cut the individual images
                cut_images  = cut_volume(image, item, self.mb_size)
                if self.train:
                    cut_gt      = cut_volume(image_gt, item, self.mb_size)
                if self.bg_channel:
                    cut_bg      = cut_volume(image_bg, item, self.mb_size)

                # Prepare a list for each "sub_image"
                for i, (cut_name, cut_image) in enumerate(cut_images):
                    # Very awkward formulation to have a nice function call, will probably lead to bugs
                    #TODO put cut name into name list
                    if self.train:
                        _, cut_gt_image = cut_gt[i]
                    else:
                        cut_gt_image    = None
                        gt_list         = None
                    if self.bg_channel:
                        _, cut_bg_image = cut_bg[i]
                    else:
                        cut_bg_image = None

                    prepare_lists(settings,\
                            cut_name,\
                            cut_image,\
                            image_list,\
                            name_list,\
                            cut_gt_image,\
                            gt_list,\
                            cut_bg_image)
            else:
                if not self.train:
                    image_gt    = None
                    gt_list     = None
                if not self.bg_channel:
                    image_bg    = None
                prepare_lists(settings,\
                        item,\
                        image,\
                        image_list,\
                        name_list,\
                        image_gt,\
                        gt_list,\
                        image_bg)
        if self.train:
            self.item_list  = [image_list, gt_list, name_list]
        else:
            self.item_list  = [image_list, name_list]

    #TODO self.mb_size not needed?
    def original_information(self):
        return self.original_shape, self.original_type, self.mb_size

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
            segmentation = self.item_list[1][idx]
            name = self.item_list[2][idx]
        else:
            name = self.item_list[1][idx]

        if self.transform_p:
            volume, segmentation = self.transform(volume, segmentation)

        if self.train:
            return {"volume":volume, "segmentation":segmentation, "name":name}
        else:
            return {"volume":volume, "name":name}
