import os
import numpy as np
import nibabel as nib
from random import shuffle
import datetime

def get_model_name(settings):
    model_name  = settings["paths"]["output_folder_prefix"] + " "
    model_name += settings["network"] + " " + settings["training"]["optimizer"]["class"]
    model_name += " factor " + settings["training"]["scheduler"]["factor"] + " "
    model_name += settings["training"]["loss"]["class"] + " LR=" + settings["training"]["optimizer"]["learning_rate"]
    model_name += " Blocksize " + settings["dataloader"]["block_size"] 
    model_name += " Epochs " + settings["training"]["epochs"] + " "+ " | " + str(datetime.datetime.now())

    return model_name

def split_list(input_list, split_rate, split_amount=-1):
    """Splits a list into n = 1 / split_rate pairs of two disjunct sublists
    Args:
        input_list (list):   List containing all elements
        split_rate (float):   Percentage of elements contained in the small list
    Returns:
        A list containing n tuples of lists
    """
    if split_amount == -1:
        split_amount = int(1/split_rate)
    split_size = int(np.ceil(len(input_list) * split_rate))
    shuffle(input_list)
    result_list = []
    for iteration in range(split_amount):
        small_split = input_list[iteration*split_size:(iteration+1)*split_size]
        big_split = [i for i in input_list if not i in small_split]
        result_list.append((big_split, small_split))
    return result_list

def read_nifti(path):
    """
    volume = read_nifti(path)

    Reads in the NiftiObject saved under path and returns a Numpy volume.
    """
    if(path.find(".nii")==-1):
        path = path + ".nii"
    NiftiObject = nib.load(path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    return volume

def write_nifti(path,volume):
    """
    write_nifti(path,volume)
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. Taken from Olivers filehandling class
    """
    if(path.find(".nii.gz")==-1):
        path = path + ".nii.gz"
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume, 0, 1), affine=affmat)
    nib.save(NiftiObject, os.path.normpath(path))

def reconstruct_patches(item_dict,dataset):
    original_shape, original_type, vdivs = dataset.original_information()
    result_list = []
    for item_name in item_dict.keys():
        print(f"Reconstructing {item_name}")
        # print(f"Item is {type(item_dict[item_name][0])} {len(item_dict[item_name][0])}")
        predictions = np.asarray([p.squeeze().squeeze() for p in item_dict[item_name]])
        reconstructed_prediction =  get_volume_from_patches3d(predictions, divs=vdivs, offset=(0,0,0))
        result_list.append((item_name, reconstructed_prediction))
    return result_list

def reconstruct_patches_2d(item_dict, dataset):
    original_shape, original_type = dataset.original_information()
    
    result_list = []
    for item_name in item_dict.keys():
        # print(f"Reconstructing {item_name}")
        
        pred = item_dict[item_name]
        sorted_pred = sorted(pred, key=lambda x: x[0])

        
        dx, dy = 0, 0
        if sorted_pred[0][1].shape[0] > original_shape[0]:
            dim_diff_x = sorted_pred[0][1].shape[0] - original_shape[0]
            dim_diff_y = sorted_pred[1][1].shape[1] - original_shape[1]
            dx = int(dim_diff_x/2)
            dy = int(dim_diff_y/2)
                
        if dx > 0:
            reconstructed_prediction = np.array([x[1][dx:-dx,dy:-dy] for x in sorted_pred]).astype(original_type)
        else:   
            reconstructed_prediction = np.array([x[1] for x in sorted_pred]).astype(original_type)
        reconstructed_prediction = np.swapaxes(reconstructed_prediction, 1, 2)
        reconstructed_prediction = np.transpose(reconstructed_prediction)

        # print(f"Reconstructed shape {reconstructed_prediction.shape}")
        result_list.append((item_name, reconstructed_prediction))
    return result_list

def get_volume_from_patches3d(patches4d, divs = (3,3,6), offset=(0,0,0)):
    """Reconstruct the minibatches, by Giles Tetteh
    Keep offset of (0,0,0) for fully padded volumes
    """
    if "torch" in str(type(patches4d)):
        patches4d = patches4d.numpy()
    new_shape = [(ps -of*2)*int(d) for ps, of, d in zip(patches4d.shape[-3:], offset, divs)]
    volume3d = np.zeros(new_shape, dtype=patches4d.dtype)
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    index = 0
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                patch = patches4d[index]
                index = index + 1
                volume3d[x:x+widths[0],y:y+widths[1],z:z+widths[2]] = \
                        patch[offset[0]:offset[0] + widths[0], offset[1]:offset[1]+widths[1], offset[2]:offset[2]+widths[2]]
    return volume3d

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

def stitch_volume(item_list, original_shape):
    item_shape = item_list[0].shape
    width = int(original_shape[0] / item_shape[0])
    counter = 0
    result_item = np.zeros(original_shape)
    for x in range(0, width):
        for y in range(0, width):
            for z in range(0, width):
                result_item[x*item_shape[0]:x*item_shape[0]+item_shape[0],\
                        y*item_shape[0]:y*item_shape[0]+item_shape[0],\
                        z*item_shape[0]:z*item_shape[0] +item_shape[0]] = item_list[counter]
                counter+=1                  
    return result_item
