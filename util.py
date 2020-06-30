import numpy as np
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

def split_list(input_list, split_rate):
    """Splits a list into n = 1 / split_rate pairs of two disjunct sublists
    Args:
        input_list (list):   List containing all elements
        split_rate (float):   Percentage of elements contained in the small list
    Returns:
        A list containing n tuples of lists
    """
    split_size = int(np.ceil(len(input_list) * split_rate))
    split_amount = int(len(input_list) / split_size)
    shuffle(input_list)
    result_list = []
    for iteration in range(split_amount):
        small_split = input_list[iteration*split_size:(iteration+1)*split_size]
        big_split = [i for i in input_list if not i in small_split]
        result_list.append((big_split, small_split))
    return result_list
