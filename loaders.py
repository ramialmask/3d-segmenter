import os
import json
import cv2
import numpy as np
import scipy.ndimage as nd
from dataset.dataset_2D import Dataset2D
from dataset.training_dataset import TrainingDataset
from dataset.training_dataset_2D import TrainingDataset2D
from dataset.training_dataset_2D_weighted import TrainingDataset2DWeighted
from dataset.prediction_dataset_2D import PredictionDataset2D
from dataset.large_prediction_dataset import LargePredictionDataset
from torch.utils.data import DataLoader
from functools import partial

def read_meta_dict(path, mode):
    """Load the meta dict / settings dict according to the mode
    """
    # Load path dict (always needed)
    settings = {}
    paths_path = os.path.join(path, "paths.json")
    with open(paths_path) as file:
        settings["paths"] = json.loads(file.read())
    
    if mode == "train":
        train_path = os.path.join(path, "train.json")
        with open(train_path) as file:
            settings["training"] = json.loads(file.read())
    elif mode == "predict":
        predict_path = os.path.join(path, "predict.json")
        with open(predict_path) as file:
            settings["prediction"] = json.loads(file.read())

    if mode == "count":
        partition_path = os.path.join(path, "partitioning.json")
        with open(partition_path) as file:
            settings["partitioning"] = json.loads(file.read())
    else:
        network_path = os.path.join(path, "network.json")
        with open(network_path) as file:
            _temp = json.loads(file.read())
            settings["computation"] = _temp["computation"]
            settings["preprocessing"] = _temp["preprocessing"]
            settings["dataloader"] = _temp["dataloader"]
            settings["network"] =  _temp["network"]
            settings["prediction"] = _temp["prediction"]
            settings["postprocessing"] =  _temp["postprocessing"]

    return settings

def write_meta_dict(path, settings, mode="train"):
    """Write down the meta dict / settings dict
    Contains deprecated code segments
    """
    path_dir = os.path.join(path, "paths.json")
    with open(path_dir, "w") as file:
        json.dump(settings["paths"], file, indent=4)

    if mode == "train":
        train_dir = os.path.join(path, "train.json")
        with open(train_dir, "w") as file:
            json.dump(settings["training"], file, indent=4)
    elif mode == "predict":
        predict_dir = os.path.join(path, "predict.json")
        with open(predict_dir, "w") as file:
            json.dump(settings["prediction"], file, indent=4)
    if mode == "count":
        partition_path = os.path.join(path, "partitioning.json")
        with open(partition_path, "w") as file:
            json.dump(settings["partitioning"], file, indent=4)
    else:
        network_path = os.path.join(path, "network.json")
        with open(network_path, "w") as file:
            _temp = {}
            _temp["computation"]    = settings["computation"]
            _temp["preprocessing"]  = settings["preprocessing"]
            _temp["dataloader"]     = settings["dataloader"]
            _temp["network"]        = settings["network"]
            _temp["prediction"]     = settings["prediction"]
            _temp["postprocessing"] = settings["postprocessing"]
            json.dump(_temp, file, indent=4)

def normalize(data):
    """Normalization
    """
    # data[data < 190] = 0
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # data = (data - 190 )/ (np.max(data) - np.min(data))
    return data

def normalize_global(data, settings):
    """Normalization based on global values
    """
    min_global = float(settings["preprocessing"]["normalization_values"]["foreground"][0])
    max_global = float(settings["preprocessing"]["normalization_values"]["foreground"][1])
    data = (data - min_global) / (max_global - min_global)
    return data

def normalize_histinfo(data, settings):
    min_global = float(settings["preprocessing"]["normalization_values"]["foreground"][0])
    max_global = float(settings["preprocessing"]["normalization_values"]["foreground"][1])
    cfreq = float(settings["preprocessing"]["normalization_values"]["cutoff"])

    bins = np.arange(min_global, max_global, 10)
    vals, bins = np.histogram(data, bins, density=True)
    acc = 0
    cutoff = np.amax(bins)
    cfreq *= sum(vals)
    for i, v in enumerate(vals):
        acc = acc + v
        if acc >= cfreq:
            cutoff = bins[i]
            break
    data[data > cutoff] = cutoff
    return data

def difference_of_gaussians(img, kernel_size):
    img_1 = cv2.GaussianBlur(img, (1,1), 0)
    img_2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return img_1 - img_2

def convoluted_background_substraction(image, method="gauss", size=5):
    res = -1
    if method == "gauss":
        filtered = nd.gaussian_filter(image, size)
    elif method == "median":
        filtered = nd.median_filter(image, size)
    res = image - filtered
    return res

def get_norm_func(settings):
    # return normalize
    # return partial(normalize_global, settings=settings)
    # return partial(normalize_histinfo, settings=settings)
    # fun = lambda x: convoluted_background_substraction(normalize(x), method="median")
    # return fun
    return normalize
    # return normalize_histinfo(data, settings=settings)

def get_loader(settings, input_list, train=True, testing=False):
    """Retrieve a dataloader for a given input list
    Args:
        settings (dict) : The meta dictionary
        input_list(list): List of items for the dataset
        train (bool)    : True if used for training, loads ground truth as well
        testing (bool)  : True if testing, False else
    """
    # assert((train and testing) or (train and not testing) or (not train and not testing))
    norm_func = get_norm_func(settings)
    shuffle = True
    if testing:
        batch_size = 1
        shuffle = False
    else:
        batch_size  = int(settings["dataloader"]["batch_size"])
    # dataset     = TrainingDataset2DWeighted(settings, input_list, norm=norm_func)
    # transform = train because there is no dedicated parameter for transformations (yet)
    #Not testing because in testing/inference we do not need transforms (yet)
    #Also really ugly 
    #TODO
    # dataset     = TrainingDataset2D(settings, input_list, norm=norm_func)
    print(f"BATCHSIZE {batch_size}")
    dataset     = TrainingDataset(settings, input_list, train=train, transform=not(testing), norm=norm_func)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset

def get_prediction_loader(settings):
    norm_func = get_norm_func(settings)
    input_list  = os.listdir(settings["paths"]["input_raw_path"])
    print(f"Creating dataset of size {len(input_list)}...")
    batch_size  = int(settings["dataloader"]["batch_size"])
    dataset     = LargePredictionDataset(settings, input_list, norm=norm_func)     
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("\nDone.")
    return loader, dataset
