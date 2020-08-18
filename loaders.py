import os
import json
from dataset.training_dataset import TrainingDataset
from dataset.training_dataset_2D import TrainingDataset2D
from torch.utils.data import DataLoader
import numpy as np

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
        json.dump(settings["paths"], file)

    if mode == "train":
        train_dir = os.path.join(path, "train.json")
        with open(train_dir, "w") as file:
            json.dump(settings["training"], file)
    elif mode == "predict":
        predict_dir = os.path.join(path, "predict.json")
        with open(predict_dir, "w") as file:
            json.dump(settings["prediction"], file)
    if mode == "count":
        partition_path = os.path.join(path, "partitioning.json")
        with open(partition_path, "w") as file:
            json.dump(settings["partitioning"], file)
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
            json.dump(_temp, file)

def normalize(data):
    """Normalization
    """
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

def get_loader(settings, input_list, testing=False):
    """Retrieve a dataloader for a given input list
    Args:
        settings (dict) : The meta dictionary
        input_list(list): List of items for the dataset
        testing (bool)  : True if testing, False else
    """
    shuffle = True
    if testing:
        batch_size = 1
        shuffle = False
    else:
        batch_size  = int(settings["dataloader"]["batch_size"])
    dataset     = TrainingDataset2D(settings, input_list, norm=normalize)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset
