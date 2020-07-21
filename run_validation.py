import os
import torch
import numpy as np
import os
import shutil
import torch

import numpy as np
import pandas as pd

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from models.deep_vessel_3d import Deep_Vessel_Net_FC
from statistics import calc_statistics, calc_metrices_stats
from loss.dice_loss import DiceLoss
from loss.weighted_binary_cross_entropy_loss import WeightedBinaryCrossEntropyLoss
from loaders import *
from util import *
from dataset.training_dataset import TrainingDataset

def run_validation(p_, model_name):
    """Splits the input list into a 1/test_split_rate splits containing test and train data
    Trains and validates 1/train_val_split_rate splits for each test split
    Uses the model with the lowest validation loss to test on the given split
    Saves all models and their respective dicts to the output path
    Saves training and testing dataframes to the output path
    Args:
        settings (dict): A settings dictionary
    """
    
    
    df = pd.DataFrame(columns=["Test Fold","Validation Fold", "Epoch", "Validation Loss", "Validation Accuracy", "Validation Precision", "Validation Recall", "Validation Dice"])

    ip_= os.path.join(p_, model_name,"0","0")
    # Read the meta dict of the model
    settings = read_meta_dict(ip_, "train")

    # for test_fold in range(2):
    #     # Load settings
    #     test_list       = settings["training"]["crossvalidation"]["test_set"]
    #     settings["training"]["crossvalidation"]["test_set"] = test_list
    #     for train_val_fold in range(2):
    #         ip_= os.path.join(p_, model_name,f"{test_fold}/{train_val_fold}")
    #         settings = read_meta_dict(ip_, "train")
    #         df = train(settings, test_fold, train_val_fold, model_name, df)

    model_save_dir = os.path.join(settings["paths"]["output_model_path"], model_name)
    # df.to_csv(f"{model_save_dir}/training.csv")
    df = pd.read_csv(f"{model_save_dir}/training.csv")
    test_crossvalidation(settings, df, model_name, model_save_dir)

def train(settings, test_fold, train_val_fold, model_name, df):
    """Trains a single model for a given test and train_val fold
    """
    learning_rate   = float(settings["training"]["optimizer"]["learning_rate"])
    net             = Deep_Vessel_Net_FC()
    criterion       = WeightedBinaryCrossEntropyLoss(class_frequency=True)#DiceLoss()#torch.nn.BCELoss()

    p_ = "/home/ramial-maskari/Documents/syndatron/segmentation/output/models/"
    net_path = os.path.join(p_, model_name, f"{test_fold}/{train_val_fold}/_{test_fold}_{train_val_fold}_49.dat")
    net.load_model(net_path)


    train_val_list = []
    train_list = settings["training"]["crossvalidation"]["training_set"] 
    val_list=settings["training"]["crossvalidation"]["validation_set"] 

    train_loader    = get_loader(settings, train_list)
    val_loader      = get_loader(settings, val_list)
    
    epoch          = int(settings["training"]["epochs"])
    
    net.cuda()
    net.eval()
    
    print("Test Fold\tVal Fold\tEpoch\tTraining Loss\tValidation Loss\tAccuracy\tPrecision\tRecall\tDice")
    validation_loss, accuracy, precision, recall, f1_dice = validate_epoch(net, criterion, val_loader)

    metrics = [accuracy, precision, recall, f1_dice]
    df = _write_progress(test_fold, train_val_fold, epoch, validation_loss, metrics, df)

    return df

def validate_epoch(net, criterion, dataloader):
    """Evaluates a single epoch
    """
    net.eval()
    running_loss = 0.0
    d_len = len(dataloader)
    result_list = [0, 0, 0, 0]
    for item in dataloader:
        volume       = item["volume"]
        segmentation = item["segmentation"]
        volume       = volume.cuda()
        segmentation = segmentation.cuda()

        logits      = net(volume)
        loss        = criterion(logits, segmentation)
        running_loss += loss.item()

        res             = logits.detach().cpu().numpy()
        res[res > 0.5] = 1.0 
        res[res <= 0.5] = 0.0
        res = np.ravel(res)

        target          = np.ravel(segmentation.cpu().numpy())
        stats_          = calc_statistics(res, target)

        result_list = [result_list[i] + stats_[i] for i in range(len(stats_))]
    precision, recall, vs, accuracy, f1_dice = calc_metrices_stats(result_list)
    return running_loss / d_len, accuracy, precision, recall, f1_dice

def test_crossvalidation(settings, df, model_name, model_path):
    """Calculate the best epoch for each test fold and compute the score of the best model
    Test scores are saved in test.csv
    """
    test_columns=["Test Fold", "Validation Fold", "Test Loss", "Test Accuracy", "Test Precision", "Test Recall", "Test Dice"]
    test_df = pd.DataFrame(columns=test_columns)

    test_folds = range(0, int(1 / float(settings["training"]["crossvalidation"]["test_split_rate"])))
    val_folds  = range(0, int(1 / float(settings["training"]["crossvalidation"]["train_val_split_rate"])) - 1)

    epoch = int(settings["training"]["epochs"])

    model_path = settings["paths"]["output_model_path"] + model_name

    min_val_loss = 90001
    best_fold = -1

    test_patch_df = pd.DataFrame(columns=['patch','axis','class','predicted class','propability'])

    print(f"DF \n{df}")
    for test_fold in test_folds:
        # For each of the models get best validation loss
        for val_fold in val_folds:
            
            df_fold = df.loc[(df["Test Fold"] == test_fold) & (df["Validation Fold"] == val_fold) & (df["Epoch"] == epoch)]
            print(f"DF FOLD{test_fold} {val_fold} {epoch}\n{df_fold}")
            if df_fold["Validation Loss"].iloc[0] < min_val_loss:
                min_val_loss = df_fold["Validation Loss"].iloc[0]
                best_fold = df_fold

        best_val_fold           = best_fold["Validation Fold"].iloc[0]
        best_model_path         = os.path.join(model_path, str(test_fold), str(val_fold))
        best_model_data_path    = best_model_path + f"/_{test_fold}_{val_fold}_{epoch-1}.dat"
        
        # Once we have the best model path, we need to update the settings to get the correct test folds
        settings        = read_meta_dict(best_model_path, "train")

        settings["paths"]["input_raw_path"] = "/home/ramial-maskari/Documents/syndatron/segmentation/input/raw/"
        settings["paths"]["input_gt_path"] = "/home/ramial-maskari/Documents/syndatron/segmentation/input/gt/"

        best_model      = Deep_Vessel_Net_FC()
        best_model.load_model(best_model_data_path)
        best_model      = best_model.cuda()
        criterion       = WeightedBinaryCrossEntropyLoss(class_frequency=True)

        # Test on the best candidate and save the settings
        test_list       = settings["training"]["crossvalidation"]["test_set"]
        test_loader, test_dataset     = get_loader(settings, test_list, True)

        test_patch_df, test_loss, accuracy, precision, recall, f1_dice, reconstructed_patches =  test(best_model, criterion, test_loader, test_dataset)

        for item_name, reconstructed_prediction in reconstructed_patches:
            item_save_path = f"{model_path}/{test_fold}/{item_name}"
            print(f"Writing {item_save_path}")
            write_nifti(item_save_path, reconstructed_prediction)

        print(f"Test scores")
        print("Test Fold\tValidation Fold\tTest Loss\tAccuracy\tPrecision\tRecall\tDice")
        print(f"{test_fold}\t{best_val_fold}\t{test_loss}\t{accuracy}\t{precision}\t{recall}\t{f1_dice}")
        test_item = pd.DataFrame({"Test Fold":[test_fold],\
                            "Validation Fold":[best_val_fold],\
                            "Test Loss":     test_loss, \
                            "Test Accuracy": [accuracy],\
                            "Test Precision":[precision],\
                            "Test Recall":   [recall],\
                            "Test Dice":     [f1_dice],\
                            })
        test_df = test_df.append(test_item)
    test_df.to_csv(f"{model_path}/test_scores.csv")
    test_patch_df.to_csv(f"{model_path}/test.csv")

def test(net, criterion, dataloader, dataset):
    """Tests a given network on provided test data
    """
    net.eval()
    running_loss = 0.0
    d_len = len(dataloader)

    # Saving the TP, TN, FP, FN for all items to calculate stats
    result_list = [0, 0, 0, 0]

    # In order to save patches, the subvolumes need to be saved in 
    # a dict grouped by their name
    item_dict = {}
    for item in dataloader:
        volume       = item["volume"]
        segmentation = item["segmentation"]
        volume       = volume.cuda()
        segmentation = segmentation.cuda()

        logits       = net(volume)
        loss        = criterion(logits, segmentation)
        running_loss += loss.item()

        res             = logits.detach().cpu().numpy()
        res[res > 0.5] = 1.0 
        res[res <= 0.5] = 0.0

        item_name = item["name"][0]
        if item_name not in item_dict:
            item_dict[item_name] = []
        item_dict[item_name].append(res.astype(np.float32))

        res             = np.ravel(res)
        target          = np.ravel(segmentation.cpu().numpy())
        stats_          = calc_statistics(res, target)

        result_list = [result_list[i] + stats_[i] for i in range(len(stats_))]
    reconstructed_patches = reconstruct_patches(item_dict, dataset)
    precision, recall, vs, accuracy, f1_dice = calc_metrices_stats(result_list)
    return running_loss / d_len, accuracy, precision, recall, f1_dice, reconstructed_patches

def _write_progress(test_fold, val_fold, epoch, eval_loss, metrics, df):
    """Writes the progress of the training both on the default output as well as the connected tensorboard writer
    """

    # Construct Dataframe for train.csv
    df_item = pd.DataFrame({"Test Fold":[test_fold],\
                            "Validation Fold":[val_fold],\
                            "Epoch":[epoch],\
                            "Validation Loss":[eval_loss],\
                            "Validation Accuracy":[metrics[0]],\
                            "Validation Precision":[metrics[1]],\
                            "Validation Recall":[metrics[-2]],\
                            "Validation Dice":[metrics[-1]],\
                            })
    df = df.append(df_item)
    print(df)
    return df

# Initialize cuda
torch.cuda.init()
torch.cuda.set_device(1)

p_ = "/home/ramial-maskari/Documents/syndatron/segmentation/output/models/"
model_name = "DeepVessel DiceLoss 2020 06 30 leanclassification2d Adam factor 0.5 BCELoss LR=5e-4 Blocksize 100 Epochs 50  | 2020-06-30 23:42:57.095610"

# Run the program
run_validation(p_, model_name)