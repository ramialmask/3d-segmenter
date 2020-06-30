import os
import shutil
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from models.deep_vessel_3d import Deep_Vessel_Net_FC
from dataset.NeuronDataset import NeuronDataset
from statistics import calc_statistics, calc_metrices_stats
from loss.dice_loss import DiceLoss
from loaders import *
from util import *

#TODO
# Learning rate sheduler
# CenterlineDiceLoss
# Remove deprecated parts

def crossvalidation(settings):
    """Splits the input list into a 1/test_split_rate splits containing test and train data
    Trains and validates 1/train_val_split_rate splits for each test split
    Uses the model with the lowest validation loss to test on the given split
    Saves all models and their respective dicts to the output path
    Saves training and testing dataframes to the output path
    Args:
        settings (dict): A settings dictionary
    """
    input_list = os.listdir(settings["paths"]["input_raw_path"])

    test_split_rate = float(settings["training"]["crossvalidation"]["test_split_rate"])
    train_val_split_rate = float(settings["training"]["crossvalidation"]["train_val_split_rate"])
    
    test_lists = split_list(input_list, test_split_rate)
    
    df = pd.DataFrame(columns=["Test Fold","Validation Fold", "Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "Validation Precision", "Validation Recall", "Validation Dice"])

    model_name = get_model_name(settings)

    for test_fold, test_train_list in enumerate(test_lists):
        test_list       = test_train_list[1]
        train_val_lists  = split_list(test_train_list[0], train_val_split_rate)
        settings["training"]["crossvalidation"]["test_set"] = test_list
        for train_val_fold, train_val_list in enumerate(train_val_lists):
            df = train(settings, train_val_list, test_fold, train_val_fold, model_name, df)

    
    test_df, test_patch_df = test_crossvalidation(settings, df, model_name)
    model_save_dir = os.path.join(settings["paths"]["output_model_path"], model_name)
    df.to_csv(f"{model_save_dir}/training.csv")
    test_df.to_csv(f"{model_save_dir}/test_scores.csv")
    test_df.to_csv(f"{model_save_dir}/test.csv")

def train(settings, train_val_list, test_fold, train_val_fold, model_name, df):
    """Trains a single model for a given test and train_val fold
    """
    learning_rate   = float(settings["training"]["optimizer"]["learning_rate"])
    net             = Deep_Vessel_Net_FC()
    criterion       = DiceLoss()#torch.nn.BCELoss()
    optimizer       = torch.optim.Adam(net.parameters(), lr=learning_rate)


    settings["training"]["crossvalidation"]["training_set"] = train_val_list[0]
    settings["training"]["crossvalidation"]["validation_set"] = train_val_list[1]

    train_loader    = get_loader(settings, train_val_list[0])
    val_loader      = get_loader(settings, train_val_list[1])
    
    epochs          = int(settings["training"]["epochs"])
    
    net.cuda()
    net.train()

    writer_path = settings["paths"]["writer_path"]
    writer = SummaryWriter(f"{writer_path}{model_name}/{test_fold}/{train_val_fold}")
    
    print("Test Fold\tVal Fold\tEpoch\tTraining Loss\tValidation Loss\tAccuracy\tPrecision\tRecall\tDice")
    last_model_dir = ""
    for epoch in range(epochs):
        net, optimizer, criterion, running_loss = train_epoch(net, optimizer, criterion, train_loader)
        validation_loss, accuracy, precision, recall, f1_dice = validate_epoch(net, criterion, val_loader)

        metrics = [accuracy, precision, recall, f1_dice]
        df = _write_progress(writer, test_fold, train_val_fold, epoch, epochs, running_loss, validation_loss, metrics, df)
    
        last_model_dir = save_epoch(settings, net, epoch, model_name, test_fold, train_val_fold, last_model_dir)
    return df

def train_epoch(net, optimizer, criterion, dataloader):
    """Trains a model for a single epoch
    """
    net.train()
    running_loss = 0.0
    d_len = len(dataloader)
    for i, item in enumerate(dataloader):
        optimizer.zero_grad()

        volume       = item["volume"]
        segmentation = item["segmentation"]
        volume       = volume.cuda()
        segmentation = segmentation.cuda()

        logits      = net(volume)
        loss        = criterion(logits, segmentation)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return net, optimizer, criterion, (running_loss / d_len)

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

def save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path):
    """Saves an epoch into a new path and deletes the model from the previous epoch
    """
    # If quicksaves should be deleted and there is a quicksave already, delete it
    if settings["training"]["delete_qs"] == "True" and last_model_path != "":
        shutil.rmtree(last_model_path)

    # Create the directory tree where the model and the meta information is saved
    model_save_dir = os.path.join(settings["paths"]["output_model_path"], model_name)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_dir = os.path.join(model_save_dir, str(test_fold))
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_dir = os.path.join(model_save_dir, str(val_fold))
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_path = os.path.join(model_save_dir, settings["paths"]["model_name"] + f"_{test_fold}_{val_fold}_{epoch}.dat")

    # Save the model and the meta information
    net.save_model(model_save_path)
    write_meta_dict(model_save_dir, settings, "train")
    return model_save_dir

def test_crossvalidation(settings, df, model_name):
    """Calculate the best epoch for each test fold and compute the score of the best model
    Test scores are saved in test.csv
    """
    test_columns=["Test Fold", "Validation Fold", "Test Loss", "Test Accuracy", "Test Precision", "Test Recall", "Test Dice"]
    test_df = pd.DataFrame(columns=test_columns)

    test_folds = range(0, int(1 / float(settings["training"]["crossvalidation"]["test_split_rate"])) - 1)
    val_folds  = range(0, int(1 / float(settings["training"]["crossvalidation"]["train_val_split_rate"])) - 1)

    epoch = int(settings["training"]["epochs"]) - 1

    model_path = settings["paths"]["output_model_path"] + model_name

    min_val_loss = 90001
    best_fold = -1

    test_patch_df = pd.DataFrame(columns=['patch','axis','class','predicted class','propability'])

    for test_fold in test_folds:
        # For each of the models get best validation loss
        for val_fold in val_folds:
            df_fold = df.loc[(df["Test Fold"] == test_fold) & (df["Validation Fold"] == val_fold) & (df["Epoch"] == epoch)]
            if df_fold["Validation Loss"][0] < min_val_loss:
                min_val_loss = df_fold["Validation Loss"][0]
                best_fold = df_fold

        best_val_fold           = best_fold["Validation Fold"][0]
        best_model_path         = os.path.join(model_path, str(test_fold), str(val_fold))
        best_model_data_path    = best_model_path + f"/_{test_fold}_{val_fold}_{epoch}.dat"
        
        # Once we have the best model path, we need to update the settings to get the correct test folds
        settings        = read_meta_dict(best_model_path, "train")
        best_model      = torch.load(best_model_data_path)
        best_model      = best_model.cuda()
        criterion       = torch.nn.BCELoss()

        # Test on the best candidate and save the settings
        test_list       = settings["training"]["crossvalidation"]["test_set"]
        test_loader     = get_loader(settings, test_list, True)

        test_patch_df, test_loss, accuracy, precision, recall, f1_dice =  test(best_model, criterion, test_loader, test_patch_df)
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
    return test_df, test_patch_df

def test(net, criterion, dataloader, df):
    """Tests a given network on provided test data
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

        logits       = net(volume)
        loss        = criterion(logits, segmentation)
        running_loss += loss.item()

        res             = logits.detach().cpu().numpy()
        res[res > 0.5] = 1.0 
        res[res <= 0.5] = 0.0

        target          = np.ravel(segmentation.cpu().numpy())
        stats_          = calc_statistics(res, target)

        df_item = pd.DataFrame({"dummy":0})

        df = df.append(df_item)
        result_list = [result_list[i] + stats_[i] for i in range(len(stats_))]
    precision, recall, vs, accuracy, f1_dice = calc_metrices_stats(result_list)
    return df, running_loss / d_len, accuracy, precision, recall, f1_dice

def _write_progress(writer, test_fold, val_fold, epoch, epochs, train_loss, eval_loss, metrics, df):
    """Writes the progress of the training both on the default output as well as the connected tensorboard writer
    """

    # Print the test progress to std.out
    print(f"{test_fold}\t\t{val_fold}\t\t{epoch}\t\t{train_loss}\t{eval_loss}\t{metrics[0]}\t{metrics[1]}\t{metrics[2]}\t{metrics[3]}")
    
    # Construct Dataframe for train.csv
    df_item = pd.DataFrame({"Test Fold":[test_fold],\
                            "Validation Fold":[val_fold],\
                            "Epoch":[epoch],\
                            "Train Loss": [train_loss],\
                            "Validation Loss":[eval_loss],\
                            "Validation Accuracy":[metrics[0]],\
                            "Validation Precision":[metrics[1]],\
                            "Validation Recall":[metrics[-2]],\
                            "Validation Dice":[metrics[-1]],\
                            })
    df = df.append(df_item)
    
    # Write the progress to the tensorboard
    writer.add_scalar(f"Loss/Training", train_loss, epoch)
    writer.add_scalar(f"Loss/Validation", eval_loss, epoch)
    writer.add_scalar(f"Validation Metrics/Precision", metrics[0], epoch)
    writer.add_scalar(f"Validation Metrics/Recall", metrics[1], epoch)
    writer.add_scalar(f"Validation Metrics/Accuracy", metrics[-2], epoch)
    writer.add_scalar(f"Validation Metrics/Dice", metrics[-1], epoch)
    return df

# Initialize cuda
torch.cuda.init()
torch.cuda.set_device(1)

# Read the meta dictionary
settings = read_meta_dict("/home/ramial-maskari/Documents/syndatron/3d-segmenter/","train")

# Run the program
crossvalidation(settings)
