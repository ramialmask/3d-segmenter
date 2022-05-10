import os
import shutil
import torch

import numpy as np
import pandas as pd

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from models.unet_3d_oliver import Unet3D
from models.deep_vessel_3d import Deep_Vessel_Net_FC
from statistics import calc_statistics, calc_metrices_stats
from loss.dice_loss import DiceLoss
from loss.cl_dice_loss import CenterlineDiceLoss, MixedDiceLoss, WBCECenterlineLoss
from loss.weighted_binary_cross_entropy_loss import WeightedBinaryCrossEntropyLoss
from loaders import *
from util import *
from blobanalysis import get_patch_overlap
from dataset.training_dataset import TrainingDataset
from classify_patches import classify_patch

#TODO
# CenterlineDiceLoss
# Remove deprecated parts
def _criterion():
    # criterion = CenterlineDiceLoss()#
    criterion = WeightedBinaryCrossEntropyLoss()#WBCECenterlineLoss(0.2)#
    return criterion

def _net():
    # Make it 2channel
    net = Unet3D(in_dim=2)
    return net

def _optimizer(settings, net):
    learning_rate   = float(settings["training"]["optimizer"]["learning_rate"])
    optimizer       = optim.Adam(net.parameters(), lr=learning_rate)
    return optimizer

def _scheduler(settings, optimizer):
    lr              = optimizer.state_dict()["param_groups"][0]["lr"]
    factor          = float(settings["training"]["scheduler"]["factor"])
    patience        = int(settings["training"]["scheduler"]["patience"])
    min_factor      = float(settings["training"]["scheduler"]["min_factor"])
    scheduler       = optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                        "min",factor=factor, \
                        patience=patience, \
                        threshold_mode="abs", \
                        min_lr=lr*min_factor, \
                        verbose=True)
    return scheduler

def crossvalidation(settings):
    """Splits the input list into a 1/test_split_rate splits containing test and train data
    Trains and validates 1/train_val_split_rate splits for each test split
    Uses the model with the lowest validation loss to test on the given split
    Saves all models and their respective dicts to the output path
    Saves training and testing dataframes to the output path
    Args:
        settings (dict): A settings dictionary
    """
    input_list = os.listdir(settings["paths"]["input_gt_path"])

    test_split_rate         = float(settings["training"]["crossvalidation"]["test_split_rate"])
    test_split_amount       = int(settings["training"]["crossvalidation"]["test_folds"])
    train_val_split_rate    = float(settings["training"]["crossvalidation"]["train_val_split_rate"])
    train_val_split_amount  = int(settings["training"]["crossvalidation"]["train_val_folds"])
    
    test_lists = split_list(input_list, test_split_rate, test_split_amount)
    
    df = pd.DataFrame(columns=["Test Fold","Validation Fold", "Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "Validation Precision", "Validation Recall", "Validation Dice"])

    df.set_index(["Test Fold","Validation Fold","Epoch"])
    model_name = get_model_name(settings)

    for test_fold, test_train_list in enumerate(test_lists):
        test_list       = test_train_list[1]
        print(f"Length Test List:\t{len(test_list)}\nFirst ten entries:\t{test_list[:10]}")
        train_val_lists  = split_list(test_train_list[0], train_val_split_rate, train_val_split_amount)
        settings["training"]["crossvalidation"]["test_set"] = test_list
        for train_val_fold, train_val_list in enumerate(train_val_lists):
            df = train(settings, train_val_list, test_fold, train_val_fold, model_name, df)

    model_save_dir = os.path.join(settings["paths"]["output_model_path"], model_name)
    df.to_csv(f"{model_save_dir}/training.csv")

    test_crossvalidation(settings, df, model_name, model_save_dir)

def train(settings, train_val_list, test_fold, train_val_fold, model_name, df):
    """Trains a single model for a given test and train_val fold
    """
    net             = _net()
    criterion       = _criterion()
    optimizer       = _optimizer(settings, net)
    scheduler       = _scheduler(settings, optimizer)


    settings["training"]["crossvalidation"]["training_set"] = train_val_list[0]
    settings["training"]["crossvalidation"]["validation_set"] = train_val_list[1]
    settings["training"]["loss"]["class"] = criterion.__class__.__name__
    settings["network"] = net.__class__.__name__

    print(f"Length Train List:\t{len(train_val_list[0])}\nFirst ten entries:\t{train_val_list[0][:10]}")
    print(f"Length Val List:\t{len(train_val_list[1])}\nFirst ten entries:\t{train_val_list[1][:10]}")

    train_loader, _    = get_loader(settings, train_val_list[0])
    val_loader, _      = get_loader(settings, train_val_list[1])
    
    epochs          = int(settings["training"]["epochs"])
    
    net.cuda()
    net.train()

    writer_path = settings["paths"]["writer_path"]
    writer = SummaryWriter(f"{writer_path}{model_name}/{test_fold}/{train_val_fold}")
    
    print("Test Fold\tVal Fold\tEpoch\tTraining Loss\tValidation Loss\t\tAccuracy\tPrecision\tRecall\tDice")
    last_model_dir = ""
    for epoch in range(epochs):
        net, optimizer, criterion, running_loss = train_epoch(net, optimizer, criterion, train_loader)
        validation_loss, accuracy, precision, recall, f1_dice = validate_epoch(net, criterion, val_loader)
        scheduler.step(validation_loss)

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
    #TODO Do transforms here
    for i, item in enumerate(dataloader):
        optimizer.zero_grad()

        volume       = item["volume"]
        centerline   = item["centerline"]
        segmentation = item["segmentation"]

        volume       = volume.cuda()
        centerline   = centerline.cuda()
        segmentation = segmentation.cuda()

        logits      = net(volume)
        # loss        = criterion(logits, segmentation, centerline)
        loss        = criterion(logits, segmentation )
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
        centerline   = item["centerline"]
        segmentation = item["segmentation"]

        volume       = volume.cuda()
        centerline   = centerline.cuda()
        segmentation = segmentation.cuda()

        logits      = net(volume)
        # loss        = criterion(logits, segmentation, centerline)
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

def test_crossvalidation(settings, df, model_name, model_save_dir):
    """Calculate the best epoch for each test fold and compute the score of the best model
    Test scores are saved in test.csv
    """
    test_columns=["Test Fold", "Validation Fold", "Test Loss", "Test Accuracy", "Test Precision", "Test Recall", "Test Dice"]
    test_df = pd.DataFrame(columns=test_columns)

    test_folds = range(int(settings["training"]["crossvalidation"]["test_folds"]))
    val_folds  = range(int(settings["training"]["crossvalidation"]["train_val_folds"]))

    epoch = int(settings["training"]["epochs"]) - 1

    model_path = settings["paths"]["output_model_path"] + model_name

    min_val_loss = 9000000001
    best_fold = -1

    test_patch_df = pd.DataFrame(columns=['patch','axis','class','predicted class','propability'])
    test_overlap_df = pd.DataFrame(columns=['Test Fold', 'Validation Fold', 'Patch','Classes','TP','FP','FN','F1 Score'])
    
    for test_fold in test_folds:
        # For each of the models get best validation loss
        for val_fold in val_folds:
            df_fold = df.loc[(df["Test Fold"] == test_fold) & (df["Validation Fold"] == val_fold) & (df["Epoch"] == epoch)]
            print(f"DF FOLD {test_fold} {val_fold} {df_fold}")
            if df_fold["Validation Loss"].iloc[0] < min_val_loss:
                min_val_loss = df_fold["Validation Loss"].iloc[0]
                best_fold = df_fold

        print(f"Best fold is {best_fold}")
        model_name              = settings["paths"]["model_name"]
        best_val_fold           = best_fold["Validation Fold"].iloc[0]
        best_model_path         = os.path.join(model_path, str(test_fold), str(val_fold))
        best_model_data_path    = best_model_path + f"/{model_name}_{test_fold}_{val_fold}_{epoch}.dat"
        
        # Once we have the best model path, we need to update the settings to get the correct test folds
        settings        = read_meta_dict(best_model_path, "train")
        best_model      = _net()
        best_model.load_model(best_model_data_path)
        best_model      = best_model.cuda()
        criterion       = _criterion()

        # Test on the best candidate and save the settings
        test_list       = settings["training"]["crossvalidation"]["test_set"]
        test_loader, test_dataset     = get_loader(settings, test_list, mode=1)

        test_loss, accuracy, precision, recall, f1_dice, reconstructed_patches =  test(best_model, criterion, test_loader, test_dataset)
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


        for item_name, reconstructed_prediction in reconstructed_patches:
            item_save_path = f"{model_save_dir}/{test_fold}/{item_name}"
            print(f"Writing {item_save_path}")
            write_nifti(item_save_path, reconstructed_prediction)

            for class_list in [[2],[1,2],[2,3],[1,2,3]]:
                # class_list = list(range(3-i,3))
                gt_path = settings["paths"]["input_gt_path"]
                raw_path = settings["paths"]["input_raw_path"]
                raw_patch = read_nifti(raw_path + item_name)
                target = read_nifti(gt_path + item_name)

                pred_classified = classify_patch(reconstructed_prediction, raw_patch, class_list)
                target_classified = classify_patch(target, raw_patch, class_list)

                tp,fp,fn = get_patch_overlap(pred_classified, target_classified)
                epsilon = 0.0000000000001
                dice = tp / (epsilon+ tp + 0.5*(fp + fn))

                test_overlap_item = pd.DataFrame({"Test Fold":[test_fold],\
                                    "Validation Fold":[best_val_fold],\
                                    "Patch":          [item_name],\
                                    "Classes":          [class_list],\
                                    "TP":           [tp],\
                                    "FP":         [fp],\
                                    "FN":         [fn],\
                                    "F1 Score":         [dice],\
                                    })
                test_overlap_df = test_overlap_df.append(test_overlap_item)


    test_df.to_csv(f"{model_save_dir}/test_scores.csv")
    test_patch_df.to_csv(f"{model_save_dir}/test.csv")
    test_overlap_df.to_csv(f"{model_path}/test_overlap.csv")

def test(net, criterion, dataloader, dataset):
    """Tests a given network on provided test data
    """
    net.eval()
    running_loss = 0.0
    d_len = len(dataloader)
    
    # Saving the TP, TN, FP, FN for all items to calculate stats
    result_list = [0, 0, 0, 0]

    item_dict = -1
    if not dataset.original_information()[2] == -1:
        # In order to save patches, the subvolumes need to be saved in 
        # a dict grouped by their name
        item_dict = {}
    else:
        reconstructed_patches = []

    for item in dataloader:
        volume       = item["volume"]
        centerline   = item["centerline"]
        segmentation = item["segmentation"]

        volume       = volume.cuda()
        centerline   = centerline.cuda()
        segmentation = segmentation.cuda()

        logits       = net(volume)
        # loss        = criterion(logits, segmentation, centerline)
        loss        = criterion(logits, segmentation )
        running_loss += loss.item()

        res             = logits.detach().cpu().numpy()
        res[res > 0.5] = 1.0 
        res[res <= 0.5] = 0.0

        item_name = item["name"][0]
        if not item_dict == -1:
            if item_name not in item_dict:
                item_dict[item_name] = []
            item_dict[item_name].append(res.astype(np.float32))
        else:
            res = res.squeeze().squeeze()
            reconstructed_patches.append([item_name, res.astype(dataset.original_information()[1])])

        res             = np.ravel(res)
        target          = np.ravel(segmentation.cpu().numpy())
        stats_          = calc_statistics(res, target)

        result_list = [result_list[i] + stats_[i] for i in range(len(stats_))]
    if not item_dict == -1:
        reconstructed_patches = reconstruct_patches(item_dict, dataset)
    precision, recall, vs, accuracy, f1_dice = calc_metrices_stats(result_list)
    return running_loss / d_len, accuracy, precision, recall, f1_dice, reconstructed_patches

def _write_progress(writer, test_fold, val_fold, epoch, epochs, train_loss, eval_loss, metrics, df):
    """Writes the progress of the training both on the default output as well as the connected tensorboard writer
    """

    # Print the test progress to std.out
    print(f"{test_fold}\t\t{val_fold}\t\t{epoch}\t{train_loss:.4f}\t\t{eval_loss:.4f}\t\t\t{metrics[0]:.4f}\t\t{metrics[1]:.4f}\t\t{metrics[2]:.4f}\t{metrics[3]:.4f}")
    
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
    df_item.set_index(["Test Fold","Validation Fold","Epoch"])
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
torch.cuda.set_device(0)

# Read the meta dictionary
settings = read_meta_dict("./","train")

# Run the program
crossvalidation(settings)
