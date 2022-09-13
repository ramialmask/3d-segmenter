import torch
import cv2
import datetime

from sliding_window_inferrer import SlidingWindowInferer
from loaders import read_meta_dict
from util import *
from monai.networks.nets import UNet as unet
from collections import deque


def load_inferer(settings, size):
    """Returns the network and corresponding inferer
    """
    in_channels = int(settings["dataloader"]["num_channels"])
    net = unet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
            channels=(4,8,16),
            strides=(2,2,2),
            num_res_units=4,
            act="mish"
    )
    model_path = settings["paths"]["input_model_path"] + settings["paths"]["input_model"]
    t_ = torch.load(model_path)
    net.load_state_dict(t_)
    net = net.cuda()


    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
    inferer = SlidingWindowInferer(
                    roi_size=size, 
                    overlap=0.7,
                    mode="gaussian",
                    padding_mode="replicate",
                    device=torch.device("cpu"),
                    progress=True
                )

    net.eval()
    return net, inferer

def norm(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

def load_nifti(settings, item):
    """
    Load Niftis, concatenate foreground and background if neccessary
    """
    nii_path = settings["paths"]["input_raw_path"]
    bg_path = settings["paths"]["input_bg_path"]
    # Load FG
    item_path = os.path.join(nii_path, item)
    print(f"loading {item_path}")
    image = nib.load(item_path)
    image = np.asanyarray(image.dataobj)
    image = image.astype(np.float32)
    image = image.squeeze()
    image = np.swapaxes(image, 0, 1)
    print(f"Norming image {np.amin(image)} {np.amax(image)} {image.dtype}")
    image = norm(image)
    print(f"After norming image {np.amin(image)} {np.amax(image)} {image.dtype}")

    # Load BG
    if int(settings["dataloader"]["num_channels"]) == 2:
        item_bg_path = os.path.join(bg_path, item)
        print(f"loading {item_bg_path}")
        image_bg = nib.load(item_bg_path)
        image_bg = np.asanyarray(image_bg.dataobj)
        image_bg = image_bg.astype(np.float32)
        image_bg = image_bg.squeeze()
        image_bg = np.swapaxes(image_bg, 0, 1)
        image_bg = norm(image_bg)

        print("Concatinating")
        img = deque()#
        img.append(image)
        img.append(image_bg)
        img = np.array(img)
        image_bg = 0
        img = np.expand_dims(img, 0)
    else:
        img = image
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, 0)
    image = 0
    print(img.shape)
    print(f"{item} {type(img)} {img.shape}")
    img = img.astype(np.float16)
    return img

def load_tiff(settings, items, size):
    """
    Load TIFFs, concatenate foreground and background if neccessary
    """
    print("Loading TIFFs")
    tiff_path = settings["paths"]["input_raw_path"]
    bg_path = settings["paths"]["input_bg_path"]
    filename = os.listdir(settings["paths"]["input_raw_path"])[0]
    print(filename)
    #TODO FIX FILENAME FINDING
    # filename = filename.replace(filename.split("_")[1], "").replace(".tif","")
    # print(filename)
    tiff_patch = deque() 
    start = datetime.datetime.now()
    for item_index in range(items, items + size):
        # item = f"{filename}{file}.tif"
        # item = filename.replace(filename.split("_")[5], f"Z{str(file).zfill(3)}")
        # print(item)
        item = f"Z_{str(item_index + 1).zfill(5)}.tif"
        img_path = os.path.join(tiff_path, item)
        if os.path.exists(img_path):
            image = cv2.imread(img_path, -1)
            image = image.astype(np.float32)
            image = norm(image)
            if int(settings["dataloader"]["num_channels"]) == 2:
                img_bg_path = os.path.join(bg_path, item)
                image_bg = cv2.imread(img_bg_path, -1)
                image_bg = image.astype(np.float32)
                image_bg = norm(image_bg)
                img = deque()
                img.append(image)
                img.append(image_bg)
                image_bg = 0
                img = np.expand_dims(img, 0)
            else:
                img = image
                img = np.expand_dims(img, 0)
                img = np.expand_dims(img, 0)
            image = 0
            tiff_patch.append(img)
            delta = datetime.datetime.now() - start
            print(f"{item_index+1}/{size} took {delta}, ETA for this patch: {delta * (items + size - item_index)}", end="\r", flush=True)
        else:
            img = np.zeros_like(tiff_patch[0])
            tiff_patch.append(img)
    tiff_patch = np.array(tiff_patch)
    tiff_patch = np.swapaxes(tiff_patch, 0, 1)
    tiff_patch = np.swapaxes(tiff_patch, 1, 2)
    tiff_patch = np.swapaxes(tiff_patch, 2, 3)
    tiff_patch = np.swapaxes(tiff_patch, 3, 4)
    print(f"TIFF PATCH {tiff_patch.shape}")
    tiff_patch = tiff_patch.astype(np.float16)
    return tiff_patch

torch.cuda.init()
torch.cuda.set_device(0)

settings = read_meta_dict("./", "predict")
prediction_path = settings["paths"]["output_prediction_path"]
net, inferer = 0, 0
size = [256, 256, 256]
NIFTI = os.listdir(settings["paths"]["input_raw_path"])[0][-3:] == "nii"

with torch.no_grad():
    print(f"NIFTI {NIFTI}")
    channel = "C01" if "C01" in settings["paths"]["input_raw_path"] else "C02"
    nii_path = settings["paths"]["input_raw_path"]
    bg_path = settings["paths"]["input_bg_path"]


    output_binary = settings["paths"]["output_prediction_path"] + f"binary_{size[-1]}/"
    output_network= settings["paths"]["output_prediction_path"] + f"network_{size[-1]}/"
    if not os.path.exists(output_binary):
        os.mkdir(output_binary)
        os.mkdir(output_network)
    else:
        print(f"{output_binary} exists, skipping..")

    # TODO Load Niftis with for Loop, tiffs with size
    if NIFTI:
        # If we have niftis in the input path, we use them
        for item in os.listdir(nii_path):
            if not os.path.exists(f"{output_network}/{item.replace('.nii','_' + channel + '.nii')}"):

                img = load_nifti(settings, item)

                if img.shape[-1] < size[-1]:
                    size[-1] = img.shape[-1]
                
                if net == 0:
                    net, inferer = load_inferer(settings, size)


                print("Starting inference")
                pred = inferer(inputs=img, network=net, SIGMOID=False)
                print(f"{type(pred)} {pred.dtype}")
                pred = pred.numpy()
                pred = np.squeeze(np.squeeze(pred))
                print(f"Pred {type(pred)} {pred.dtype}")

                if pred.shape[-1] > pred.shape[0]:
                    pred = np.swapaxes(pred, -1, 1)
                    pred = np.swapaxes(pred, 0, -1)
                print("Writing network output")

                write_nifti(f"{output_network}/{item.replace('.nii','_' + channel + '.nii')}", pred.astype(np.float32))
                print(f"Binarizing {pred.shape}")
                pred[pred > 0.5] = 1.0
                pred[pred <= 0.5] = 0
                pred = pred.astype(np.uint8)
                print(f"bin {type(pred)} {pred.dtype} {np.max(pred)}")
                write_nifti(f"{output_binary}/{item.replace('.nii','_' + channel + '.nii')}", pred)
                print(f"Done {item}")
            else:
                print(f"{output_network}/{item.replace('.nii','_' + channel + '.nii')} exists, skipping...")
    else:
        print(f"range 0 {len(os.listdir(nii_path))} {size[-1]}")
        for items in range(0, len(os.listdir(nii_path)), size[-1]):
            if not os.path.exists(f"{items}_{items+size[-1]}.nii.gz"):

                img = load_tiff(settings, items, size[-1])
                if img.shape[-1] < size[-1]:
                    size[-1] = img.shape[-1]
                
                if net == 0:
                    net, inferer = load_inferer(settings, size)


                print("Starting inference")
                pred = inferer(inputs=img, network=net, SIGMOID=False)
                print(f"{type(pred)} {pred.dtype}")
                pred = pred.numpy()
                pred = np.squeeze(np.squeeze(pred))
                print(f"Pred {type(pred)} {pred.dtype}")

                if pred.shape[-1] > pred.shape[0]:
                    pred = np.swapaxes(pred, -1, 1)
                    pred = np.swapaxes(pred, 0, -1)
                print("Writing network output")
                #TODO finalize output name

                write_nifti(f"{output_network}/{items}_{items+size[-1]}", pred.astype(np.float32))
                print(f"Binarizing {pred.shape}")
                pred[pred > 0.5] = 1.0
                pred[pred <= 0.5] = 0
                pred = pred.astype(np.uint8)
                print(f"bin {type(pred)} {pred.dtype} {np.max(pred)}")
                write_nifti(f"{output_binary}/{items}_{items+size[-1]}", pred)
                print(f"Done {items} - {items + size[-1]}")
            else:
                print(f"{items}_{items+size[-1]}.nii.gz exists, skipping...")



