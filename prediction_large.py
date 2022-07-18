import torch

from sliding_window_inferrer import SlidingWindowInferer
from loaders import get_prediction_loader, read_meta_dict
from util import *
from monai.networks.nets import unet
from collections import deque


#TODO /media/rami/18TB/LNP/Organs by Izabela/Niftis/C01
#TODO anderes Netz

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
                    overlap=0.5,
                    mode="constant",
                    padding_mode="replicate",
                    device=torch.device("cpu")
                )

    net.eval()
    return net, inferer

def norm(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

torch.cuda.init()
torch.cuda.set_device(0)

settings = read_meta_dict("./", "predict")
prediction_path = settings["paths"]["output_prediction_path"]
net, inferer = 0, 0

with torch.no_grad():
    channel = "C01" if "C01" in settings["paths"]["input_raw_path"] else "C02"
    nii_path = settings["paths"]["input_raw_path"]
    bg_path = settings["paths"]["input_bg_path"]


    output_binary = settings["paths"]["output_prediction_path"] + "binary/"
    output_network= settings["paths"]["output_prediction_path"] + "network/"
    if not os.path.exists(output_binary):
        os.mkdir(output_binary)
        os.mkdir(output_network)
    else:
        print(f"{output_binary} exists, skipping..")

    for item in os.listdir(nii_path):
        if not os.path.exists(f"{output_network}/{item.replace('.nii','_' + channel + '.nii')}"):
            # Load FG
            item_path = os.path.join(nii_path, item)
            print(f"loading {item_path}")
            image = nib.load(item_path)
            # print(f"{image.get_data_dtype()}")
            # print(f"Header:\n{image.header}\n")
            # print(f"GetFdata {type(image)}")
            # image = np.array(image.get_data())
            image = np.asanyarray(image.dataobj)
            # print(f"As F16 {type(image)} {image.dtype}")
            image = image.astype(np.float16)
            # print(f"Squeezing {type(image)} {image.dtype}")
            image = image.squeeze()
            # print(f"Swapaxes {type(image)} {image.dtype}")
            image = np.swapaxes(image, 0, 1)
            # print(f"Expand Dims {type(image)} {image.dtype}")
            # image = np.expand_dims(image, 0)
            print(f"Norming image {np.amin(image)} {np.amax(image)} {image.dtype}")
            image = norm(image)
            print(f"After norming image {np.amin(image)} {np.amax(image)} {image.dtype}")

            # Load BG
            if int(settings["dataloader"]["num_channels"]) == 2:
                item_bg_path = os.path.join(bg_path, item)
                print(f"loading {item_bg_path}")
                image_bg = nib.load(item_bg_path)
                # image_bg = np.array(image_bg.get_data())
                image_bg = np.asanyarray(image_bg.dataobj)
                image_bg = image_bg.astype(np.float16)
                image_bg = image_bg.squeeze()
                image_bg = np.swapaxes(image_bg, 0, 1)
                # image_bg = np.expand_dims(image_bg, 0)
                image_bg = norm(image_bg)

                print("Concatinating")
                img = deque()#
                img.append(image)
                img.append(image_bg)
                img = np.array(img)
            # image = [np.concatenate((image, image_bg), 0)]
                image_bg = 0
                img = np.expand_dims(img, 0)
            else:
                img = image
                img = np.expand_dims(img, 0)
                img = np.expand_dims(img, 0)
            image = 0
            print(img.shape)
            print(f"{item} {type(img)} {img.shape}")

            size = [64, 64, 64]
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

