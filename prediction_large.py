import torch

from monai.inferers import SlidingWindowInferer
from loaders import get_prediction_loader, read_meta_dict
from util import *
from models.unet_3d_oliver import Unet3D

torch.cuda.init()
torch.cuda.set_device(0)

settings = read_meta_dict("./", "predict")
prediction_path = settings["paths"]["output_prediction_path"]

net = Unet3D(in_dim=2)
net.load_model("/media/rami/4E7F10B756ABEBBE/Antibody project/Neuron segmentation/input/models/2Channel WBCE Baseline Unet3D Augmentation Global normalization_0_0_99.dat")
net = net.cuda()

size = 101
inferer = SlidingWindowInferer(roi_size=(size,size,size))

net.eval()
with torch.no_grad():
    nii_path = settings["paths"]["input_raw_path"]
    bg_path = settings["paths"]["input_bg_path"]
    for item in os.listdir(settings["paths"]["input_raw_path"]):
        # Load FG
        item_path = os.path.join(nii_path, item)
        image = nib.load(item_path)
        image = np.array(image.get_fdata())
        image = image.squeeze()
        image = np.swapaxes(image, 0, 1)
        image = np.expand_dims(image, 0)

        # Load BG
        item_bg_path = os.path.join(bg_path, item)
        image_bg = nib.load(item_bg_path)
        image_bg = np.array(image_bg.get_fdata())
        image_bg = image_bg.squeeze()
        image_bg = np.swapaxes(image_bg, 0, 1)
        image_bg = np.expand_dims(image_bg, 0)
        image = [np.concatenate((image, image_bg), 0)]
        volume = torch.tensor(image).float()
        print(f"{item}")

        volume = volume.cuda()
        pred = inferer(inputs=volume, network=net)
        pred = pred.detach().cpu().numpy()
        pred[pred > 0.5] = 1.0
        pred[pred <= 0.5] = 0
        write_nifti(f"{prediction_path}/{name}", pred)

