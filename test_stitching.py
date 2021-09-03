import os
from matplotlib import pyplot as plt
from util import *
from loaders import *
from dataset.dataset_2D import cut_volume, prepare_lists

#TODO
settings = read_meta_dict("./","train")
input_list = os.listdir(settings["paths"]["input_gt_path"])
test_loader, dataset     = get_loader(settings, input_list, train=True, testing=True)

item_dict = {}
reconstructed_patches = []

for item in test_loader:
    volume       = item["volume"].numpy()
    item_name = item["name"][0]
    item_z = int(item_name.split("$")[0])
    item_image = item_name.split("$")[1]
    
    # print(item_name)
    # if item_name == "33$patchvolume_695_5.nii.gz":
    #     plt.imshow(volume[0,0,:,:])
    #     plt.show()
    # if item_name == "34$patchvolume_695_5.nii.gz":
    #     plt.imshow(volume[0,0,:,:])
    #     plt.show()
    #     exit()
    
    if item_image in item_dict.keys():
        item_dict[item_image].append((item_z, volume.squeeze().squeeze()))
    else:
        item_dict[item_image] = [(item_z, volume.squeeze().squeeze())]

orig_shape, orig_type, mb_size = dataset.original_information()
if orig_shape[0] > mb_size:
    intermediate_patches    = reconstruct_patches_2d(item_dict, dataset)
    reconstructed_patches   = dict_to_patches(intermediate_patches, orig_shape)
else:
    reconstructed_patches   = reconstruct_patches_2d(item_dict, dataset)

for item_name, reconstructed_prediction in reconstructed_patches:
    item_save_path      = f"/media/10TB/Projects/cFos/segmentation/test/{item_name}"
    write_nifti(item_save_path, reconstructed_prediction)            
# Read, cut into z patches
# orig_shape, orig_type = 0, 0
# for item in os.listdir(input_path):
#     image = read_nifti(input_path + item)
#     orig_shape, orig_type = image.shape, image.dtype
#     if image.shape[0] > 100:
#         cut_images = cut_volume(image, item, 100)
#         for i, (cut_name, cut_image) in enumerate(cut_images):
#             prepare_lists(settings,\
#                     cut_name,\
#                     cut_image,\
#                     image_list,\
#                     name_list,\
#                     cut_gt_image,\
#                     gt_list,\
#                     cut_bg_image)

# Reconstruct

# Save

