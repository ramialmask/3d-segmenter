from util_pkg.filehandling import read_nifti
from classify_patches import classify_patch as cp
import numpy as np
a = read_nifti("/home/ramial-maskari/Documents/cFos/input/raw_small/patchvolume_695_0.nii.gz")
b = read_nifti("/home/ramial-maskari/Documents/cFos/input/gt_v3/patchvolume_695_0.nii.gz")
x = cp(b,a)
y = cp(b,a,[1,2])
z = cp(b,a,[2])
print(np.sum(x))
print(np.sum(y))
print(np.sum(z))
