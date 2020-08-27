import os
import numpy as np
from cc3d import connected_components as cc
from util_pkg.filehandling import read_nifti, write_nifti
"""
This script is used to classify cell annotations into three classes
based on their relative brightness compared to local background:
    Class 1:    cells which may be background noise
    Class 2:    cells which are most likely cells
    Class 3:    cells which may be agglomeration of cFos
"""

pj = lambda x, y : os.path.join(x, y)

def _get_area(patch_input, center, area=25):
    """Get the area around a given point
    """
    patch = patch_input.copy()
    p_max = patch.shape[0]
    x, y, z = center[0][0], center[1][0], center[2][0]

    xx, xx_ = x-area,x+area
    yy, yy_ = y-area,y+area
    zz, zz_ = z-area,z+area

    if xx < 0:xx= 0
    if yy < 0:yy= 0
    if zz < 0:zz= 0

    if xx_ > p_max:xx_ = p_max
    if yy_ > p_max:yy_ = p_max
    if zz_ > p_max:zz_ = p_max

    return patch[xx:xx_,yy:yy_,zz:zz_]

def _get_bb(patch):
    """Get the bounding box for a cell in a binary matrix
    """
    a = np.where(patch > 0)
    bb = ((np.amin(a[0]), np.amin(a[1]), np.amin(a[2])), (np.amax(a[0]), np.amax(a[1]), np.amax(a[2])))
    return bb

def find_center(raw_input, patch_input, return_cc = False):
    """Isolate the single cc in labels, cut out in raw, set the brightest point as center point
    Args:
        raw     : Raw input patch
        patch   : Binary annotation mask
    Returns:
        result  : Numpy array consisting of the centers for each connected component
    """
    raw = raw_input.copy()
    patch = patch_input.copy()
    labels = cc(patch.astype(np.uint8))
    result = np.zeros_like(raw)
    max_l = np.amax(labels)
    for i in range(1, max_l):
        sub_label = np.copy(labels)
        sub_label[sub_label != i] = 0
        if (np.count_nonzero(sub_label > 0)) > 4:
            bb = _get_bb(sub_label)
            sub_label = sub_label[bb[0][0]:bb[1][0]+1,bb[0][1]:bb[1][1]+1,bb[0][2]:bb[1][2]+1]
            sub_raw = raw[bb[0][0]:bb[1][0]+1,bb[0][1]:bb[1][1]+1,bb[0][2]:bb[1][2]+1]
            sub_label[sub_label > 0] = 1
            sub_raw *= sub_label
            center_value = np.amax(sub_raw)
            center_coords = np.where(sub_raw == center_value)
            result[bb[0][0] + center_coords[0], bb[0][1] + center_coords[1], bb[0][2] + center_coords[2]] = i
    if return_cc:
        return result, labels
    else:
        return result

def classify_patch(patch_gt, patch_raw, classes=[1,2,3]):
    # Get the labels and their centerpoints
    patch_center, patch_labels      = find_center(patch_raw, patch_gt,True)
    # Empty result matrix
    patch_class = np.zeros_like(patch_gt)

    for center in range(1, np.max(patch_labels)):
        # Coords of the cell _center_ to determine the ratio
        center_coords = np.where(patch_center == center)

        # Coords of the entire cell to color the entire annoation
        label_coords = np.where(patch_labels == center)

        if len(center_coords[0] > 0):
            # Value is defined by center value, whole label value would also work
            center_value = patch_raw[center_coords][0]

            # Get the relevant/neighbourhood
            area_raw = _get_area(patch_raw, center_coords)
            area_gt  = _get_area(patch_gt, center_coords)

            # Remove all cells to only get the background
            area_raw[np.where(area_gt > 0)] = 0

            # Select max value of the background
            background_max      = np.max(area_raw[area_raw > 0])

            # Calculate the relevant ratio
            max_ratio = center_value / background_max

            # Classify based on figures obtained with classify_cells_brightness.py
            cell_class = 1
            if 0.86 < max_ratio < 2.5:
                cell_class = 2
            elif max_ratio > 2.5:
                cell_class = 3
            if cell_class in classes:
                # Write into the result matrix
                patch_class[label_coords] = 1#cell_class

    return patch_class
