import numpy as np
from cc3d import connected_components


def calc_volumetric_score(pred, target):
    la = lambda a, b: np.logical_and(a, b)
    no = lambda a: np.logical_not(a)

    tp = la(pred, target).sum()
    tn = la(no(pred), no(target)).sum()
    fp = la(pred, no(target)).sum()
    fn = la(no(pred), target).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn)
    vs = 1 - abs(fn - fp) / (2*tp + fp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_dice = (2*tp) / (2*tp + fp +fn)
    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        f1_dice = 1
    return precision, recall, vs, accuracy, f1_dice

def calc_statistics(pred, target):
    """Calculate test metrices
    Args:
        - pred      (torch.tensor)  : The predicted values in binary (0,1) format
        - target    (torch.tensor)  : The target value in a binary(0,1) format
    Returns:
        - tp        (int)           : True poitives
        - tn        (int)           : True negatives
        - fp        (int)           : False positives
        - fn        (int)           : False negatives
    """
    # pred = np.asarray(pred).astype(bool)
    # target = np.asarray(target).astype(bool)

    la = lambda a, b: np.logical_and(a, b)
    no = lambda a: np.logical_not(a)

    tp = la(pred, target).sum()
    tn = la(no(pred), no(target)).sum()
    fp = la(pred, no(target)).sum()
    fn = la(no(pred), target).sum()
    return tp, tn, fp, fn

def calc_metrices_stats(m_list):
    """Calculate test statistics on a given list of binary metrices
    Args:
        - m_lis     (list of ints)  : A list containing true positives, true negatives, 
                                        false positives, false negatives in this order
    Returns:
        - precision (double)        : Precision
        - recall    (double)        : Recall
        - vs        (double)        : Volumetric Similarity
        - accuracy  (double)        : Accuracy score
        - f1_dice   (double)        : Dice/F1-Score
    """
    tp, tn, fp, fn = m_list[0], m_list[1], m_list[2], m_list[3]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn)
    vs = 1 - abs(fn - fp) / (2*tp + fp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_dice = (2*tp) / (2*tp + fp +fn)
    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        f1_dice = 1
    return precision, recall, vs, accuracy, f1_dice

def create_overlay(pred, target):
    pred    = np.squeeze(pred)
    target  = np.squeeze(target)
    target[target > 1] = 1 
    pred[pred > 1] = 1 

    res = target*2 + pred
    res[res==2] = 10
    res[res==3] = 2 
    res[res==10]= 3
    return res

def get_patch_overlap(pred_patch, target_patch):
    """Test for overlap in two patches
    Returns:
    true positive, true negative, false positive, false negative
    """
    pred = get_blobs(pred_patch)
    target = get_blobs(target_patch)

    # Create a list of all available targets
    # Could work better with a dict
    t_s = set()
    t_d = {}
    for i in target:
        t_s.add(i["id"])
        t_d[i["id"]] = i["points"]

    tp, fp, fn = 0, 0, 0
    for blob_pred in pred:
        hits_i = 0
        pred_l = blob_pred["points"]
        pred_id = blob_pred["id"]
        for target_id in t_s:
            target_l = t_d[target_id]
            if test_overlap(pred_l, target_l) and hits_i == 0:
                hits_i = 1
                t_s.remove(target_id)
                break
        if hits_i == 0:
            fp += 1
        else:
            tp += 1
    fn = len(t_s)
    return tp, fp, fn

def get_blobs(volume):
    '''
    bloblist = get_blobs(volume)

    This function returns a list of dictionaries, in which each dictionary
    represents one blob in the given 'searchvolume'. A blob is defined as
    a set of connected points. The 'searchvolume' is expected to be a
    p-dimensional Numpy array of zero and non-zero values. All neighboring
    non-zero values will be treated as connected points, i.e. a blob.

    Each blob dictionary in the list 'blobs' has the following entries:
        * blob['id'] - Number of blob in searchvolume, starting with 0
        * blob['points'] - List of points in this blob. Each point is a 1D Numpy array with p coordinates (one per dimension)
        * blob['offset'] - Offset from bounding box to global coordinate system
        * blob['boundingbox'] - Size of 3D box enclosing the entire blob
        * blob['volume'] - Number of voxels in blob
        * blob['CoM'] - Center of Mass (within bounding box)
        * blob['max_dist'] - Largest distance between any two points of blob
        * blob['characterization'] - Dict of further characterizations

    NB: The runtime of this function is largely independent of size of the
    searchvolume, but grows with the number as well as the size of blobs.
    For busy 3D volumes, get_blobs_fast() can >100 times faster (but might
    falsly merge two almost-overlapping points in rare cases)

    This version is using an external library for connected components (26-connectedness)
    that was not available at the beginning of Project Leo. Please see:
        https://github.com/seung-lab/connected-components-3d
    '''
    # print("Performing cca...")
    #todo wieder raus
    if np.amax(volume) == 1:
        volume = volume.astype(np.bool)
        labeled_volume = connected_components(volume)
    else:
        print("\tUsing predefined labels...")
        labeled_volume = volume
    labels = [ x for x in np.unique(labeled_volume) if x != 0 ]
    bloblist = []
    for label in labels:
        allpoints = np.asarray(np.where(labeled_volume == label)).T.tolist() # returns list of pointers; slow for large vols
        blob = {}
        blob['id'] = len(bloblist)
        blob['points'] = allpoints
        bloblist.append(blob)
    return bloblist

def test_overlap(pointlist1,pointlist2):
    '''
    test_result = test_overlap(pointlist1,pointlist2)

    Checks whether any point in pointlist1 is also in pointlist2
    '''
    # First check trivial cases with fast methods
    pa1 = np.asarray(pointlist1)
    pa2 = np.asarray(pointlist2)
    ndims = pa1.shape[1]
    if(np.min(pa1[:,0]) > np.max(pa2[:,0])): return False
    if(np.min(pa2[:,0]) > np.max(pa1[:,0])): return False
    if(np.min(pa1[:,1]) > np.max(pa2[:,1]) and ndims >= 2): return False
    if(np.min(pa2[:,1]) > np.max(pa1[:,1]) and ndims >= 2): return False
    if(np.min(pa1[:,2]) > np.max(pa2[:,2]) and ndims >= 3): return False
    if(np.min(pa2[:,2]) > np.max(pa1[:,2]) and ndims >= 3): return False
    # Then check point by point overlap
    test_result = False
    maxind = len(pointlist1) - 1
    ind = 0
    while(test_result == False and ind <= maxind):
        if(pointlist1[ind] in pointlist2): test_result = True
        ind += 1
    return test_result

def create_f1_overlay(pred, target):
    #TODO
    # Use blobanalysis to get patch ids and corresponding connected component
    # Mark missed blobs/correrctly identified blobs etc
    pass
