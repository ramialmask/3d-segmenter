import numpy as np

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

def calc_metrices(pred, target):
    """Calculate test statistics
    Args:
        - pred      (torch.tensor)  : The predicted values in binary (0,1) format
        - target    (torch.tensor)  : The target value in a binary(0,1) format
    Returns:
        - precision (double)        : Precision
        - recall    (double)        : Recall
        - vs        (double)        : Volumetric Similarity
        - accuracy  (double)        : Accuracy score
        - f1_dice   (double)        : Dice/F1-Score
    """
    pred = np.hstack(pred)
    target = np.hstack(target)
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        precision = 1
        recall = 1
        vs = 1
        accuracy = 1
        f1_dice = 1
    else:
        tp, tn, fp, fn = calc_statistics(pred, target)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn)
        vs = 1 - abs(fn - fp) / (2*tp + fp + fn)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1_dice = (2*tp) / (2*tp + fp +fn)
    return precision, recall, vs, accuracy, f1_dice

def calc_metrics_stats(m_list):
    tp, tn, fp, fn = m_list[0], m_list[1], m_list[2], m_list[3]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn)
    vs = 1 - abs(fn - fp) / (2*tp + fp + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    f1_dice = (2*tp) / (2*tp + fp +fn)
    return precision, recall, vs, accuracy, f1_dice
