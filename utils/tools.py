import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment



def ordered_cmat(labels, pred, label_assignment = True, return_ri_ci = False):
    """
    Compute the confusion matrix and accuracy corresponding to the best cluster-to-class assignment.

    :param labels: Label array
    :type labels: np.array
    :param pred: Predictions array
    :type pred: np.array
    :return: Accuracy and confusion matrix
    :rtype: Tuple[float, np.array]
    """
    cmat = confusion_matrix(labels, pred)
    if label_assignment:
        ri, ci = linear_sum_assignment(-cmat)
        ordered = cmat[np.ix_(ri, ci)]
    else:
        ordered = cmat
        ri = ci = None
        
    acc = np.sum(np.diag(ordered))/np.sum(ordered)
    if return_ri_ci:
        return acc, ordered, (ri, ci)
    else:
        return acc, ordered


def correct_predictions(pred, ri, ci):
    idxs = [np.where(pred == _) for _ in ri]
    for idx, cor_val in zip(idxs, ci):
        pred[idx] = cor_val
    return pred