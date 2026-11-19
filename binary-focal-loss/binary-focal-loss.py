import math
import numpy as np

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    # Write code here
    # pass
    pt = []
    for i in range(len(targets)):
        if targets[i] == 1:
            pt.append(predictions[i])
        else:
            pt.append(1-predictions[i])

    pt = np.array(pt)
    focal_loss = -1*(alpha*(1 - pt)**gamma)*np.log(pt)
    return float(np.mean(focal_loss))