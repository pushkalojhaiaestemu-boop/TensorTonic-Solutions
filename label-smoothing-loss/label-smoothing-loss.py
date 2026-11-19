import math
import numpy as np

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    # Write code here
    N = len(predictions)
    qi = []
    for i in range(N):
        if i == target:
            qi.append(1-epsilon + (epsilon/N))
        else:
            qi.append(epsilon/N)


    Loss = -1*np.sum(qi*np.log(np.array(predictions)))
    return Loss
    